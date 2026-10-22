import { getSql, type Sql } from "@/lib/db";
import {
  scoreInputSchema,
  trainInputSchema,
  type FeatureRow,
} from "./contract";
import { scoreFeatures, trainSynthetic } from "./engine";
import type { ForestModel } from "./forest";

export type ModelSummary = {
  id: number;
  versionLabel: string;
  contractVersion: string;
  contamination: number;
  nEstimators: number;
  seed: number;
  trainDays: string;
  testDays: string;
  nTrain: number;
  nTest: number;
  nTestInjected: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  rocAuc: number | null;
  testFlagRate: number;
  threshold: number;
  status: "active" | "archived" | "candidate";
  createdAt: string;
};

export type QueueRow = {
  id: number;
  transactionId: string;
  customerId: string;
  amount: number;
  hour: number;
  velocity24h: number;
  distanceKm: number;
  newDevice: number;
  eventDay: number;
  anomalyScore: number;
  flagged: boolean;
  reviewReasons: string;
  injectedScenario: boolean;
  split: "train" | "test";
  decision: "clear" | "hold" | "escalate" | null;
  reasonCode: string | null;
};

export type LiveScore = FeatureRow & {
  anomalyScore: number;
  flagged: boolean;
  reviewReasons: string;
};

function num(value: unknown): number {
  return typeof value === "number" ? value : Number(value);
}

function nullableNum(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const next = num(value);
  return Number.isFinite(next) ? next : null;
}

export type AuditEvent = {
  id: number;
  kind: "model_registered" | "review_recorded" | "batch_scored";
  modelId: number | null;
  summary: string;
  createdAt: string;
};

function flag(value: unknown): boolean {
  return value === true || value === 1 || value === "t" || value === "true";
}

function mapAudit(row: Record<string, unknown>): AuditEvent {
  return {
    id: num(row.id),
    kind: String(row.kind) as AuditEvent["kind"],
    modelId: row.model_id === null || row.model_id === undefined ? null : num(row.model_id),
    summary: String(row.summary),
    createdAt: String(row.created_at),
  };
}

async function writeAudit(
  sql: Sql,
  kind: AuditEvent["kind"],
  modelId: number | null,
  summary: string,
): Promise<AuditEvent> {
  const rows = await sql.query<Record<string, unknown>>(
    `insert into audit_events (kind, model_id, summary)
     values ($1, $2, $3)
     returning id, kind, model_id, summary,
       to_char(created_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as created_at`,
    [kind, modelId, summary.slice(0, 180)],
  );
  return mapAudit(rows[0] ?? {});
}

async function listAudit(sql: Sql): Promise<AuditEvent[]> {
  const rows = await sql.query<Record<string, unknown>>(
    `select id, kind, model_id, summary,
            to_char(created_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as created_at
     from audit_events
     order by id desc
     limit 20`,
  );
  return rows.map(mapAudit);
}

function mapModel(row: Record<string, unknown>): ModelSummary {
  return {
    id: num(row.id),
    versionLabel: String(row.version_label),
    contractVersion: String(row.contract_version),
    contamination: num(row.contamination),
    nEstimators: num(row.n_estimators),
    seed: num(row.seed),
    trainDays: String(row.train_days),
    testDays: String(row.test_days),
    nTrain: num(row.n_train),
    nTest: num(row.n_test),
    nTestInjected: num(row.n_test_injected),
    precision: nullableNum(row.precision_at_flag),
    recall: nullableNum(row.recall_at_flag),
    f1: nullableNum(row.f1),
    rocAuc: nullableNum(row.roc_auc),
    testFlagRate: num(row.test_flag_rate),
    threshold: num(row.threshold),
    status: String(row.status) as ModelSummary["status"],
    createdAt: String(row.created_at),
  };
}

const MODEL_COLUMNS = `
  id, version_label, contract_version, contamination, n_estimators, seed,
  train_days, test_days, n_train, n_test, n_test_injected,
  precision_at_flag, recall_at_flag, f1, roc_auc, test_flag_rate, threshold, status,
  to_char(created_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as created_at
`;

async function listModels(sql: Sql): Promise<ModelSummary[]> {
  const rows = await sql.query<Record<string, unknown>>(
    `select ${MODEL_COLUMNS} from model_versions order by id desc limit 12`,
  );
  return rows.map(mapModel);
}

async function queueFor(sql: Sql, modelId: number): Promise<QueueRow[]> {
  const rows = await sql.query<Record<string, unknown>>(
    `select s.id, s.transaction_id, s.customer_id, s.amount, s.hour, s.velocity_24h,
            s.distance_km, s.new_device, s.event_day, s.anomaly_score, s.flagged,
            s.review_reasons, s.injected_scenario, s.split,
            r.decision, r.reason_code
     from scored_transactions s
     left join lateral (
       select decision, reason_code
       from reviews
       where scored_transaction_id = s.id
       order by id desc
       limit 1
     ) r on true
     where s.model_version_id = $1
     order by s.anomaly_score desc`,
    [modelId],
  );
  return rows.map((row) => ({
    id: num(row.id),
    transactionId: String(row.transaction_id),
    customerId: String(row.customer_id),
    amount: num(row.amount),
    hour: num(row.hour),
    velocity24h: num(row.velocity_24h),
    distanceKm: num(row.distance_km),
    newDevice: num(row.new_device),
    eventDay: num(row.event_day),
    anomalyScore: num(row.anomaly_score),
    flagged: flag(row.flagged),
    reviewReasons: String(row.review_reasons),
    injectedScenario: flag(row.injected_scenario),
    split: String(row.split) as "train" | "test",
    decision: row.decision ? (String(row.decision) as QueueRow["decision"]) : null,
    reasonCode: row.reason_code ? String(row.reason_code) : null,
  }));
}

async function insertScores(
  sql: Sql,
  modelId: number,
  rows: ReturnType<typeof trainSynthetic>["rows"],
) {
  const width = 14;
  const chunk = 80;
  for (let offset = 0; offset < rows.length; offset += chunk) {
    const slice = rows.slice(offset, offset + chunk);
    const params: unknown[] = [];
    const tuples = slice.map((row) => {
      const base = params.length;
      params.push(
        modelId,
        row.transaction_id,
        row.customer_id,
        row.amount,
        row.hour,
        row.velocity_24h,
        row.distance_km,
        row.new_device,
        row.event_day,
        row.anomaly_score,
        row.flagged,
        row.review_reasons,
        row.injected_scenario,
        row.split,
      );
      return `(${Array.from({ length: width }, (_, index) => `$${base + index + 1}`).join(",")})`;
    });
    await sql.query(
      `insert into scored_transactions (
         model_version_id, transaction_id, customer_id, amount, hour, velocity_24h,
         distance_km, new_device, event_day, anomaly_score, flagged, review_reasons,
         injected_scenario, split
       ) values ${tuples.join(",")}`,
      params,
    );
  }
}

export async function loadWorkspace() {
  const sql = await getSql();
  const models = await listModels(sql);
  const active = models.find((model) => model.status === "active") ?? null;
  const queue = active ? await queueFor(sql, active.id) : [];
  const events = await listAudit(sql);
  return { models, activeId: active?.id ?? null, queue, events };
}

export async function loadQueue(modelId: number) {
  const sql = await getSql();
  const found = await sql.query<{ id: number }>(
    "select id from model_versions where id = $1",
    [modelId],
  );
  if (found.length === 0) throw new Error("That model version is not in the registry.");
  return queueFor(sql, modelId);
}

export async function registerModel(input: unknown) {
  const parsed = trainInputSchema.parse(input);
  const trained = trainSynthetic(parsed);
  const artifact = JSON.stringify(trained.model);
  if (artifact.length > 4_000_000) throw new Error("Model artifact is too large to store.");
  const sql = await getSql();
  const inserted = await sql.query<{ id: number }>(
    `insert into model_versions (
       version_label, contract_version, contamination, n_estimators, seed,
       train_days, test_days, n_train, n_test, n_test_injected,
       precision_at_flag, recall_at_flag, f1, roc_auc, test_flag_rate,
       threshold, amount_p95, status, artifact
     ) values (
       $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,'candidate',$18
     ) returning id`,
    [
      trained.label,
      trained.contractVersion,
      parsed.contamination,
      parsed.trees,
      parsed.seed,
      trained.trainDays,
      trained.testDays,
      trained.nTrain,
      trained.nTest,
      trained.metrics.positives,
      trained.metrics.precision,
      trained.metrics.recall,
      trained.metrics.f1,
      trained.metrics.rocAuc,
      trained.metrics.flagRate,
      trained.model.threshold,
      trained.amountP95,
      artifact,
    ],
  );
  const id = num(inserted[0]?.id);
  await insertScores(sql, id, trained.rows);
  const current = await sql.query<{ id: number }>(
    "select id from model_versions where status = 'active'",
  );
  for (const row of current) {
    await sql.query("update model_versions set status = 'archived' where id = $1", [num(row.id)]);
  }
  await sql.query("update model_versions set status = 'active' where id = $1", [id]);
  const models = await listModels(sql);
  const queue = await queueFor(sql, id);
  const auc = trained.metrics.rocAuc === null ? "n/a" : trained.metrics.rocAuc.toFixed(2);
  await writeAudit(sql, "model_registered", id, `registered ${trained.label} v${id} auc ${auc}`);
  const events = await listAudit(sql);
  return { models, activeId: id, queue, events };
}

export async function scoreUpload(input: unknown): Promise<{
  modelId: number;
  versionLabel: string;
  event: AuditEvent;
  rows: LiveScore[];
}> {
  const parsed = scoreInputSchema.parse(input);
  const sql = await getSql();
  const found = await sql.query<Record<string, unknown>>(
    `select id, version_label, artifact, amount_p95
     from model_versions where status = 'active' order by id desc limit 1`,
  );
  const row = found[0];
  if (!row) throw new Error("Register a model before scoring a file. Uploads are checked against the active version.");
  let model: ForestModel;
  try {
    model = JSON.parse(String(row.artifact)) as ForestModel;
  } catch {
    throw new Error("The active model artifact could not be read.");
  }
  const scored = scoreFeatures(model, parsed.rows, num(row.amount_p95));
  const flagged = scored.filter((item) => item.flagged).length;
  const modelId = num(row.id);
  const event = await writeAudit(
    sql,
    "batch_scored",
    modelId,
    `scored ${scored.length} rows, ${flagged} flagged, on v${modelId}`,
  );
  return {
    modelId,
    versionLabel: String(row.version_label),
    event,
    rows: scored.map((item) => ({
      amount: item.amount,
      hour: item.hour,
      velocity_24h: item.velocity_24h,
      distance_km: item.distance_km,
      new_device: item.new_device,
      anomalyScore: item.anomaly_score,
      flagged: item.flagged,
      reviewReasons: item.review_reasons,
    })),
  };
}

export async function saveReview(input: {
  scoredTransactionId: number;
  decision: "clear" | "hold" | "escalate";
  reasonCode: "amount" | "velocity" | "distance" | "device" | "overnight" | "pattern";
}) {
  const sql = await getSql();
  const found = await sql.query<{ id: number }>(
    "select id from scored_transactions where id = $1",
    [input.scoredTransactionId],
  );
  if (found.length === 0) throw new Error("That transaction is not in the saved queue.");
  await sql.query(
    `insert into reviews (scored_transaction_id, decision, reason_code)
     values ($1, $2, $3)`,
    [input.scoredTransactionId, input.decision, input.reasonCode],
  );
  const event = await writeAudit(
    sql,
    "review_recorded",
    null,
    `review ${input.scoredTransactionId} ${input.decision} ${input.reasonCode}`,
  );
  return {
    scoredTransactionId: input.scoredTransactionId,
    decision: input.decision,
    reasonCode: input.reasonCode,
    event,
  };
}
