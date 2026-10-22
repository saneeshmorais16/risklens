import { spawnSync } from "node:child_process";
import { investigateInputSchema } from "./contract";

export type Investigation = {
  report: string;
  recommend_review: boolean;
  action: "none";
  evidence: { source: string; excerpt: string; score: number }[];
  steps: { thought: string; action: string }[];
};

export function investigateCase(input: unknown): Investigation {
  const facts = investigateInputSchema.parse(input);
  const payload = {
    transaction_id: facts.transactionId,
    customer_id: facts.customerId,
    amount: facts.amount,
    hour: facts.hour,
    velocity_24h: facts.velocity,
    distance_km: facts.distanceKm,
    new_device: facts.newDevice ? 1 : 0,
    score: facts.score,
    note: facts.note,
    history: facts.history.map((row) => ({
      transaction_id: row.transactionId,
      customer_id: row.customerId,
      amount: row.amount,
      hour: row.hour,
    })),
  };
  const ran = spawnSync("python3", ["-m", "agents.react_agent"], {
    input: JSON.stringify(payload),
    cwd: process.cwd(),
    encoding: "utf8",
    timeout: 8000,
    env: { ...process.env, PYTHONPATH: process.cwd() },
  });
  if (ran.error || ran.status !== 0) {
    throw new Error("The investigation did not finish. Try again.");
  }
  const body = JSON.parse(ran.stdout) as {
    report?: string;
    recommend_review?: boolean;
    action?: string;
    evidence?: { source?: string; excerpt?: string; score?: number }[];
    steps?: { thought?: string; action?: string }[];
  };
  if (body.action !== "none" || !body.report) throw new Error("The investigation did not finish. Try again.");
  return {
    report: body.report,
    recommend_review: Boolean(body.recommend_review),
    action: "none",
    evidence: (body.evidence ?? []).map((hit) => ({
      source: String(hit.source ?? ""),
      excerpt: String(hit.excerpt ?? ""),
      score: Number(hit.score ?? 0),
    })),
    steps: (body.steps ?? []).map((step) => ({
      thought: String(step.thought ?? ""),
      action: String(step.action ?? ""),
    })),
  };
}