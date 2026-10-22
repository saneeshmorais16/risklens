import { CONTRACT_VERSION, DAY_COUNT, TRAIN_LAST_DAY, type FeatureRow, type TrainInput } from "./contract";
import { fitForest, scorePoint, toVector, type ForestModel } from "./forest";
import { generateTransactions, type SyntheticTx } from "./generate";
import { holdoutMetrics, quantile, type HoldoutMetrics } from "./metrics";

export type ScoredSynthetic = SyntheticTx & {
  anomaly_score: number;
  flagged: boolean;
  review_reasons: string;
};

export type TrainResult = {
  label: string;
  contractVersion: string;
  trainDays: string;
  testDays: string;
  model: ForestModel;
  rows: ScoredSynthetic[];
  metrics: HoldoutMetrics;
  amountP95: number;
  nTrain: number;
  nTest: number;
};

export function explainRow(row: FeatureRow, amountP95: number): string {
  const reasons: string[] = [];
  if (row.amount > amountP95) reasons.push("Unusually large amount");
  if (row.velocity_24h >= 8) reasons.push("High 24-hour transaction velocity");
  if (row.distance_km >= 100) reasons.push("Large distance from usual location");
  if (row.new_device === 1) reasons.push("New device");
  if (row.hour < 5) reasons.push("Overnight activity");
  return reasons.join("; ") || "Multivariate pattern outside typical activity";
}

export function trainSynthetic(input: TrainInput): TrainResult {
  const rows = generateTransactions(input.n, input.seed);
  const train = rows.filter((row) => row.split === "train");
  const test = rows.filter((row) => row.split === "test");
  if (train.length < 8 || test.length < 8) {
    throw new Error("This seed did not fill both the training window and the holdout. Try another seed.");
  }
  const amountP95 = quantile(
    train.map((row) => row.amount),
    0.95,
  );
  const model = fitForest(train.map(toVector), input.trees, input.seed, input.contamination);
  const scored = rows.map((row) => {
    const raw = scorePoint(model, toVector(row));
    const anomaly_score = Math.round(raw * 10000) / 10000;
    return {
      ...row,
      anomaly_score,
      flagged: raw >= model.threshold,
      review_reasons: explainRow(row, amountP95),
    };
  });
  const holdout = scored.filter((row) => row.split === "test");
  const metrics = holdoutMetrics(
    holdout.map((row) => row.anomaly_score),
    holdout.map((row) => row.injected_scenario),
    holdout.map((row) => row.flagged),
  );
  return {
    label: `IF-${input.seed}`,
    contractVersion: CONTRACT_VERSION,
    trainDays: `1–${TRAIN_LAST_DAY + 1}`,
    testDays: `${TRAIN_LAST_DAY + 2}–${DAY_COUNT}`,
    model,
    rows: scored,
    metrics,
    amountP95,
    nTrain: train.length,
    nTest: test.length,
  };
}

export function scoreFeatures(
  model: ForestModel,
  rows: FeatureRow[],
  amountP95: number,
): Array<FeatureRow & { anomaly_score: number; flagged: boolean; review_reasons: string }> {
  return rows.map((row) => {
    const raw = scorePoint(model, toVector(row));
    const anomaly_score = Math.round(raw * 10000) / 10000;
    return {
      ...row,
      anomaly_score,
      flagged: raw >= model.threshold,
      review_reasons: explainRow(row, amountP95),
    };
  });
}
