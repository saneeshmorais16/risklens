import { quantile } from "./metrics";

export type MonitorRow = {
  split: "train" | "test";
  amount: number;
  hour: number;
  velocity24h: number;
  distanceKm: number;
  newDevice: number;
  flagged: boolean;
};

export type DriftRow = {
  feature: string;
  trainMean: number;
  testMean: number;
  smd: number;
  status: "stable" | "watch" | "shift";
};

export type SliceRow = {
  name: string;
  n: number;
  flagRate: number;
  complementRate: number;
  gap: number;
};

function mean(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sd(values: number[]) {
  if (values.length < 2) return 0;
  const center = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function statusOf(smd: number): DriftRow["status"] {
  const abs = Math.abs(smd);
  if (abs < 0.2) return "stable";
  if (abs < 0.5) return "watch";
  return "shift";
}

export function featureDrift(rows: MonitorRow[]): DriftRow[] {
  const train = rows.filter((row) => row.split === "train");
  const test = rows.filter((row) => row.split === "test");
  const specs = [
    ["Amount", (row: MonitorRow) => row.amount],
    ["Hour", (row: MonitorRow) => row.hour],
    ["Velocity", (row: MonitorRow) => row.velocity24h],
    ["Distance", (row: MonitorRow) => row.distanceKm],
    ["New device", (row: MonitorRow) => row.newDevice],
  ] as const;
  return specs.map(([feature, pick]) => {
    const trainValues = train.map(pick);
    const testValues = test.map(pick);
    const trainMean = mean(trainValues);
    const testMean = mean(testValues);
    const spread = sd(trainValues);
    const smd = spread < 1e-9 ? 0 : (testMean - trainMean) / spread;
    return { feature, trainMean, testMean, smd, status: statusOf(smd) };
  });
}

function flagRate(rows: MonitorRow[]) {
  if (rows.length === 0) return 0;
  return rows.filter((row) => row.flagged).length / rows.length;
}

export function sliceGaps(rows: MonitorRow[]): SliceRow[] {
  const train = rows.filter((row) => row.split === "train");
  const test = rows.filter((row) => row.split === "test");
  const amountP95 = quantile(train.map((row) => row.amount), 0.95);
  const cuts = [
    ["Overnight", (row: MonitorRow) => row.hour < 5],
    ["New device", (row: MonitorRow) => row.newDevice === 1],
    ["Large amount", (row: MonitorRow) => row.amount > amountP95],
  ] as const;
  return cuts.map(([name, pred]) => {
    const inside = test.filter(pred);
    const outside = test.filter((row) => !pred(row));
    const insideRate = flagRate(inside);
    const outsideRate = flagRate(outside);
    return {
      name,
      n: inside.length,
      flagRate: insideRate,
      complementRate: outsideRate,
      gap: insideRate - outsideRate,
    };
  });
}
