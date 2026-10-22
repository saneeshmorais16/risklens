export type HoldoutMetrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  rocAuc: number | null;
  flagRate: number;
  positives: number;
  flagged: number;
};

export function holdoutMetrics(
  scores: number[],
  labels: boolean[],
  flagged: boolean[],
): HoldoutMetrics {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let flaggedCount = 0;
  for (let i = 0; i < labels.length; i += 1) {
    if (flagged[i]) flaggedCount += 1;
    if (flagged[i] && labels[i]) tp += 1;
    else if (flagged[i] && !labels[i]) fp += 1;
    else if (!flagged[i] && labels[i]) fn += 1;
  }
  const precision = tp + fp === 0 ? null : tp / (tp + fp);
  const recall = tp + fn === 0 ? null : tp / (tp + fn);
  const f1 =
    precision === null || recall === null || precision + recall === 0
      ? null
      : (2 * precision * recall) / (precision + recall);
  return {
    precision,
    recall,
    f1,
    rocAuc: rocAuc(scores, labels),
    flagRate: labels.length === 0 ? 0 : flaggedCount / labels.length,
    positives: labels.filter(Boolean).length,
    flagged: flaggedCount,
  };
}

function rocAuc(scores: number[], labels: boolean[]): number | null {
  const positives = labels.filter(Boolean).length;
  const negatives = labels.length - positives;
  if (positives === 0 || negatives === 0) return null;
  const order = scores.map((score, index) => ({ score, index }));
  order.sort((a, b) => a.score - b.score);
  const ranks = new Array<number>(scores.length);
  let cursor = 0;
  while (cursor < order.length) {
    let end = cursor;
    while (end + 1 < order.length && order[end + 1].score === order[cursor].score) end += 1;
    const average = (cursor + 1 + (end + 1)) / 2;
    for (let k = cursor; k <= end; k += 1) ranks[order[k].index] = average;
    cursor = end + 1;
  }
  let rankSum = 0;
  for (let i = 0; i < labels.length; i += 1) if (labels[i]) rankSum += ranks[i];
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

export function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[index] ?? 0;
}
