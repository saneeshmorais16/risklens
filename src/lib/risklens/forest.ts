import { makeRng } from "./rng";

export type Leaf = { k: "l"; n: number };
export type Split = { k: "s"; f: number; v: number; L: Node; R: Node };
export type Node = Leaf | Split;

export type ForestModel = {
  trees: Node[];
  sampleSize: number;
  threshold: number;
  c: number;
};

function cFactor(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649015329) - (2 * (n - 1)) / n;
}

function buildTree(data: number[][], rng: () => number, maxDepth: number): Node {
  const dims = data[0]?.length ?? 0;
  const grow = (idx: number[], depth: number): Node => {
    if (idx.length <= 1 || depth >= maxDepth || dims === 0) {
      return { k: "l", n: idx.length };
    }
    let feature = -1;
    let lo = 0;
    let hi = 0;
    for (let attempt = 0; attempt < dims * 2; attempt += 1) {
      const f = Math.floor(rng() * dims);
      let min = Infinity;
      let max = -Infinity;
      for (const i of idx) {
        const value = data[i][f];
        if (value < min) min = value;
        if (value > max) max = value;
      }
      if (max > min) {
        feature = f;
        lo = min;
        hi = max;
        break;
      }
    }
    if (feature < 0) return { k: "l", n: idx.length };
    const split = lo + rng() * (hi - lo);
    const left: number[] = [];
    const right: number[] = [];
    for (const i of idx) {
      if (data[i][feature] < split) left.push(i);
      else right.push(i);
    }
    if (left.length === 0 || right.length === 0) return { k: "l", n: idx.length };
    return {
      k: "s",
      f: feature,
      v: split,
      L: grow(left, depth + 1),
      R: grow(right, depth + 1),
    };
  };
  return grow(
    data.map((_, index) => index),
    0,
  );
}

function pathLength(node: Node, point: number[], depth: number): number {
  if (node.k === "l") return depth + cFactor(node.n);
  return point[node.f] < node.v
    ? pathLength(node.L, point, depth + 1)
    : pathLength(node.R, point, depth + 1);
}

export function scorePoint(model: ForestModel, point: number[]): number {
  if (model.trees.length === 0 || model.c === 0) return 0;
  let sum = 0;
  for (const tree of model.trees) sum += pathLength(tree, point, 0);
  return 2 ** (-sum / model.trees.length / model.c);
}

export function fitForest(
  points: number[][],
  trees: number,
  seed: number,
  contamination: number,
): ForestModel {
  if (points.length < 8) throw new Error("Need at least 8 training rows to fit a model.");
  const { next } = makeRng(seed ^ 0x9e3779b9);
  const sampleSize = Math.min(256, points.length);
  const maxDepth = Math.ceil(Math.log2(sampleSize));
  const built: Node[] = [];
  for (let t = 0; t < trees; t += 1) {
    const bag = points.map((_, index) => index);
    const take = Math.min(sampleSize, bag.length);
    const subset: number[][] = [];
    for (let i = 0; i < take; i += 1) {
      const j = i + Math.floor(next() * (bag.length - i));
      const swap = bag[i];
      bag[i] = bag[j];
      bag[j] = swap;
      subset.push(points[bag[i]]);
    }
    built.push(buildTree(subset, next, maxDepth));
  }
  const model: ForestModel = {
    trees: built,
    sampleSize,
    threshold: 1,
    c: cFactor(sampleSize),
  };
  const scores = points.map((point) => scorePoint(model, point)).sort((a, b) => a - b);
  const cut = Math.min(
    scores.length - 1,
    Math.max(0, Math.floor((1 - contamination) * scores.length)),
  );
  model.threshold = scores[cut] ?? 1;
  return model;
}

export function toVector(row: {
  amount: number;
  hour: number;
  velocity_24h: number;
  distance_km: number;
  new_device: number;
}): number[] {
  return [
    Math.log1p(Math.max(0, row.amount)),
    row.hour,
    row.velocity_24h,
    Math.log1p(Math.max(0, row.distance_km)),
    row.new_device,
  ];
}
