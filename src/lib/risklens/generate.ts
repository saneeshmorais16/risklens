import { DAY_COUNT, TRAIN_LAST_DAY, type FeatureRow } from "./contract";
import { makeRng } from "./rng";

export type SyntheticTx = FeatureRow & {
  transaction_id: string;
  customer_id: string;
  event_day: number;
  injected_scenario: boolean;
  split: "train" | "test";
};

export function generateTransactions(n: number, seed: number): SyntheticTx[] {
  const rng = makeRng(seed);
  const injectedCount = Math.max(1, Math.floor(n / 25));
  const injected = new Set<number>();
  while (injected.size < injectedCount) {
    injected.add(Math.floor(rng.next() * n));
  }

  const rows: SyntheticTx[] = [];
  for (let i = 0; i < n; i++) {
    const customer = 1 + Math.floor(rng.next() * 150);
    let amount = Math.exp(3.6 + 0.9 * rng.normal());
    let velocity = rng.poisson(2) + 1;
    let distance = -12 * Math.log(Math.max(rng.next(), 1e-12));
    let device: 0 | 1 = rng.next() < 0.08 ? 1 : 0;
    let hour = Math.floor(rng.next() * 24);
    const eventDay = Math.floor(rng.next() * DAY_COUNT);
    const scenario = injected.has(i);
    if (scenario) {
      amount *= 1.7 + rng.next() * 2.6;
      if (rng.next() < 0.75) velocity += 3 + Math.floor(rng.next() * 7);
      if (rng.next() < 0.75) distance += 50 + rng.next() * 200;
      if (rng.next() < 0.55) device = 1;
      if (rng.next() < 0.3) hour = Math.floor(rng.next() * 5);
    }
    rows.push({
      transaction_id: `TX-${String(i).padStart(6, "0")}`,
      customer_id: `C-${String(customer).padStart(4, "0")}`,
      amount: Math.round(amount * 100) / 100,
      hour,
      velocity_24h: velocity,
      distance_km: Math.round(distance * 10) / 10,
      new_device: device,
      event_day: eventDay,
      injected_scenario: scenario,
      split: eventDay <= TRAIN_LAST_DAY ? "train" : "test",
    });
  }
  return rows;
}
