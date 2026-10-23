"""Reproducible synthetic book. Customers and labels are fictional."""

from __future__ import annotations

import numpy as np
import pandas as pd

MERCHANT_RISK = {
    "grocery": 0.1,
    "fuel": 0.2,
    "travel": 0.7,
    "electronics": 0.55,
    "wire": 0.85,
}


# A fictional book. The label is a noisy injection. It is not a finding about a real customer.
def generate(n_customers: int = 360, days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    txn_id = 1
    for customer in range(n_customers):
        home = float(rng.uniform(0, 80))
        typical_amount = float(rng.uniform(12, 80))
        typical_hour = int(rng.integers(8, 20))
        device = 0
        for day in range(1, days + 1):
            n_tx = int(rng.integers(0, 3))
            for _ in range(n_tx):
                scenario = "ordinary"
                amount = float(np.clip(rng.normal(typical_amount, typical_amount * 0.25), 1, 400))
                hour = int(np.clip(rng.normal(typical_hour, 2), 0, 23))
                distance = float(np.clip(abs(rng.normal(0, 8)), 0, 40))
                new_device = 0
                merchant = str(rng.choice(["grocery", "fuel", "electronics"]))
                if rng.random() < 0.015 and day >= 10:
                    scenario = "far_overnight"
                    amount = float(rng.uniform(180, 900))
                    hour = int(rng.choice([0, 1, 2, 3, 23]))
                    distance = float(rng.uniform(120, 480))
                    new_device = 1
                    merchant = str(rng.choice(["travel", "wire", "electronics"]))
                elif rng.random() < 0.012 and day >= 10:
                    scenario = "burst"
                    amount = float(rng.uniform(40, 220))
                    hour = typical_hour
                    distance = float(rng.uniform(0, 15))
                    merchant = "wire"
                if rng.random() < 0.02:
                    device = int(rng.integers(0, 5))
                    new_device = 1
                if rng.random() < 0.04:
                    distance = float(rng.uniform(50, 180))
                    hour = int(rng.choice([0, 1, 22, 23, 6]))
                rows.append(
                    {
                        "transaction_id": f"TX-{txn_id:06d}",
                        "customer_id": f"C-{customer:04d}",
                        "day": day,
                        "hour": hour,
                        "amount": round(amount, 2),
                        "distance_km": round(home * 0 + distance, 2),
                        "velocity_24h": 0,
                        "new_device": new_device,
                        "merchant": merchant,
                        "merchant_risk": MERCHANT_RISK[merchant],
                        "scenario": scenario,
                        "label": int(scenario != "ordinary"),
                    }
                )
                txn_id += 1
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["customer_id", "day", "hour", "transaction_id"]).reset_index(drop=True)
    # Velocity counts earlier rows only. The row being written is not included yet.
    counts: dict[tuple[str, int], int] = {}
    velocity: list[int] = []
    for row in frame.itertuples(index=False):
        key = (row.customer_id, int(row.day))
        prior = counts.get(key, 0) + counts.get((row.customer_id, int(row.day) - 1), 0)
        velocity.append(prior)
        counts[key] = counts.get(key, 0) + 1
    frame["velocity_24h"] = velocity
    is_burst = frame["scenario"] == "burst"
    frame.loc[is_burst, "label"] = (frame.loc[is_burst, "velocity_24h"] >= 2).astype(int)
    frame.loc[is_burst & (frame["label"] == 0), "scenario"] = "ordinary"
    # Drop some injected labels on purpose. A perfect score would mean the model memorised the generator.
    positives = frame.index[frame["label"] == 1]
    drop = rng.choice(positives.to_numpy(), size=max(1, int(0.2 * len(positives))), replace=False)
    frame.loc[drop, "label"] = 0
    return frame


# A sentence built from the numbers, for the narrative encoder. It is not a customer note.
def narrative(row: pd.Series) -> str:
    device = "new device" if int(row["new_device"]) == 1 else "known device"
    return (
        f"Amount {float(row['amount']):.0f} at hour {int(row['hour'])}. "
        f"Velocity {int(row['velocity_24h'])}. Distance {float(row['distance_km']):.0f} km. "
        f"{device}. Merchant risk {float(row['merchant_risk']):.2f}."
    )
