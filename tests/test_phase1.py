import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from training.generate import generate
from training.train import split

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_split_does_not_leak():
    frame = generate(n_customers=40, seed=7)
    train, valid, test = split(frame)
    assert train["day"].max() <= 21
    assert valid["day"].min() >= 22 and valid["day"].max() <= 24
    assert test["day"].min() >= 25
    assert train["day"].max() < test["day"].min()
    assert set(frame["label"].unique()) <= {0, 1}


def test_velocity_uses_only_earlier_rows():
    frame = generate(n_customers=30, seed=3)
    first = frame.groupby("customer_id", sort=False).head(1)
    assert (first["velocity_24h"] == 0).all()


def test_metrics_come_from_training_artifact():
    report = json.loads((ROOT / "models" / "metrics.json").read_text())
    baseline = report["baseline_logistic"]
    for key in ("precision", "recall", "f1", "pr_auc"):
        assert 0.0 <= baseline[key] <= 1.0
    confusion = baseline["confusion"]
    assert sum(confusion.values()) == baseline["n"]
    assert "synthetic" in report["limitation"].lower()


def test_score_does_not_take_an_account_action():
    from backend.risklens_api.main import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    scored = client.post(
        "/v1/score",
        json={
            "amount": 40,
            "hour": 14,
            "velocity_24h": 1,
            "distance_km": 3,
            "new_device": 0,
            "merchant_risk": 0.1,
        },
    )
    assert scored.status_code == 200
    body = scored.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["action"] == "none"
    assert isinstance(body["review"], bool)
    assert np.isfinite(body["score"])
