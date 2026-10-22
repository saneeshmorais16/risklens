"""Phase 3: local registry, feature store, drift, and the score log."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from features.build import FEATURE_COLUMNS
from mlops.drift import column_psi, drift_report, load_reference, profile_column, profile_frame
from mlops.feature_registry import apply_store, feature_view, online_features, registered_feature_names
from mlops.registry import CHAMPION, MODEL_NAME, alias_version, register_baseline, set_alias
from mlops.score_log import recent, record_score, uses_postgres

ROOT = Path(__file__).resolve().parents[1]


def _book(hour: int, n: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": np.full(n, 40.0),
            "hour": np.full(n, hour),
            "velocity_24h": np.zeros(n, dtype=int),
            "distance_km": np.full(n, 4.0),
            "new_device": np.zeros(n, dtype=int),
            "merchant_risk": np.full(n, 0.1),
        }
    )


def test_identical_sample_has_no_drift():
    values = np.random.default_rng(0).normal(40, 8, 400)
    assert column_psi(profile_column(values), values) < 0.01


def test_shifted_hour_asks_for_a_look_and_takes_no_action():
    reference = profile_frame(_book(14))
    report = drift_report(reference, _book(2))
    assert "hour" in report["alerts"]
    assert "amount" not in report["alerts"]
    assert report["action"] == "none"


def test_checked_in_reference_is_the_training_window():
    profile = load_reference(ROOT / "models" / "reference_profile.json")
    assert set(profile["features"]) == set(FEATURE_COLUMNS)
    assert "score" in profile
    assert profile["psi_alert"] == 0.25
    assert "days 1-21" in profile["source"]
    assert profile["model"] == "baseline_logistic"
    for column in profile["features"].values():
        assert abs(sum(column["proportions"]) - 1) < 1e-6
        assert column["n"] > 100


def test_sqlite_score_log_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    assert uses_postgres() is False
    record_score(
        amount=25,
        hour=11,
        velocity_24h=1,
        distance_km=3,
        new_device=0,
        merchant_risk=0.2,
        score=0.2,
        review=False,
    )
    frame = recent(limit=10)
    assert len(frame) == 1
    assert frame.loc[0, "action"] == "none"
    assert frame.loc[0, "hour"] == 11
    assert frame.loc[0, "log_amount"] == pytest.approx(np.log1p(25))


def test_registry_alias_moves_back_to_the_older_version(tmp_path):
    model = joblib.load(ROOT / "models" / "baseline.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())["baseline_logistic"]
    tracking = tmp_path.as_uri()
    first = register_baseline(tracking, model, metrics)
    second = register_baseline(tracking, model, metrics)
    assert first["name"] == MODEL_NAME
    assert first["deployed"] is False
    assert int(second["version"]) > int(first["version"])
    assert alias_version(tracking, CHAMPION) == second["version"]
    set_alias(tracking, first["version"], CHAMPION)
    assert alias_version(tracking, CHAMPION) == first["version"]


def test_feast_online_store_returns_latest_features_without_the_label(tmp_path):
    frame = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-1"],
            "day": [1, 2],
            "amount": [10.0, 80.0],
            "hour": [10, 11],
            "velocity_24h": [0, 1],
            "distance_km": [2.0, 6.0],
            "new_device": [0, 1],
            "merchant_risk": [0.1, 0.2],
            "label": [0, 1],
            "scenario": ["ordinary", "burst"],
        }
    )
    _, view = feature_view(tmp_path / "unused.parquet")
    names = registered_feature_names(view)
    assert set(names) == set(FEATURE_COLUMNS)
    assert "label" not in names
    assert "scenario" not in names
    store = apply_store(tmp_path / "repo", frame)
    found = online_features(store, "C-1")
    assert found["hour"] == 11
    assert found["new_device"] == 1
    assert found["log_amount"] == pytest.approx(np.log1p(80))


def test_monitor_endpoint_reports_drift_without_an_account_action(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    profile = profile_frame(_book(14), scores=np.full(80, 0.1))
    profile_path = tmp_path / "reference.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    monkeypatch.setenv("REFERENCE_PROFILE_PATH", str(profile_path))
    client = TestClient(app)
    early = client.get("/v1/monitor")
    assert early.status_code == 200
    assert early.json()["ready"] is False
    assert early.json()["action"] == "none"
    for _ in range(30):
        record_score(
            amount=40,
            hour=2,
            velocity_24h=0,
            distance_km=4,
            new_device=0,
            merchant_risk=0.1,
            score=0.9,
            review=True,
        )
    report = client.get("/v1/monitor").json()
    assert report["ready"] is True
    assert "hour" in report["alerts"]
    assert "score" in report["alerts"]
    assert report["action"] == "none"


def test_registry_endpoint_does_not_claim_a_deployment(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    model = joblib.load(ROOT / "models" / "baseline.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())["baseline_logistic"]
    tracking = tmp_path.as_uri()
    registered = register_baseline(tracking, model, metrics)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    body = TestClient(app).get("/v1/registry").json()
    assert body["ready"] is True
    assert body["version"] == registered["version"]
    assert body["deployed"] is False
