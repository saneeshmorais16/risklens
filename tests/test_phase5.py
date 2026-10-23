"""Phase 5: served-row counts by model. One PSI for the whole book."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlops.drift import profile_frame
from mlops.score_log import record_score, version_mix


def _row(model: str, hour: int = 14) -> dict:
    return {
        "amount": 40,
        "hour": hour,
        "velocity_24h": 0,
        "distance_km": 4,
        "new_device": 0,
        "merchant_risk": 0.1,
        "score": 0.2,
        "review": False,
        "model": model,
    }


def test_version_mix_orders_by_count_then_name():
    frame = pd.DataFrame(
        [
            {"model": "baseline_logistic@2"},
            {"model": "baseline_logistic@1"},
            {"model": "baseline_logistic@2"},
            {"model": "  "},
        ]
    )
    mix = version_mix(frame)
    assert [item["model"] for item in mix] == ["baseline_logistic@2", "baseline_logistic", "baseline_logistic@1"]
    assert mix[0]["n"] == 2
    assert mix[0]["share"] == 0.5
    assert version_mix(pd.DataFrame()) == []


def test_monitor_counts_versions_and_keeps_one_psi(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    profile = profile_frame(
        pd.DataFrame(
            {
                "amount": np.full(80, 40.0),
                "hour": np.full(80, 14),
                "velocity_24h": np.zeros(80, dtype=int),
                "distance_km": np.full(80, 4.0),
                "new_device": np.zeros(80, dtype=int),
                "merchant_risk": np.full(80, 0.1),
            }
        ),
        scores=np.full(80, 0.1),
    )
    profile_path = tmp_path / "reference.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    monkeypatch.setenv("REFERENCE_PROFILE_PATH", str(profile_path))
    client = TestClient(app)
    record_score(**_row("baseline_logistic@2", hour=2))
    early = client.get("/v1/monitor").json()
    assert early["ready"] is False
    assert early["action"] == "none"
    assert early["versions"] == [{"model": "baseline_logistic@2", "n": 1, "share": 1.0}]
    for _ in range(20):
        record_score(**_row("baseline_logistic", hour=2))
    for _ in range(9):
        record_score(**_row("baseline_logistic@2", hour=2))
    report = client.get("/v1/monitor").json()
    assert report["ready"] is True
    assert report["n"] == 30
    assert report["action"] == "none"
    assert "hour" in report["alerts"]
    assert set(report["features"]) == {
        "log_amount",
        "hour",
        "velocity_24h",
        "distance_km",
        "new_device",
        "merchant_risk",
    }
    assert "psi" in report["score"]
    assert [item["model"] for item in report["versions"]] == ["baseline_logistic", "baseline_logistic@2"]
    assert report["versions"][0]["n"] == 20
    assert report["versions"][0]["share"] == 20 / 30
    assert report["versions"][1]["n"] == 10
    assert "whole book" in report["versions_note"]
