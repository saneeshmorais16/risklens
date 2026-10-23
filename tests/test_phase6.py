"""Phase 6: PSI per served model, once that model has 30 rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlops.drift import profile_frame


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


def _row(model: str, hour: int, score: float) -> dict:
    return {
        "amount": 40,
        "hour": hour,
        "velocity_24h": 0,
        "distance_km": 4,
        "new_device": 0,
        "merchant_risk": 0.1,
        "score": score,
        "review": score >= 0.5,
        "model": model,
    }


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    profile = profile_frame(_book(14), scores=np.full(80, 0.1))
    profile_path = tmp_path / "reference.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    monkeypatch.setenv("REFERENCE_PROFILE_PATH", str(profile_path))
    return TestClient(app)


def test_a_small_version_stays_unread_while_the_book_is_ready(tmp_path, monkeypatch):
    from mlops.score_log import record_score

    client = _client(tmp_path, monkeypatch)
    for _ in range(20):
        record_score(**_row("baseline_logistic", hour=2, score=0.9))
    for _ in range(10):
        record_score(**_row("baseline_logistic@2", hour=2, score=0.9))
    report = client.get("/v1/monitor").json()
    assert report["ready"] is True
    assert report["n"] == 30
    assert "hour" in report["alerts"]
    assert report["action"] == "none"
    by_model = {item["model"]: item for item in report["by_version"]}
    assert by_model["baseline_logistic"]["ready"] is False
    assert by_model["baseline_logistic@2"]["ready"] is False
    assert by_model["baseline_logistic"]["n"] == 20
    assert by_model["baseline_logistic"]["action"] == "none"
    assert "30" in report["by_version_note"]


def test_each_ready_version_keeps_its_own_psi(tmp_path, monkeypatch):
    from mlops.score_log import record_score

    client = _client(tmp_path, monkeypatch)
    for _ in range(30):
        record_score(**_row("baseline_logistic", hour=2, score=0.9))
    for _ in range(30):
        record_score(**_row("baseline_logistic@2", hour=14, score=0.1))
    report = client.get("/v1/monitor").json()
    assert report["ready"] is True
    assert report["action"] == "none"
    assert "hour" in report["alerts"]
    by_model = {item["model"]: item for item in report["by_version"]}
    shifted = by_model["baseline_logistic"]
    steady = by_model["baseline_logistic@2"]
    assert shifted["ready"] is True
    assert steady["ready"] is True
    assert shifted["n"] == 30
    assert "hour" in shifted["alerts"]
    assert "score" in shifted["alerts"]
    assert "hour" not in steady["alerts"]
    assert "score" not in steady["alerts"]
    assert shifted["action"] == "none"
    assert steady["action"] == "none"
