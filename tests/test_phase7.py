"""Phase 7: name the model to look at. Leave the route and the alias alone."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlops.drift import look_at, profile_frame


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


def test_a_short_model_is_not_the_quiet_comparison():
    report = look_at(
        [
            {"model": "baseline_logistic", "ready": True, "alerts": ["hour"], "action": "none"},
            {"model": "baseline_logistic@2", "ready": False, "n": 10, "action": "none"},
        ]
    )
    assert report["ready"] is False
    assert report["models"] == []
    assert report["reason"] == "fewer than two models have a PSI"
    assert report["alias_moved"] is False
    assert report["route_changed"] is False
    assert report["action"] == "none"


def test_disagreement_names_the_alerting_model_and_leaves_the_route(tmp_path, monkeypatch):
    from mlops.score_log import record_score

    route_path = tmp_path / "route.json"
    route_path.write_text('{"mode": "canary", "primary_version": "1", "candidate_version": "2", "candidate_weight": 0.1}', encoding="utf-8")
    before = route_path.read_text(encoding="utf-8")
    monkeypatch.setenv("ROUTE_PATH", str(route_path))
    client = _client(tmp_path, monkeypatch)
    for _ in range(30):
        record_score(**_row("baseline_logistic", hour=2, score=0.9))
    for _ in range(30):
        record_score(**_row("baseline_logistic@2", hour=14, score=0.1))
    report = client.get("/v1/monitor").json()
    look = report["look"]
    assert report["action"] == "none"
    assert look["ready"] is True
    assert look["models"] == ["baseline_logistic"]
    assert look["quiet"] == ["baseline_logistic@2"]
    assert look["alias_moved"] is False
    assert look["route_changed"] is False
    assert look["action"] == "none"
    assert "champion alias" in look["note"]
    assert route_path.read_text(encoding="utf-8") == before


def test_two_alerting_models_are_not_a_disagreement(tmp_path, monkeypatch):
    from mlops.score_log import record_score

    client = _client(tmp_path, monkeypatch)
    for _ in range(30):
        record_score(**_row("baseline_logistic", hour=2, score=0.9))
    for _ in range(30):
        record_score(**_row("baseline_logistic@2", hour=2, score=0.9))
    look = client.get("/v1/monitor").json()["look"]
    assert look["ready"] is False
    assert look["models"] == []
    assert look["reason"] == "every ready model is alerting"
    assert look["alias_moved"] is False
    assert look["route_changed"] is False
    assert look["action"] == "none"
