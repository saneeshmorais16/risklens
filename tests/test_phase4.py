"""Phase 4: local canary, shadow, and A/B split. No cluster."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from mlops.deploy import Route, assignment, load_version, score_route
from mlops.registry import CHAMPION, alias_version, register_baseline, set_alias

ROOT = Path(__file__).resolve().parents[1]
ROW = {
    "amount": 40,
    "hour": 14,
    "velocity_24h": 1,
    "distance_km": 3,
    "new_device": 0,
    "merchant_risk": 0.1,
}


class _Marked:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, rows):
        matrix = np.asarray(rows)
        positive = np.full(len(matrix), self.probability)
        return np.column_stack([1.0 - positive, positive])


def _route_file(path: Path, mode: str, weight: float, primary: str = "1", candidate: str = "2") -> None:
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "primary_version": primary,
                "candidate_version": candidate,
                "candidate_weight": weight,
            }
        ),
        encoding="utf-8",
    )


class _Wired(_Marked):
    def predict_proba(self, rows):
        matrix = np.asarray(rows)
        assert matrix.shape == (1, 6)
        assert float(matrix[0, 0]) == pytest.approx(np.log1p(ROW["amount"]))
        assert float(matrix[0, 1]) == pytest.approx(ROW["hour"])
        return super().predict_proba(rows)


def _load_marked(_uri: str, version: str) -> _Wired:
    return _Wired(0.99 if str(version) == "2" else 0.2)


def test_split_is_stable_and_matches_the_weight():
    keys = [f"C-{index}" for index in range(4000)]
    canary = Route("canary", "1", "2", 0.25)
    experiment = Route("ab", "1", "2", 0.25)
    shadow = Route("shadow", "1", "2", 0.25)
    chosen = 0
    for key in keys:
        left = assignment(canary, key)
        assert assignment(canary, key) == left
        right = assignment(experiment, key)
        mirrored = assignment(shadow, key)
        assert left["served_version"] == right["served_version"]
        assert left["shadow_version"] is None
        assert mirrored["served_version"] == "1"
        assert mirrored["served_slot"] == "primary"
        if left["served_slot"] == "candidate":
            chosen += 1
            assert mirrored["shadow_version"] == "2"
        else:
            assert mirrored["shadow_version"] is None
    assert 900 <= chosen <= 1100
    assert assignment(Route("canary", "1", "2", 0), "C-1")["served_version"] == "1"
    assert assignment(Route("ab", "1", "2", 1), "C-1")["served_version"] == "2"


def test_shadow_scores_the_candidate_and_serves_the_primary():
    route = Route("shadow", "1", "2", 1)
    models = {"1": _Marked(0.2), "2": _Marked(0.99)}
    result = score_route(route, "C-1", models, np.zeros((1, 6), dtype=np.float32))
    assert result["served_version"] == "1"
    assert result["score"] == pytest.approx(0.2)
    assert result["shadow_version"] == "2"
    assert result["shadow_score"] == pytest.approx(0.99)
    assert result["shadow_served"] is False
    assert result["action"] == "none"
    assert result["cluster"] is False
    assert result["deployed"] is False
    assert result["alias_moved"] is False


def test_route_rejects_a_split_that_is_not_a_real_choice():
    with pytest.raises(ValueError, match="canary, shadow, or ab"):
        Route("rollout", "1", "2", 0.1)
    with pytest.raises(ValueError, match="must differ"):
        Route("canary", "1", "1", 0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        Route("ab", "1", "2", 1.2)
    with pytest.raises(ValueError, match="key is required"):
        assignment(Route("canary", "1", "2", 1), "")


def test_missing_or_invalid_route_takes_no_action(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    monkeypatch.delenv("ROUTE_PATH", raising=False)
    client = TestClient(app)
    missing = client.get("/v1/route").json()
    assert missing["ready"] is False
    assert missing["reason"] == "ROUTE_PATH is not set"
    assert missing["action"] == "none"
    assert missing["cluster"] is False
    assert missing["deployed"] is False
    assert missing["alias_moved"] is False
    bad = tmp_path / "route.json"
    bad.write_text('{"mode": "blue-green"}', encoding="utf-8")
    monkeypatch.setenv("ROUTE_PATH", str(bad))
    refused = client.post("/v1/route/score", json={"key": "C-1", **ROW}).json()
    assert refused["ready"] is False
    assert refused["action"] == "none"
    assert refused["cluster"] is False
    valid = tmp_path / "valid.json"
    _route_file(valid, "ab", 0.5)
    monkeypatch.setenv("ROUTE_PATH", str(valid))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    untracked = client.post("/v1/route/score", json={"key": "C-1", **ROW}).json()
    assert untracked["ready"] is False
    assert untracked["reason"] == "MLFLOW_TRACKING_URI is not set"
    assert untracked["action"] == "none"
    assert untracked["cluster"] is False
    assert untracked["deployed"] is False


def test_canary_endpoint_serves_the_candidate_without_a_cluster(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    route_path = tmp_path / "route.json"
    _route_file(route_path, "canary", 1)
    monkeypatch.setenv("ROUTE_PATH", str(route_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    monkeypatch.setattr("mlops.deploy.load_version", _load_marked)
    listed = TestClient(app).get("/v1/route").json()
    assert listed["ready"] is True
    assert listed["mode"] == "canary"
    assert listed["cluster"] is False
    assert listed["deployed"] is False
    assert listed["action"] == "none"
    body = TestClient(app).post("/v1/route/score", json={"key": "C-9", **ROW}).json()
    assert body["ready"] is True
    assert body["served_version"] == "2"
    assert body["served_slot"] == "candidate"
    assert body["score"] == pytest.approx(0.99)
    assert body["shadow_score"] is None
    assert body["review"] is (body["score"] >= body["threshold"])
    assert body["action"] == "none"
    assert body["cluster"] is False
    assert body["deployed"] is False
    assert body["alias_moved"] is False
    assert "No cluster was created" in body["note"]


def test_shadow_endpoint_logs_the_served_score_only(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app
    from mlops.score_log import recent

    route_path = tmp_path / "route.json"
    _route_file(route_path, "shadow", 1)
    monkeypatch.setenv("ROUTE_PATH", str(route_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    monkeypatch.setattr("mlops.deploy.load_version", _load_marked)
    body = TestClient(app).post("/v1/route/score", json={"key": "C-9", **ROW}).json()
    assert body["served_version"] == "1"
    assert body["score"] == pytest.approx(0.2)
    assert body["shadow_score"] == pytest.approx(0.99)
    assert body["shadow_served"] is False
    assert body["action"] == "none"
    frame = recent(limit=10)
    assert len(frame) == 1
    assert frame.loc[0, "score"] == pytest.approx(0.2)
    assert frame.loc[0, "model"] == "baseline_logistic@1"
    assert frame.loc[0, "action"] == "none"


def test_route_pins_versions_and_does_not_move_the_champion_alias(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    model = joblib.load(ROOT / "models" / "baseline.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())["baseline_logistic"]
    tracking = tmp_path.as_uri()
    first = register_baseline(tracking, model, metrics)
    second = register_baseline(tracking, model, metrics)
    set_alias(tracking, first["version"], CHAMPION)
    route_path = tmp_path / "route.json"
    _route_file(route_path, "ab", 0, primary=second["version"], candidate=first["version"])
    monkeypatch.setenv("ROUTE_PATH", str(route_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCORE_LOG_PATH", str(tmp_path / "scores.db"))
    client = TestClient(app)
    body = client.post("/v1/route/score", json={"key": "C-3", **ROW}).json()
    assert body["ready"] is True
    assert body["mode"] == "ab"
    assert body["served_version"] == second["version"]
    assert body["action"] == "none"
    assert body["cluster"] is False
    assert body["alias_moved"] is False
    assert alias_version(tracking, CHAMPION) == first["version"]
    set_alias(tracking, second["version"], CHAMPION)
    again = client.post("/v1/route/score", json={"key": "C-3", **ROW}).json()
    assert again["served_version"] == second["version"]
    assert alias_version(tracking, CHAMPION) == second["version"]


def test_load_version_reads_the_registered_baseline(tmp_path):
    model = joblib.load(ROOT / "models" / "baseline.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())["baseline_logistic"]
    tracking = tmp_path.as_uri()
    registered = register_baseline(tracking, model, metrics)
    loaded = load_version(tracking, registered["version"])
    probability = float(loaded.predict_proba(np.zeros((1, 6), dtype=np.float32))[0, 1])
    assert 0.0 <= probability <= 1.0
    assert np.isfinite(probability)
