"""Live PostgreSQL check.

Skipped unless `docker info` succeeds. The test starts postgres:16-alpine,
writes through the score log and the scoring API, and removes the container.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import uuid

import pytest

ROW = {
    "amount": 40,
    "hour": 14,
    "velocity_24h": 1,
    "distance_km": 3,
    "new_device": 0,
    "merchant_risk": 0.1,
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker() -> str:
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("Docker is not installed, so live PostgreSQL was not started")
    info = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=30)
    if info.returncode != 0:
        pytest.skip("Docker is installed but the daemon is not running, so live PostgreSQL was not started")
    return docker


@pytest.fixture(scope="module")
def postgres_url():
    docker = _docker()
    name = f"risklens-live-{uuid.uuid4().hex[:8]}"
    port = _free_port()
    try:
        started = subprocess.run(
            [
                docker,
                "run",
                "-d",
                "--name",
                name,
                "-e",
                "POSTGRES_USER=risklens",
                "-e",
                "POSTGRES_PASSWORD=risklens",
                "-e",
                "POSTGRES_DB=risklens",
                "-p",
                f"127.0.0.1:{port}:5432",
                "postgres:16-alpine",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if started.returncode != 0:
            raise RuntimeError(started.stderr.strip() or "docker run failed")
        deadline = time.time() + 40
        ready = False
        while time.time() < deadline:
            probe = subprocess.run(
                [docker, "exec", name, "pg_isready", "-U", "risklens", "-d", "risklens"],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                ready = True
                break
            time.sleep(1)
        if not ready:
            logs = subprocess.run([docker, "logs", name], capture_output=True, text=True)
            detail = (logs.stderr or logs.stdout or "postgres did not become ready")[-500:]
            raise RuntimeError(detail)
        url = f"postgresql://risklens:risklens@127.0.0.1:{port}/risklens"
        connected = False
        connect_deadline = time.time() + 20
        last_error = "connection was not attempted"
        while time.time() < connect_deadline:
            try:
                import psycopg

                with psycopg.connect(url) as conn:
                    conn.execute("SELECT 1")
                connected = True
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1)
        if not connected:
            raise RuntimeError(last_error)
        yield url
    finally:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, text=True, timeout=30)


def test_postgres_score_log_roundtrip(postgres_url, monkeypatch):
    from mlops.score_log import recent, record_score, uses_postgres

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    assert uses_postgres() is True
    record_score(
        amount=18,
        hour=9,
        velocity_24h=1,
        distance_km=2,
        new_device=0,
        merchant_risk=0.1,
        score=0.3,
        review=False,
    )
    frame = recent(limit=20)
    matched = frame[frame["amount"] == 18]
    assert len(matched) == 1
    assert matched.iloc[0]["action"] == "none"
    assert bool(matched.iloc[0]["review"]) is False


def test_score_endpoint_persists_action_none(postgres_url, monkeypatch):
    import psycopg
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    body = TestClient(app).post("/v1/score", json=dict(ROW, amount=19)).json()
    assert body["action"] == "none"
    with psycopg.connect(postgres_url) as conn:
        row = conn.execute(
            "SELECT action, model FROM score_events WHERE amount = %s ORDER BY id DESC LIMIT 1",
            (19,),
        ).fetchone()
    assert row is not None
    assert row[0] == "none"
    assert row[1] == "baseline_logistic"


def test_shadow_route_persists_the_served_score(postgres_url, monkeypatch, tmp_path):
    import json

    import psycopg
    from fastapi.testclient import TestClient

    from backend.risklens_api.main import app

    class _Fixed:
        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict_proba(self, rows):
            import numpy as np

            count = len(rows)
            positive = np.full(count, self.probability)
            return np.column_stack([1.0 - positive, positive])

    route_path = tmp_path / "route.json"
    route_path.write_text(
        json.dumps(
            {
                "mode": "shadow",
                "primary_version": "1",
                "candidate_version": "2",
                "candidate_weight": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("ROUTE_PATH", str(route_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())

    def _load(_uri: str, version: str) -> _Fixed:
        return _Fixed(0.99 if str(version) == "2" else 0.2)

    monkeypatch.setattr("mlops.deploy.load_version", _load)
    body = TestClient(app).post("/v1/route/score", json={"key": "C-live", **ROW, "amount": 21}).json()
    assert body["ready"] is True
    assert body["score"] == pytest.approx(0.2)
    assert body["shadow_score"] == pytest.approx(0.99)
    assert body["shadow_served"] is False
    assert body["action"] == "none"
    assert body["cluster"] is False
    with psycopg.connect(postgres_url) as conn:
        rows = conn.execute(
            "SELECT score, model, action FROM score_events WHERE amount = %s",
            (21,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.2)
    assert rows[0][1] == "baseline_logistic@1"
    assert rows[0][2] == "none"
