"""Risk scoring API. A high score asks for a review. It does not freeze an account."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops.drift import drift_report, load_reference, reference_path
from mlops.score_log import recent, record_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "baseline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DB_PATH = ROOT / "models" / "scores.db"

app = FastAPI(title="RiskLens AI", version="0.1.0")


class ScoreIn(BaseModel):
    amount: float = Field(gt=0, lt=1_000_000)
    hour: int = Field(ge=0, le=23)
    velocity_24h: int = Field(ge=0, le=10_000)
    distance_km: float = Field(ge=0, le=20_000)
    new_device: int = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)


def threshold() -> float:
    if not METRICS_PATH.exists():
        return 0.5
    report = json.loads(METRICS_PATH.read_text())
    return float(report["baseline_logistic"]["threshold"])


def audit(score: float, review: bool) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scores (id INTEGER PRIMARY KEY, score REAL, review INTEGER)"
        )
        conn.execute("INSERT INTO scores (score, review) VALUES (?, ?)", (score, int(review)))


@app.get("/health")
def health() -> dict:
    return {"ok": MODEL_PATH.exists(), "model": "baseline_logistic"}


@app.get("/v1/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        return {"ready": False}
    return json.loads(METRICS_PATH.read_text())


@app.post("/v1/score")
def score(body: ScoreIn) -> dict:
    if not MODEL_PATH.exists():
        return {"ready": False, "detail": "Train the baseline before scoring."}
    model = joblib.load(MODEL_PATH)
    row = np.array(
        [[
            np.log1p(body.amount),
            body.hour,
            body.velocity_24h,
            body.distance_km,
            body.new_device,
            body.merchant_risk,
        ]],
        dtype=np.float32,
    )
    proba = float(model.predict_proba(row)[0, 1])
    cut = threshold()
    review = proba >= cut
    audit(proba, review)
    record_score(
        amount=body.amount,
        hour=body.hour,
        velocity_24h=body.velocity_24h,
        distance_km=body.distance_km,
        new_device=body.new_device,
        merchant_risk=body.merchant_risk,
        score=proba,
        review=review,
    )
    return {
        "score": proba,
        "threshold": cut,
        "review": review,
        "model": "baseline_logistic",
        "action": "none",
        "note": "A review flag is not an account action.",
    }


class RouteScoreIn(ScoreIn):
    key: str = Field(min_length=1, max_length=80)


class InvestigateIn(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=40)
    customer_id: str = Field(min_length=1, max_length=40)
    amount: float = Field(ge=0, lt=1_000_000)
    hour: int = Field(ge=0, le=23)
    velocity_24h: int = Field(ge=0, le=10_000)
    distance_km: float = Field(ge=0, le=20_000)
    new_device: int = Field(ge=0, le=1)
    score: float | None = None
    note: str | None = Field(default=None, max_length=240)


@app.post("/v1/investigate")
def investigate(body: InvestigateIn) -> dict:
    from agents.react_agent import run

    return run(body.model_dump())


@app.get("/v1/monitor")
def monitor() -> dict:
    path = reference_path()
    if not path.exists():
        return {"ready": False, "reason": "reference profile is missing", "action": "none"}
    current = recent(limit=400)
    if len(current) < 30:
        return {
            "ready": False,
            "n": int(len(current)),
            "reason": "fewer than 30 scored rows",
            "action": "none",
        }
    report = drift_report(load_reference(path), current, scores=current["score"].to_numpy())
    report["ready"] = True
    return report


@app.get("/v1/registry")
def registry() -> dict:
    tracking = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking:
        return {
            "ready": False,
            "reason": "MLFLOW_TRACKING_URI is not set",
            "deployed": False,
        }
    from mlops.registry import CHAMPION, MODEL_NAME, alias_version

    version = alias_version(tracking)
    return {
        "ready": version is not None,
        "name": MODEL_NAME,
        "alias": CHAMPION,
        "version": version,
        "tracking_uri": tracking,
        "deployed": False,
        "note": "Local file registry. This process did not deploy to Vertex, KServe, or a cloud account.",
    }


@app.get("/v1/route")
def route() -> dict:
    from mlops.deploy import configured_route, describe, not_ready

    try:
        return describe(configured_route())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return not_ready(str(exc))


@app.post("/v1/route/score")
def route_score(body: RouteScoreIn) -> dict:
    import pandas as pd

    from features.build import matrix
    from mlops.deploy import configured_route, load_version, not_ready, score_route

    try:
        chosen = configured_route()
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return not_ready(str(exc))
    tracking = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking:
        return not_ready("MLFLOW_TRACKING_URI is not set")
    try:
        from mlflow.exceptions import MlflowException
    except ImportError as exc:
        return not_ready(str(exc))
    try:
        models = {
            chosen.primary_version: load_version(tracking, chosen.primary_version),
            chosen.candidate_version: load_version(tracking, chosen.candidate_version),
        }
        frame = pd.DataFrame(
            [
                {
                    "amount": body.amount,
                    "hour": body.hour,
                    "velocity_24h": body.velocity_24h,
                    "distance_km": body.distance_km,
                    "new_device": body.new_device,
                    "merchant_risk": body.merchant_risk,
                }
            ]
        )
        result = score_route(chosen, body.key, models, matrix(frame))
    except (OSError, ValueError, KeyError, ImportError, MlflowException) as exc:
        return not_ready(str(exc))
    cut = threshold()
    review = result["score"] >= cut
    record_score(
        amount=body.amount,
        hour=body.hour,
        velocity_24h=body.velocity_24h,
        distance_km=body.distance_km,
        new_device=body.new_device,
        merchant_risk=body.merchant_risk,
        score=result["score"],
        review=review,
        model=f"baseline_logistic@{result['served_version']}",
    )
    result["ready"] = True
    result["threshold"] = cut
    result["review"] = review
    result["model"] = "baseline_logistic"
    return result

