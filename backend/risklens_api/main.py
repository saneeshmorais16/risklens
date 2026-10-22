"""Risk scoring API. A high score asks for a review. It does not freeze an account."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

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
    return {
        "score": proba,
        "threshold": cut,
        "review": review,
        "model": "baseline_logistic",
        "action": "none",
        "note": "A review flag is not an account action.",
    }


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

