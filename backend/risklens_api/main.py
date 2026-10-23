"""Risk scoring API. A high score asks for a review. It does not freeze an account."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops.drift import MIN_ROWS, drift_report, load_reference, look_at, reference_path, version_drift
from mlops.score_log import recent, record_score, version_mix

VERSION_MIX_NOTE = "Counts of served rows by model. PSI above is for the whole book."
BY_VERSION_NOTE = (
    "PSI for one served model. Ready only when that model has at least 30 rows. "
    "The book PSI is still the whole window."
)

# The baseline file is the model this API serves. metrics.json is the last training run, not a target.
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "baseline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DB_PATH = ROOT / "models" / "scores.db"

app = FastAPI(title="RiskLens AI", version="0.1.0")


# The numbers a caller is allowed to send. The score sees these fields and nothing else.
class ScoreIn(BaseModel):
    amount: float = Field(gt=0, lt=1_000_000)
    hour: int = Field(ge=0, le=23)
    velocity_24h: int = Field(ge=0, le=10_000)
    distance_km: float = Field(ge=0, le=20_000)
    new_device: int = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)


# The cutoff chosen on days 22-24. A missing metrics file falls back to 0.5.
def threshold() -> float:
    if not METRICS_PATH.exists():
        return 0.5
    report = json.loads(METRICS_PATH.read_text())
    return float(report["baseline_logistic"]["threshold"])


# A small local note of the score and the review flag. The durable log is record_score.
def audit(score: float, review: bool) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scores (id INTEGER PRIMARY KEY, score REAL, review INTEGER)"
        )
        conn.execute("INSERT INTO scores (score, review) VALUES (?, ?)", (score, int(review)))


# The baseline file is on disk, or it is not. This does not score a row.
@app.get("/health")
def health() -> dict:
    return {"ok": MODEL_PATH.exists(), "model": "baseline_logistic"}


# The last local training run. These numbers are what happened. They are not a target.
@app.get("/v1/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        return {"ready": False}
    return json.loads(METRICS_PATH.read_text())


# Score the checked-in baseline. A high score asks for a look. action stays none.
@app.post("/v1/score")
def score(body: ScoreIn) -> dict:
    if not MODEL_PATH.exists():
        return {"ready": False, "detail": "Train the baseline before scoring."}
    model = joblib.load(MODEL_PATH)
    # Same six columns the baseline and the route both score. The label is not in this row.
    import pandas as pd

    from features.build import matrix

    row = matrix(
        pd.DataFrame(
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


# The route needs a key as well as the row. The same key keeps the same version.
class RouteScoreIn(ScoreIn):
    key: str = Field(min_length=1, max_length=80)


# The case the investigation loop is allowed to read. Free text on the row is data.
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


# Four allow-listed steps. The loop does not pick its own tools, and it takes no account action.
@app.post("/v1/investigate")
def investigate(body: InvestigateIn) -> dict:
    from agents.react_agent import run

    return run(body.model_dump())


# One book PSI, a PSI per ready model, and a look at the model that disagrees. action stays none.
@app.get("/v1/monitor")
def monitor() -> dict:
    path = reference_path()
    if not path.exists():
        return {"ready": False, "reason": "reference profile is missing", "action": "none"}
    current = recent(limit=400)
    mix = version_mix(current)
    reference = load_reference(path)
    by_version = version_drift(reference, current)
    look = look_at(by_version)
    if len(current) < MIN_ROWS:
        return {
            "ready": False,
            "n": int(len(current)),
            "reason": f"fewer than {MIN_ROWS} scored rows",
            "versions": mix,
            "versions_note": VERSION_MIX_NOTE,
            "by_version": by_version,
            "by_version_note": BY_VERSION_NOTE,
            "look": look,
            "action": "none",
        }
    report = drift_report(reference, current, scores=current["score"].to_numpy())
    report["ready"] = True
    report["versions"] = mix
    report["versions_note"] = VERSION_MIX_NOTE
    report["by_version"] = by_version
    report["by_version_note"] = BY_VERSION_NOTE
    report["look"] = look
    return report


# Read the champion alias from a local file store. deployed stays false. This claims no cluster.
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


# Read the route file. A missing or bad file is ready false, still with action none.
@app.get("/v1/route")
def route() -> dict:
    from mlops.deploy import configured_route, describe, not_ready

    try:
        return describe(configured_route())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return not_ready(str(exc))


# Score the version the route picked. Log the served row only. Leave the champion alias where it is.
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
    # The shadow score stays in the response. It does not get a row in the score log.
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

