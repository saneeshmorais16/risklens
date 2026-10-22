"""Population stability for features and for the served score.

A high PSI asks for a look at the book. It is not a finding about any account.
The 0.25 cutoff is the usual warning band, not a tuned business rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import FEATURE_COLUMNS, add_features

ROOT = Path(__file__).resolve().parents[1]
PSI_ALERT = 0.25
N_BINS = 10


def reference_path() -> Path:
    override = os.environ.get("REFERENCE_PROFILE_PATH", "").strip()
    if override:
        return Path(override)
    return ROOT / "models" / "reference_profile.json"


def quantile_edges(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(sample, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        center = float(sample[0]) if len(sample) else 0.0
        edges = np.array([center - 1e-6, center + 1e-6])
    return edges


def proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin against the reference cuts, plus an underflow bin and an overflow bin.

    A constant reference still has a bin below its only cut, so a shifted
    batch is visible. Quantile cuts alone would hide that shift.
    """

    sample = np.asarray(values, dtype=float)
    cuts = np.unique(np.asarray(edges, dtype=float))
    index = np.digitize(sample, cuts, right=False)
    counts = np.bincount(index, minlength=len(cuts) + 1).astype(float)
    total = counts.sum()
    if total == 0:
        return np.full(len(cuts) + 1, 1.0 / (len(cuts) + 1))
    return counts / total


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    ref = np.clip(np.asarray(expected, dtype=float), eps, None)
    cur = np.clip(np.asarray(actual, dtype=float), eps, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def profile_column(values: np.ndarray) -> dict:
    sample = np.asarray(values, dtype=float)
    edges = quantile_edges(sample)
    share = proportions(sample, edges)
    return {
        "edges": [float(item) for item in edges],
        "proportions": [float(item) for item in share],
        "mean": float(sample.mean()) if len(sample) else 0.0,
        "n": int(len(sample)),
    }


def column_psi(profile: dict, values: np.ndarray) -> float:
    edges = np.asarray(profile["edges"], dtype=float)
    expected = np.asarray(profile["proportions"], dtype=float)
    return psi(expected, proportions(values, edges))


def profile_frame(frame: pd.DataFrame, scores: np.ndarray | None = None) -> dict:
    featured = add_features(frame)
    report = {
        "psi_alert": PSI_ALERT,
        "features": {
            name: profile_column(featured[name].to_numpy(dtype=float)) for name in FEATURE_COLUMNS
        },
    }
    if scores is not None:
        report["score"] = profile_column(np.asarray(scores, dtype=float))
    return report


def drift_report(
    reference: dict,
    current: pd.DataFrame,
    scores: np.ndarray | None = None,
) -> dict:
    featured = add_features(current)
    features: dict[str, dict] = {}
    alerts: list[str] = []
    for name, column in reference["features"].items():
        value = column_psi(column, featured[name].to_numpy(dtype=float))
        flagged = value >= PSI_ALERT
        features[name] = {"psi": value, "alert": flagged}
        if flagged:
            alerts.append(name)
    score_block = None
    if scores is not None and "score" in reference:
        value = column_psi(reference["score"], np.asarray(scores, dtype=float))
        flagged = value >= PSI_ALERT
        score_block = {"psi": value, "alert": flagged}
        if flagged:
            alerts.append("score")
    return {
        "psi_alert": PSI_ALERT,
        "n": int(len(current)),
        "features": features,
        "score": score_block,
        "alerts": alerts,
        "action": "none",
        "note": "A drift alert asks for a look at the book. It does not change an account.",
    }


def load_reference(path: Path | None = None) -> dict:
    return json.loads((path or reference_path()).read_text(encoding="utf-8"))


def write_reference(path: Path | None = None, n_customers: int = 360) -> dict:
    """Training-window profile for the same seed the saved baseline was fit on."""

    import joblib

    from features.build import matrix
    from training.generate import generate
    from training.train import split

    frame = generate(n_customers=n_customers, seed=42)
    train, _, _ = split(frame)
    model = joblib.load(ROOT / "models" / "baseline.joblib")
    scores = model.predict_proba(matrix(train))[:, 1]
    profile = profile_frame(train, scores)
    profile["source"] = "training days 1-21 from generate(n_customers=360, seed=42)"
    profile["model"] = "baseline_logistic"
    profile["n_customers"] = n_customers
    target = path or reference_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile
