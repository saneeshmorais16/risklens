"""In-process traffic split across registered model versions.

Canary and A/B serve the candidate for a share of keys and the primary
for the rest. The share is candidate_weight. The same key always picks
the same version. Shadow serves the primary and, for that same share of
keys, also scores the candidate. The shadow score is not the decision.

The route names versions. It does not follow the champion alias, and
scoring does not move that alias. Nothing here starts a cluster, a
KServe InferenceService, or a cloud endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlops.registry import MODEL_NAME

MODES = ("canary", "shadow", "ab")
NOTE = (
    "In-process split of local model versions. "
    "No cluster was created, and the champion alias was not moved."
)


@dataclass(frozen=True)
class Route:
    mode: str
    primary_version: str
    candidate_version: str
    candidate_weight: float

    def __post_init__(self) -> None:
        mode = str(self.mode)
        primary = str(self.primary_version).strip()
        candidate = str(self.candidate_version).strip()
        weight = float(self.candidate_weight)
        if mode not in MODES:
            raise ValueError("mode must be canary, shadow, or ab")
        if not primary or not candidate:
            raise ValueError("versions are required")
        if primary == candidate:
            raise ValueError("primary_version and candidate_version must differ")
        if not math.isfinite(weight) or weight < 0 or weight > 1:
            raise ValueError("candidate_weight must be between 0 and 1")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "primary_version", primary)
        object.__setattr__(self, "candidate_version", candidate)
        object.__setattr__(self, "candidate_weight", weight)


def bucket(key: str) -> float:
    """Unit interval from SHA-256. The same key always lands in the same place."""

    text = str(key)
    if text == "":
        raise ValueError("key is required")
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(raw[:4], "big") / 2**32


def _candidate_selected(route: Route, key: str) -> bool:
    return bucket(key) < route.candidate_weight


def assignment(route: Route, key: str) -> dict:
    selected = _candidate_selected(route, key)
    if route.mode == "shadow":
        return {
            "served_version": route.primary_version,
            "served_slot": "primary",
            "shadow_version": route.candidate_version if selected else None,
        }
    version = route.candidate_version if selected else route.primary_version
    return {
        "served_version": version,
        "served_slot": "candidate" if selected else "primary",
        "shadow_version": None,
    }


def score_route(route: Route, key: str, models: dict, row: np.ndarray) -> dict:
    chosen = assignment(route, key)
    served = models[chosen["served_version"]]
    scored = float(served.predict_proba(row)[0, 1])
    shadow_score = None
    if chosen["shadow_version"] is not None:
        shadow_score = float(models[chosen["shadow_version"]].predict_proba(row)[0, 1])
    return {
        "mode": route.mode,
        "key": str(key),
        "primary_version": route.primary_version,
        "candidate_version": route.candidate_version,
        "candidate_weight": route.candidate_weight,
        "served_version": chosen["served_version"],
        "served_slot": chosen["served_slot"],
        "score": scored,
        "shadow_version": chosen["shadow_version"],
        "shadow_score": shadow_score,
        "shadow_served": False,
        "action": "none",
        "cluster": False,
        "deployed": False,
        "alias_moved": False,
        "note": NOTE,
    }


def read_route(path: Path) -> Route:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("route file must be an object")
    missing = [
        name
        for name in ("mode", "primary_version", "candidate_version", "candidate_weight")
        if name not in payload
    ]
    if missing:
        raise ValueError("route file is missing " + ", ".join(missing))
    return Route(
        mode=payload["mode"],
        primary_version=payload["primary_version"],
        candidate_version=payload["candidate_version"],
        candidate_weight=payload["candidate_weight"],
    )


def configured_route() -> Route:
    raw = os.environ.get("ROUTE_PATH", "").strip()
    if not raw:
        raise ValueError("ROUTE_PATH is not set")
    path = Path(raw)
    if not path.is_file():
        raise ValueError("route file is missing")
    return read_route(path)


def describe(route: Route) -> dict:
    return {
        "ready": True,
        "mode": route.mode,
        "primary_version": route.primary_version,
        "candidate_version": route.candidate_version,
        "candidate_weight": route.candidate_weight,
        "action": "none",
        "cluster": False,
        "deployed": False,
        "alias_moved": False,
        "note": NOTE,
    }


def not_ready(reason: str) -> dict:
    return {
        "ready": False,
        "reason": reason,
        "action": "none",
        "cluster": False,
        "deployed": False,
        "alias_moved": False,
    }


def load_version(tracking_uri: str, version: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    found = MlflowClient(tracking_uri=tracking_uri).get_model_version(MODEL_NAME, str(version))
    return mlflow.sklearn.load_model(found.source)
