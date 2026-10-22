"""Observable features only. Labels and scenario names never enter the matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "log_amount",
    "hour",
    "velocity_24h",
    "distance_km",
    "new_device",
    "merchant_risk",
)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    return out


def matrix(frame: pd.DataFrame) -> np.ndarray:
    featured = add_features(frame)
    return featured.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
