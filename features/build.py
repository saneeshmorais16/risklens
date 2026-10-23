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


# log_amount is the only derived column. The label never enters this matrix.
def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    return out


# The column order the baseline, RiskNet, and the route all score with.
def matrix(frame: pd.DataFrame) -> np.ndarray:
    featured = add_features(frame)
    return featured.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
