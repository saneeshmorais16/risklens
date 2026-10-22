"""Durable score log.

SQLite is the default so the API runs with no database process.
PostgreSQL is used only when DATABASE_URL starts with postgresql://.
That URL is also the cloud-database shape (for example Cloud SQL). This module
does not provision a database.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE = ROOT / "models" / "score_events.db"

FIELDS = (
    "amount",
    "hour",
    "velocity_24h",
    "distance_km",
    "new_device",
    "merchant_risk",
    "log_amount",
    "score",
    "review",
    "model",
    "action",
)

POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS score_events (
  id BIGSERIAL PRIMARY KEY,
  scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  amount DOUBLE PRECISION NOT NULL,
  hour INTEGER NOT NULL,
  velocity_24h INTEGER NOT NULL,
  distance_km DOUBLE PRECISION NOT NULL,
  new_device INTEGER NOT NULL,
  merchant_risk DOUBLE PRECISION NOT NULL,
  log_amount DOUBLE PRECISION NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  review BOOLEAN NOT NULL,
  model TEXT NOT NULL,
  action TEXT NOT NULL
)
"""

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS score_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scored_at TEXT NOT NULL DEFAULT (datetime('now')),
  amount REAL NOT NULL,
  hour INTEGER NOT NULL,
  velocity_24h INTEGER NOT NULL,
  distance_km REAL NOT NULL,
  new_device INTEGER NOT NULL,
  merchant_risk REAL NOT NULL,
  log_amount REAL NOT NULL,
  score REAL NOT NULL,
  review INTEGER NOT NULL,
  model TEXT NOT NULL,
  action TEXT NOT NULL
)
"""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def uses_postgres() -> bool:
    return database_url().startswith("postgresql")


def sqlite_path() -> Path:
    override = os.environ.get("SCORE_LOG_PATH", "").strip()
    if override:
        return Path(override)
    return DEFAULT_SQLITE


def _payload(row: dict) -> dict:
    return {
        "amount": float(row["amount"]),
        "hour": int(row["hour"]),
        "velocity_24h": int(row["velocity_24h"]),
        "distance_km": float(row["distance_km"]),
        "new_device": int(row["new_device"]),
        "merchant_risk": float(row["merchant_risk"]),
        "log_amount": float(np.log1p(row["amount"])),
        "score": float(row["score"]),
        "review": bool(row["review"]),
        "model": str(row.get("model") or "baseline_logistic"),
        "action": "none",
    }


def record_score(**row) -> None:
    payload = _payload(row)
    if uses_postgres():
        _record_postgres(payload)
        return
    _record_sqlite(payload)


def recent(limit: int = 400) -> pd.DataFrame:
    if uses_postgres():
        return _recent_postgres(limit)
    return _recent_sqlite(limit)


def _record_sqlite(payload: dict) -> None:
    path = sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(SQLITE_DDL)
        conn.execute(
            f"INSERT INTO score_events ({', '.join(FIELDS)}) VALUES ({', '.join('?' for _ in FIELDS)})",
            [int(payload["review"]) if name == "review" else payload[name] for name in FIELDS],
        )


def _recent_sqlite(limit: int) -> pd.DataFrame:
    path = sqlite_path()
    if not path.exists():
        return pd.DataFrame(columns=list(FIELDS))
    with sqlite3.connect(path) as conn:
        conn.execute(SQLITE_DDL)
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM score_events ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return pd.DataFrame(rows, columns=list(FIELDS))


def _record_postgres(payload: dict) -> None:
    import psycopg

    with psycopg.connect(database_url()) as conn:
        conn.execute(POSTGRES_DDL)
        conn.execute(
            f"INSERT INTO score_events ({', '.join(FIELDS)}) VALUES ({', '.join('%s' for _ in FIELDS)})",
            [payload[name] for name in FIELDS],
        )


def _recent_postgres(limit: int) -> pd.DataFrame:
    import psycopg

    with psycopg.connect(database_url()) as conn:
        conn.execute(POSTGRES_DDL)
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM score_events ORDER BY id DESC LIMIT %s",
            (int(limit),),
        ).fetchall()
    return pd.DataFrame(rows, columns=list(FIELDS))
