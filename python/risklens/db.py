import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

DB_PATH = Path(os.getenv("RISKLENS_DB_PATH", "risklens.db"))

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT NOT NULL,
                seed INTEGER,
                contamination REAL,
                n_transactions INTEGER,
                n_flagged INTEGER,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scored_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                transaction_id TEXT,
                customer_id TEXT,
                amount REAL,
                hour INTEGER,
                velocity_24h INTEGER,
                distance_km REAL,
                new_device INTEGER,
                anomaly_score REAL,
                flagged INTEGER,
                review_reasons TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scored_transaction_id INTEGER,
                decision TEXT NOT NULL,  -- e.g. 'approved', 'rejected', 'escalated', 'needs_more_info'
                notes TEXT,
                reviewer TEXT,
                reviewed_at TEXT NOT NULL,
                FOREIGN KEY (scored_transaction_id) REFERENCES scored_transactions(id)
            )
        """)
        conn.commit()

def save_analysis(seed: int, contamination: float, result_df, notes: str = "") -> int:
    now = datetime.now(timezone.utc).isoformat()
    flagged_count = int(result_df["flagged"].sum())
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO analyses (run_at, seed, contamination, n_transactions, n_flagged, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (now, seed, contamination, len(result_df), flagged_count, notes)
        )
        analysis_id = cur.lastrowid
        for _, row in result_df.iterrows():
            conn.execute(
                """INSERT INTO scored_transactions
                   (analysis_id, transaction_id, customer_id, amount, hour, velocity_24h,
                    distance_km, new_device, anomaly_score, flagged, review_reasons)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    str(row.get("transaction_id", "")),
                    str(row.get("customer_id", "")),
                    float(row["amount"]),
                    int(row["hour"]),
                    int(row["velocity_24h"]),
                    float(row["distance_km"]),
                    int(row["new_device"]),
                    float(row["anomaly_score"]),
                    int(row["flagged"]),
                    str(row["review_reasons"]),
                )
            )
        conn.commit()
    return analysis_id

def record_review(scored_transaction_id: int, decision: str, notes: str, reviewer: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO reviews (scored_transaction_id, decision, notes, reviewer, reviewed_at) VALUES (?, ?, ?, ?, ?)",
            (scored_transaction_id, decision, notes, reviewer, now)
        )
        conn.commit()

def get_recent_analyses(limit: int = 10) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY run_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

def get_scored_for_analysis(analysis_id: int) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM scored_transactions WHERE analysis_id = ? ORDER BY anomaly_score DESC",
            (analysis_id,)
        ).fetchall()
    return [dict(r) for r in rows]
