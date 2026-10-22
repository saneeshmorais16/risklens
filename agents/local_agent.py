"""Local investigation loop. No network. It cannot freeze an account."""

from __future__ import annotations

from agents.react_agent import run


def investigate(transaction: dict) -> dict:
    payload = dict(transaction)
    payload.setdefault("transaction_id", "TX-OPEN")
    payload.setdefault("customer_id", "C-OPEN")
    payload.setdefault("history", [])
    return run(payload)
