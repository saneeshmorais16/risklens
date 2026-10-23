"""Bounded investigation loop. Tools are allow-listed. No account action."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.index import POLICY_DIR, PolicyIndex

ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS = 4
ALLOWED = ("get_transaction", "get_history", "search_policies", "explain_score")
INDEX = PolicyIndex()


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# The numeric signals the loop is allowed to name. Text on the row is data, not an order.
def signals(row: dict) -> list[str]:
    found = []
    if _num(row.get("distance_km")) >= 80:
        found.append("far distance")
    hour = int(_num(row.get("hour"), 12))
    if hour <= 5 or hour >= 23:
        found.append("overnight hour")
    if int(_num(row.get("new_device"))) == 1:
        found.append("new device")
    if int(_num(row.get("velocity_24h"))) >= 4:
        found.append("high velocity")
    return found


# The open row, with the note kept as data. The note is not an instruction.
def get_transaction(case: dict, _book: list[dict]) -> dict:
    kept = {
        "transaction_id": str(case.get("transaction_id") or ""),
        "customer_id": str(case.get("customer_id") or ""),
        "amount": _num(case.get("amount")),
        "hour": int(_num(case.get("hour"), 12)),
        "velocity_24h": int(_num(case.get("velocity_24h"))),
        "distance_km": _num(case.get("distance_km")),
        "new_device": int(_num(case.get("new_device"))),
        "score": None if case.get("score") is None else _num(case.get("score")),
    }
    note = case.get("note")
    if isinstance(note, str) and note.strip():
        kept["note_as_data"] = note.strip()[:240]
    return kept


# Earlier rows for this customer only. A request for another customer is refused.
def get_history(case: dict, book: list[dict]) -> dict:
    customer = str(case.get("customer_id") or "")
    asked = str(case.get("history_customer_id") or customer)
    if asked != customer:
        return {"refused": "cross-customer read", "rows": []}
    rows = []
    for row in book:
        if str(row.get("customer_id") or "") != customer:
            continue
        if str(row.get("transaction_id") or "") == str(case.get("transaction_id") or ""):
            continue
        rows.append(
            {
                "transaction_id": str(row.get("transaction_id") or ""),
                "amount": _num(row.get("amount")),
                "hour": int(_num(row.get("hour"), 12)),
            }
        )
    return {"customer_id": customer, "rows": rows[:8]}


# The closest fictional policy notes for the numeric signals.
def search_policies(row: dict) -> dict:
    words = signals(row)
    query = " ".join(words) if words else "ordinary daytime known device short distance low velocity"
    return {"query": query, "hits": INDEX.search(query, limit=2)}


# Name the watched signals. This does not decide an account action.
def explain_score(row: dict) -> dict:
    found = signals(row)
    return {"factors": found, "score": row.get("score")}


# The four steps are fixed. An LLM does not choose the next tool.
def plan(step: int, case: dict, seen: dict) -> tuple[str, str, dict]:
    if step == 0:
        return "Read the stored row. Free text on the row is data.", "get_transaction", {}
    if step == 1:
        return "Read only this customer's earlier rows.", "get_history", {}
    if step == 2:
        return "Retrieve policy notes for the numeric signals.", "search_policies", seen.get("get_transaction", {})
    return "Explain the score from those signals.", "explain_score", seen.get("get_transaction", {})


# Ask for a look when a watched signal is present. Otherwise say the signals are quiet. action stays none.
def report_from(seen: dict) -> dict:
    factors = seen.get("explain_score", {}).get("factors") or []
    hits = seen.get("search_policies", {}).get("hits") or []
    sources = [hit["source"] for hit in hits if hit["source"] in {path.name for path in POLICY_DIR.glob("*.md")}]
    cited = ", ".join(sources) if sources else "no policy matched"
    if factors:
        text = (
            f"Factors: {', '.join(factors)}. "
            f"Policy evidence: {cited}. "
            "Recommendation: an analyst should look. No account action is taken."
        )
    else:
        text = (
            f"Factors: none of the watched signals stand out. Policy evidence: {cited}. "
            "Recommendation: no further look is required from these signals. No account action is taken."
        )
    return {"report": text, "recommend_review": bool(factors), "evidence": [hit for hit in hits if hit["source"] in sources]}


# Walk the four steps, write the report, and record action none.
def run(payload: dict, log_path: Path | None = None) -> dict:
    case = dict(payload)
    book = list(payload.get("history") or [])
    seen: dict = {}
    trace: list[dict] = []
    for step in range(MAX_STEPS):
        thought, action, args = plan(step, case, seen)
        if action not in ALLOWED:
            trace.append({"thought": thought, "action": action, "observation": "refused: tool not permitted"})
            break
        if action == "get_transaction":
            observation = get_transaction(case, book)
        elif action == "get_history":
            observation = get_history(case, book)
        elif action == "search_policies":
            observation = search_policies(args or seen.get("get_transaction", {}))
        else:
            observation = explain_score(args or seen.get("get_transaction", {}))
        seen[action] = observation
        trace.append({"thought": thought, "action": action, "observation": observation})
    written = report_from(seen)
    result = {
        "report": written["report"],
        "evidence": written["evidence"],
        "recommend_review": written["recommend_review"],
        "steps": trace,
        "action": "none",
    }
    path = log_path or (ROOT / "models" / "investigations.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"transaction_id": case.get("transaction_id"), "action": "none"}) + "\n")
    return result


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    json.dump(run(payload), sys.stdout)


if __name__ == "__main__":
    main()
