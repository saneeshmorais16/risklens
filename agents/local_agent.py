"""Local investigation loop. No network. It cannot freeze an account."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "rag" / "policies"
MAX_STEPS = 4


def search_policies(query: str, limit: int = 2) -> list[dict]:
    tokens = set(query.lower().split())
    hits: list[tuple[int, str, str]] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text()
        score = sum(1 for token in tokens if token in text.lower())
        if score:
            hits.append((score, path.name, text.strip()))
    hits.sort(key=lambda item: item[0], reverse=True)
    return [{"source": name, "excerpt": body[:280], "score": score} for score, name, body in hits[:limit]]


def investigate(transaction: dict) -> dict:
    steps: list[dict] = []
    if len(steps) >= MAX_STEPS:
        raise RuntimeError("tool budget")
    query = " ".join(
        str(transaction.get(key, ""))
        for key in ("distance_km", "hour", "new_device", "velocity_24h", "amount")
    )
    query = f"review distance device overnight velocity amount {query}"
    hits = search_policies(query)
    steps.append({"tool": "search_policies", "hits": [hit["source"] for hit in hits]})
    factors = []
    if float(transaction.get("distance_km") or 0) >= 80:
        factors.append("distance is far from the customer's usual area")
    if int(transaction.get("hour") or 12) <= 5 or int(transaction.get("hour") or 12) >= 23:
        factors.append("the hour is overnight")
    if int(transaction.get("new_device") or 0) == 1:
        factors.append("the device is new")
    if int(transaction.get("velocity_24h") or 0) >= 4:
        factors.append("velocity is high for this customer")
    steps.append({"tool": "explain_factors", "factors": factors})
    review = bool(factors)
    evidence = ", ".join(hit["source"] for hit in hits) or "no policy matched"
    report = (
        f"Factors: {', '.join(factors) if factors else 'none stand out'}. "
        f"Policy evidence: {evidence}. "
        "Recommendation: an analyst should look."
        if review
        else "Factors: none of the watched signals stand out. Policy evidence: "
        f"{evidence}. Recommendation: no further look is required from these signals."
    )
    return {
        "report": report,
        "evidence": hits,
        "recommend_review": review,
        "steps": steps,
        "action": "none",
    }
