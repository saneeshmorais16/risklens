#!/usr/bin/env python3
"""Instant case brief for the console. No model call."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def brief(facts: dict) -> str:
    item = facts.get("item") or None
    if not item:
        return "No case is open. The doubt is in the holdout numbers, not in one row."
    score = float(item.get("score") or 0)
    name = str(item.get("transactionId") or "This row")
    line = f"{name} is flagged at {score:.2f}." if item.get("flagged") else f"{name} is under the threshold at {score:.2f}."
    odd = []
    if item.get("newDevice"):
        odd.append("a new device")
    if float(item.get("distanceKm") or 0) >= 80:
        odd.append("a long distance")
    hour = int(item.get("hour") or 12)
    if hour <= 5 or hour >= 23:
        odd.append("an overnight hour")
    if float(item.get("velocity") or 0) >= 4:
        odd.append("a high velocity")
    if not odd:
        return f"{line} Amount, hour, velocity, and distance do not stand out on their own."
    detail = odd[0] if len(odd) == 1 else ", ".join(odd[:-1]) + " and " + odd[-1]
    return f"{line} The unusual part is {detail}."


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/brief":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(length, 200_000))
        try:
            facts = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        body = json.dumps({"text": brief(facts)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8091), Handler).serve_forever()
