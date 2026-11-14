"""EinsteinArena client. Reads are public GETs. Writes are deliberately not implemented in the loop:
registration, submissions and posts require a human decision (see `scripts/arena_submit.py`).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

BASE = "https://einsteinarena.com"


def get(path: str, timeout: float = 30.0, **params):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "ante/0.1 (read-only)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def leaderboard(problem_id: int, limit: int = 10) -> list[dict] | None:
    """Live leaderboard, or None if the arena is unreachable (the caller falls back to the snapshot)."""
    try:
        rows = get("/api/leaderboard", problem_id=problem_id, limit=limit)
        return rows if isinstance(rows, list) and rows else None
    except Exception:
        return None
