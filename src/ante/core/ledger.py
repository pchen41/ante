"""Append-only decision ledger.

Every decision the loop makes (and why) is one JSON line. The dashboard, the referee and the README
tables are all derived by replaying this file; nothing else is a source of truth.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator


class Ledger:
    def __init__(self, path: str | os.PathLike, on_write: Callable[[], None] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = sum(1 for _ in self.events()) if self.path.exists() else 0
        self._on_write = on_write

    def append(self, type: str, **data: Any) -> dict:
        event = _strict({"seq": self._seq, "ts": round(time.time(), 3), "type": type, **data})
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=_jsonable, allow_nan=False) + "\n")
        self._seq += 1
        if self._on_write:
            self._on_write()
        return event

    def events(self, type: str | None = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if type is None or event["type"] == type:
                    yield event


def _strict(obj: Any) -> Any:
    """Strict JSON only: non-finite floats become null (an unreachable price is `null`, not Infinity)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_strict(v) for v in obj]
    if hasattr(obj, "item") and not hasattr(obj, "__len__"):  # numpy scalar
        return _strict(obj.item())
    if hasattr(obj, "tolist"):
        return _strict(obj.tolist())
    return obj


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)
