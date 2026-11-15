"""Problem interface: what the loop needs from each research problem.

A problem exposes *kernels* (parameterised search programs). A kernel run produces a batch of
independent *attempts*; the distribution of attempt outcomes is what the loop prices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data" / "arena"


@dataclass
class AttemptBatch:
    scores: list[float]  # one outcome per independent attempt, in the problem's own units
    best_score: float | None  # score of `best_solution` as measured by the kernel (not yet verified)
    best_solution: dict | None  # arena-schema solution
    seconds: float
    meta: dict = field(default_factory=dict)
    start_score: float | None = None  # best score among the known solutions this worker started from (None = from scratch)


@dataclass
class Knob:
    name: str
    kind: str  # "int" | "float" | "choice" | "code" (an arithmetic expression in `variables`, see safe_code.py)
    default: Any
    lo: float | None = None
    hi: float | None = None
    choices: list | None = None
    doc: str = ""
    variables: tuple = ()

    def clamp(self, value: Any) -> Any:
        if self.kind == "code":
            from . import safe_code

            return value.strip() if isinstance(value, str) and safe_code.check(value, self.variables) is None else self.default
        if self.kind == "choice":
            for c in self.choices:
                if str(c).lower() == str(value).strip().lower():
                    return c
            return self.default
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.default
        if v != v:  # NaN
            return self.default
        v = min(max(v, self.lo), self.hi)
        return int(round(v)) if self.kind == "int" else v


@dataclass
class KernelSpec:
    name: str
    doc: str
    knobs: list[Knob]

    def resolve(self, params: dict | None) -> dict:
        params = params or {}
        return {k.name: k.clamp(params.get(k.name, k.default)) for k in self.knobs}

    def describe(self) -> str:
        lines = [f"kernel `{self.name}`: {self.doc}"]
        for k in self.knobs:
            rng = (f"one of {k.choices}" if k.kind == "choice" else f"an arithmetic expression in {', '.join(k.variables)}" if k.kind == "code"
                   else f"{k.kind} in [{k.lo}, {k.hi}]")
            lines.append(f"  - {k.name} ({rng}, default {k.default}): {k.doc}")
        return "\n".join(lines)


class Problem:
    slug: str = ""
    title: str = ""
    maximize: bool = False
    integer_scores: bool = False

    # ---- arena snapshot -------------------------------------------------------------------
    def meta(self) -> dict:
        return json.loads((DATA / self.slug / "meta.json").read_text())

    def leaderboard(self) -> list[dict]:
        return json.loads((DATA / self.slug / "leaderboard.json").read_text())

    def public_solutions(self) -> list[dict]:
        return json.loads((DATA / self.slug / "best_solutions.json").read_text())

    def record(self) -> float:
        return float(self.leaderboard()[0]["bestScore"])

    def target(self, record: float | None = None) -> float:
        """Score needed to take #1: beat the record by the arena's minImprovement."""
        record = self.record() if record is None else record
        margin = float(self.meta()["minImprovement"])
        return record + margin if self.maximize else record - margin

    def better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    # ---- to implement -----------------------------------------------------------------------
    def kernels(self) -> dict[str, KernelSpec]:
        raise NotImplementedError

    def run(self, kernel: str, params: dict, seed: int, seconds: float, elites: list[dict] | None = None) -> AttemptBatch:
        """Run one worker. `elites` are our own best solutions so far (arena schema), best first."""
        raise NotImplementedError

    def exact_score(self, solution: dict) -> float:
        """Independent checker (never the downloaded verifier). Raises ValueError if invalid."""
        raise NotImplementedError

    def canonical(self, solution: dict) -> str:
        """Canonical text of a solution modulo the problem's symmetries (for dedup and provenance)."""
        return json.dumps(solution, sort_keys=True)

    def is_public(self, solution: dict) -> str | None:
        """Name of the agent whose public solution this equals (modulo symmetry), else None."""
        key = self.canonical(solution)
        for s in self.public_solutions():
            if self.canonical(s["data"]) == key:
                return s["agentName"]
        return None

    def lineage(self, kernel: str, params: dict) -> str:
        """Does this strategy build on public solutions? Reported with every candidate."""
        return "from scratch" if params.get("start") in ("random", "scratch") else "builds on public or earlier solutions"

    def audit(self, solution: dict) -> dict:
        """Problem-specific checks that a score is not a verifier artefact."""
        return {}

    def verifier_source(self) -> str:
        """Source of the official (third-party) verifier; it must define evaluate(solution) -> score. Only ever run sandboxed."""
        return (DATA / self.slug / "verifier.py.txt").read_text()

    def brief(self) -> str:
        """What the PI is told about this problem."""
        raise NotImplementedError
