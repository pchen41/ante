"""The strategy market: every strategy has a price (expected dollars to a record), every dollar is
allocated against prices, and every allocation carries a forecast that the referee scores later.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import pricing

LAMBDA_NEW_BEST = 0.05  # weight of "improves our own best" relative to "sets a record" in the allocator
RECENT_CARDS = 3  # a strategy is priced on its most recent cards (its starting pool drifts as elites improve)


def solution_sha(solution: dict) -> str:
    return hashlib.sha256(json.dumps(solution, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Strategy:
    sid: str
    problem: str
    kernel: str
    params: dict
    hypothesis: str
    rationale: str = ""
    source: str = "seed"
    proposer: str = "seed"
    parent: str | None = None
    prior: dict = field(default_factory=dict)
    # accumulated evidence
    batches: list[list[float]] = field(default_factory=list)  # attempt outcomes, one list per card
    core_seconds: float = 0.0
    cost_usd: float = 0.0
    best: float | None = None
    cards: int = 0
    funded: int = 0  # number of scale cards
    retired: bool = False

    def attempts(self) -> list[float]:
        return [s for b in self.batches[-RECENT_CARDS:] for s in b]

    def n_attempts(self) -> int:
        return sum(len(b) for b in self.batches)

    def cost_per_attempt(self) -> float:
        n = self.n_attempts()
        return self.cost_usd / n if n else float("inf")

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("sid", "problem", "kernel", "params", "hypothesis", "rationale", "source", "proposer", "parent", "prior")}


@dataclass
class ProblemState:
    slug: str
    maximize: bool
    record: float
    target: float
    best: float | None = None  # best independently checked score from our own cards
    best_sha: str | None = None
    elites: list[tuple[float, str, dict]] = field(default_factory=list)  # (score, canonical key, solution), best first

    def better(self, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return a > b if self.maximize else a < b

    def beats_record(self, score: float) -> bool:
        return score >= self.target if self.maximize else score <= self.target

    def add_elite(self, score: float, key: str, solution: dict, keep: int = 8) -> None:
        if any(k == key for _, k, _ in self.elites):
            return
        self.elites.append((score, key, solution))
        self.elites.sort(key=lambda e: -e[0] if self.maximize else e[0])
        del self.elites[keep:]

    def elite_solutions(self) -> list[dict]:
        return [s for _, _, s in self.elites]


class Calibration:
    """Multiplicative recalibration of forecast probabilities, learned per forecast source.

    If a source's forecasts of "new best" have summed to F while the event happened O times, future
    forecasts from that source are multiplied by (O + 1) / (F + 1), clipped to [0.2, 5].
    """

    def __init__(self):
        self.sum_f: dict[str, float] = {}
        self.sum_o: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def update(self, source: str, forecast_p: float, happened: bool) -> None:
        self.sum_f[source] = self.sum_f.get(source, 0.0) + float(forecast_p)
        self.sum_o[source] = self.sum_o.get(source, 0.0) + float(bool(happened))
        self.n[source] = self.n.get(source, 0) + 1

    def trust(self, source: str) -> float:
        r = (self.sum_o.get(source, 0.0) + 1.0) / (self.sum_f.get(source, 0.0) + 1.0)
        return float(min(max(r, 0.2), 5.0))

    def apply(self, source: str, p: float) -> float:
        return float(min(max(p * self.trust(source), 0.0), 1.0))


class Market:
    def __init__(self, seed: int = 0):
        self.strategies: dict[str, Strategy] = {}
        self.problems: dict[str, ProblemState] = {}
        self.calibration = Calibration()
        self.shapes: dict[str, dict[str, float]] = {}  # problem -> sid -> fitted tail shape
        self.rng = np.random.default_rng(seed)
        self._n = 0

    # ---- strategies -------------------------------------------------------------------------
    def new_sid(self) -> str:
        self._n += 1
        return f"s{self._n:03d}"

    def add(self, s: Strategy) -> Strategy:
        self.strategies[s.sid] = s
        return s

    def live(self, problem: str | None = None) -> list[Strategy]:
        return [s for s in self.strategies.values() if not s.retired and (problem is None or s.problem == problem)]

    def is_duplicate(self, problem: str, kernel: str, params: dict) -> bool:
        return any(s.problem == problem and s.kernel == kernel and s.params == params for s in self.strategies.values())

    # ---- pricing ----------------------------------------------------------------------------
    def xi_prior(self, problem: str) -> float:
        """Tail shape borrowed from this problem's well-sampled strategies (median), else the default."""
        xs = self.shapes.get(problem, {})
        return float(np.median(list(xs.values()))) if xs else pricing.XI_PRIOR

    def quotes(self, s: Strategy) -> tuple[pricing.Quote, pricing.Quote]:
        """(quote against the arena record target, quote against our own best)."""
        ps = self.problems[s.problem]
        att, cpa = s.attempts(), s.cost_per_attempt()
        seed = int(self.rng.integers(1 << 31))
        xi0 = self.xi_prior(s.problem)
        q_rec = pricing.quote(att, ps.maximize, ps.target, cpa, seed=seed, xi_prior=xi0)
        if q_rec.fit is not None and q_rec.fit.n * q_rec.fit.q0 >= pricing.MIN_EXCEEDANCES_FOR_SHAPE:
            self.shapes.setdefault(s.problem, {})[s.sid] = q_rec.fit.xi
        if ps.best is None:
            q_best = pricing.Quote("none", len(att), cpa, np.ones(1))
        else:
            eps = 0.5 if float(ps.best).is_integer() and ps.maximize else abs(ps.best) * 1e-12
            own = ps.best + eps if ps.maximize else ps.best - eps
            q_best = pricing.quote(att, ps.maximize, own, cpa, seed=seed + 1, xi_prior=xi0)
        return q_rec, q_best

    def forecast(self, s: Strategy, budget_usd: float) -> dict:
        """Pre-registered forecast for spending `budget_usd` on `s`."""
        q_rec, q_best = self.quotes(s)
        if q_rec.model == "none":
            # no evidence yet: the proposer's prior if it gave one, else the base rate of past pilots
            p_rec = float(s.prior.get("p_record") or 0.0)
            return {"source": "prior", "p_new_best": None, "p_record": self.calibration.apply("prior", p_rec),
                    "best_q10": None, "best_q50": s.prior.get("expected_best"), "best_q90": None}
        source = "tail-fit" if q_rec.model == "gpd" else "beta"
        n_future = budget_usd / s.cost_per_attempt() if s.cost_per_attempt() > 0 else 0.0
        qs = pricing.best_of_n_quantiles(s.attempts(), self.problems[s.problem].maximize, n_future, fit=q_rec.fit)
        return {
            "source": source,
            "p_new_best": self.calibration.apply(source, q_best.p_within(budget_usd)),
            "p_record": self.calibration.apply(source, q_rec.p_within(budget_usd)),
            "best_q10": qs[0] if qs else None, "best_q50": qs[1] if qs else None, "best_q90": qs[2] if qs else None,
        }

    def price_event(self, s: Strategy) -> dict:
        q, _ = self.quotes(s)
        fit = q.fit
        return {
            "sid": s.sid, "problem": s.problem, "model": q.model, "n_attempts": s.n_attempts(), "cost_per_attempt": s.cost_per_attempt(),
            "p_attempt": q.p_attempt, "p_lo": q.p_lo, "p_hi": q.p_hi, "usd_to_record": _finite(q.usd_to_target),
            "xi": fit.xi if fit else None, "sigma": fit.sigma if fit else None,
            "u": (fit.u if self.problems[s.problem].maximize else -fit.u) if fit else None,
        }

    # ---- allocation -------------------------------------------------------------------------
    def allocate(self, tranche_usd: float, k: int, min_share_usd: float) -> tuple[list[dict], list[tuple[Strategy, float]]]:
        """Thompson sampling over priced strategies.

        For each strategy draw one plausible value of P(attempt reaches the record) and of P(attempt
        beats our best), turn them into P(event within an equal share of the tranche), and rank by
        P(record) + LAMBDA * P(new best), scaled by how well that forecast source has been calibrated.
        The top `k` are funded with equal shares.
        """
        cands = [s for s in self.live() if s.n_attempts() > 0]
        if not cands:
            return [], []
        share = max(tranche_usd / max(min(k, len(cands)), 1), min_share_usd)
        rows = []
        for s in cands:
            q_rec, q_best = self.quotes(s)
            source = "tail-fit" if q_rec.model == "gpd" else "beta"
            trust = self.calibration.trust(source)
            pr = q_rec.p_within_samples(share)
            pb = q_best.p_within_samples(share)
            draw = float(self.rng.choice(pr)) + LAMBDA_NEW_BEST * float(self.rng.choice(pb))
            rows.append({
                "sid": s.sid, "problem": s.problem, "usd_to_record": _finite(q_rec.usd_to_target),
                "p_record_tranche": float(min(pr.mean() * trust, 1.0)), "p_new_best_tranche": float(min(pb.mean() * trust, 1.0)),
                "draw": draw * trust, "trust": trust, "chosen": False, "budget_usd": 0.0,
            })
        rows.sort(key=lambda r: -r["draw"])
        chosen = []
        for r in rows[:k]:
            r["chosen"], r["budget_usd"] = True, share
            chosen.append((self.strategies[r["sid"]], share))
        return rows, chosen


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None
