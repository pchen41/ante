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

LAMBDA_NEW_BEST = 0.002  # tie-breaker: "improves our own best" counts for 1/500 of "sets a record"
DEFAULT_RATING = 0.0005  # P(record per $1) assumed for a strategy nobody has rated: an opening price of $2,000
RATING_BOUNDS = (1.0 / 5000.0, 1.0 / 10.0)  # opening prices are kept between $10 and $5,000
RECENT_CARDS = 3  # tail fits use a strategy's most recent cards (its starting pool drifts as elites improve)
W_CLIMB = 0.01  # allocator weight of climbing evidence (a heuristic, not a calibrated probability of a record; logged with every allocation)


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
    start_score: float | None = None  # best known solution its workers started from (None = from scratch)
    started: int = 0  # attempts that began from a known solution ...
    improved: int = 0  # ... and how many of them ended strictly better than it
    rating: float | None = None  # the PI's current P(record per $1); sets the opening price
    ladder: dict = field(default_factory=dict)  # start score -> [attempts started there, attempts that ended strictly better]
    top_started: int = 0  # attempts that began from one of the three best known solutions ...
    top_improved: int = 0  # ... and ended strictly better than it
    record_hits: int = 0  # attempts, over the whole history, that reached the record target
    empty_cards: int = 0  # cards that produced no attempt outcomes
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

    def opening_price(self, trust: float = 1.0) -> float:
        """Dollars to a record assumed before counting this strategy's own failures: 1 / (rated P(record per $1))."""
        r = self.rating if self.rating is not None else (self.prior.get("p_record") or DEFAULT_RATING)
        r = min(max(float(r) * trust, RATING_BOUNDS[0]), RATING_BOUNDS[1])
        return 1.0 / r

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("sid", "problem", "kernel", "params", "hypothesis", "rationale", "source", "proposer", "parent", "prior")}


@dataclass
class ProblemState:
    slug: str
    maximize: bool
    record: float
    target: float
    integer: bool = False
    best: float | None = None  # best independently checked score from our own cards
    best_sha: str | None = None
    elites: list[tuple[float, str, dict]] = field(default_factory=list)  # (score, canonical key, solution), best first

    def better(self, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return a > b if self.maximize else a < b

    def beats_record(self, score: float) -> bool:
        return score >= self.target if self.maximize else score <= self.target

    def add_elite(self, score: float, key: str, solution: dict, keep: int = 16) -> None:
        if any(k == key for _, k, _ in self.elites):
            return
        self.elites.append((score, key, solution))
        self.elites.sort(key=lambda e: -e[0] if self.maximize else e[0])
        del self.elites[keep:]

    def elite_solutions(self) -> list[dict]:
        return [s for _, _, s in self.elites]


class Calibration:
    """Multiplicative recalibration of forecast probabilities, learned per (forecast source, event).

    If a source's forecasts of an event ("new best" or "record") have summed to F while the event
    happened O times, future forecasts of that event from that source are multiplied by
    (O + 1) / (F + 1), clipped to [0.01, 5]. A source that keeps promising records that do not come
    is discounted by up to 100x; one that keeps being surprised by successes is boosted.
    """

    def __init__(self):
        self.sum_f: dict[str, float] = {}
        self.sum_o: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def update(self, source: str, event: str, forecast_p: float, happened: bool) -> None:
        k = f"{source}:{event}"
        self.sum_f[k] = self.sum_f.get(k, 0.0) + float(forecast_p)
        self.sum_o[k] = self.sum_o.get(k, 0.0) + float(bool(happened))
        self.n[k] = self.n.get(k, 0) + 1

    def trust(self, source: str, event: str) -> float:
        k = f"{source}:{event}"
        r = (self.sum_o.get(k, 0.0) + 1.0) / (self.sum_f.get(k, 0.0) + 1.0)
        return float(min(max(r, 0.01), 5.0))

    def apply(self, source: str, event: str, p: float) -> float:
        return float(min(max(p * self.trust(source, event), 0.0), 1.0))

    def table(self) -> dict:
        return {k: {"n": self.n[k], "forecast_sum": round(self.sum_f[k], 4), "observed": self.sum_o[k], "trust": round(self.trust(*k.split(":")), 4)} for k in self.n}


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
        q_rec = pricing.quote(att, ps.maximize, ps.target, cpa, seed=seed, xi_prior=xi0, integer=ps.integer,
                              prior_usd=s.opening_price(self.calibration.trust("prior", "record")),
                              n_total=s.n_attempts(), s_total=s.record_hits)
        if q_rec.fit is not None and q_rec.fit.n * q_rec.fit.q0 >= pricing.MIN_EXCEEDANCES_FOR_SHAPE:
            self.shapes.setdefault(s.problem, {})[s.sid] = q_rec.fit.xi
        # "new best" = better than our best so far AND better than what the strategy started from
        bar = ps.best
        if s.start_score is not None and (bar is None or ps.better(s.start_score, bar)):
            bar = s.start_score
        if bar is None:
            q_best = pricing.Quote("none", len(att), cpa, np.ones(1))
        else:
            eps = 0.5 if ps.integer else abs(bar) * 1e-12
            own = bar + eps if ps.maximize else bar - eps
            nxt = (bar + 1 if ps.maximize else bar - 1) if ps.integer else own
            if ps.beats_record(nxt):
                q_best = q_rec  # beating our best would itself be a record: one event, one forecast
            else:
                q_best = pricing.quote(att, ps.maximize, own, cpa, seed=seed + 1, xi_prior=xi0, integer=ps.integer, sceptical=False)
        return q_rec, q_best

    def forecast(self, s: Strategy, budget_usd: float) -> dict:
        """Pre-registered forecast for spending `budget_usd` on `s`."""
        q_rec, q_best = self.quotes(s)
        if q_rec.model == "none":
            # no evidence yet: the proposer's prior if it gave one, else the base rate of past pilots
            p_rec = float(s.prior.get("p_record") or 0.0)
            return {"source": "prior", "p_new_best": None, "p_record": self.calibration.apply("prior", "record", p_rec),
                    "best_q10": None, "best_q50": s.prior.get("expected_best"), "best_q90": None}
        source = "tail-fit" if q_rec.model == "gpd" else "beta"
        n_future = budget_usd / s.cost_per_attempt() if s.cost_per_attempt() > 0 else 0.0
        qs = pricing.best_of_n_quantiles(s.attempts(), self.problems[s.problem].maximize, n_future, fit=q_rec.fit)
        return {
            "source": source,
            "p_new_best": self.calibration.apply(source, "new_best", q_best.p_within(budget_usd)),
            "p_record": self.calibration.apply(source, "record", q_rec.p_within(budget_usd)),
            "best_q10": qs[0] if qs else None, "best_q50": qs[1] if qs else None, "best_q90": qs[2] if qs else None,
        }

    def price_event(self, s: Strategy) -> dict:
        q, _ = self.quotes(s)
        fit = q.fit
        return {
            "sid": s.sid, "problem": s.problem, "model": q.model, "n_attempts": s.n_attempts(), "cost_per_attempt": s.cost_per_attempt(),
            "opening_price_usd": s.opening_price(self.calibration.trust("prior", "record")), "started_from_known": s.started, "improved_on_start": s.improved,
            "top_started": s.top_started, "top_improved": s.top_improved, "ladder": s.ladder,
            "p_attempt": q.p_attempt, "p_lo": q.p_lo, "p_hi": q.p_hi, "usd_to_record": _finite(q.usd_to_target),
            "xi": fit.xi if fit else None, "sigma": fit.sigma if fit else None,
            "u": (fit.u if self.problems[s.problem].maximize else -fit.u) if fit else None,
        }

    # ---- allocation -------------------------------------------------------------------------
    def allocate(self, tranche_usd: float, k: int, min_share_usd: float, uniform: bool = False) -> tuple[list[dict], list[tuple[Strategy, float]]]:
        """Thompson sampling over priced strategies.

        For each strategy, draw one plausible value of
          P(record within a nominal share of the tranche)                      [success counting, capped tail fit]
        + W_CLIMB * P(an attempt improves a top-3 known solution it started from) [the strongest evidence short of a record]
        + LAMBDA_NEW_BEST * P(new own best within the share)                    [tie-breaker]
        each scaled by how well that forecast source has been calibrated. The top `k` draws are funded,
        and the tranche is split in proportion to their draws (never less than `min_share_usd` each).
        With `uniform=True` (the ablation arm) k live strategies are funded at random with equal shares.
        """
        cands = [s for s in self.live() if s.n_attempts() > 0]
        if not cands:
            return [], []
        k = max(min(k, len(cands)), 1)
        share = max(tranche_usd / k, min_share_usd)
        rows = []
        for s in cands:
            q_rec, q_best = self.quotes(s)
            source = "tail-fit" if q_rec.model == "gpd" else "beta"
            t_rec, t_best = self.calibration.trust(source, "record"), self.calibration.trust(source, "new_best")
            pr = np.minimum(q_rec.p_within_samples(share) * t_rec, 1.0)
            pb = np.minimum(q_best.p_within_samples(share) * t_best, 1.0)
            climb = float(self.rng.beta(0.5 + s.top_improved, 0.5 + s.top_started - s.top_improved)) if s.top_started else 0.0
            draw = float(self.rng.choice(pr)) + W_CLIMB * climb + LAMBDA_NEW_BEST * float(self.rng.choice(pb))
            if uniform:
                draw = float(self.rng.random())
            rows.append({
                "sid": s.sid, "problem": s.problem, "usd_to_record": _finite(q_rec.usd_to_target), "model": source,
                "p_record_tranche": float(pr.mean()), "p_new_best_tranche": float(pb.mean()),
                "climb": [s.top_improved, s.top_started], "draw": draw, "trust": t_rec, "trust_new_best": t_best, "chosen": False, "budget_usd": 0.0,
            })
        rows.sort(key=lambda r: -r["draw"])
        top = rows[:k]
        total = sum(r["draw"] for r in top)
        chosen = []
        for r in top:
            frac = (1.0 / len(top)) if (uniform or total <= 0) else r["draw"] / total
            r["chosen"], r["budget_usd"] = True, max(tranche_usd * frac, min_share_usd)
            chosen.append((self.strategies[r["sid"]], r["budget_usd"]))
        return rows, chosen


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None
