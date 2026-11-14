"""Pilot-based pricing: how many dollars until this strategy beats the record?

A strategy reports the outcome of each independent *attempt* (one restart, one basin hop, one
episode). From a cheap pilot we fit the tail of the attempt-outcome distribution with a generalized
Pareto distribution (peaks over threshold, standard extreme-value theory) and extrapolate

    p = P(one attempt reaches the target)        ->  expected cost to record = cost_per_attempt / p
    P(record within budget B) = 1 - (1 - p)^(B / cost_per_attempt)

Parameter uncertainty is carried by bootstrap replicates of the tail fit; the allocator Thompson-samples
from those replicates, and the referee later scores these forecasts against what actually happened.

All scores are internally oriented so that larger is better (minimization problems are negated).

Shape constraint: a search for the optimum of a bounded objective has bounded outcomes, so the tail
lies in the Weibull domain or, at the optimistic limit, is exponential. We therefore constrain
xi <= 0. Without the constraint a 60-point tail sample can fit a spurious heavy tail and promise
records that the strategy cannot reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import genpareto

XI_LO, XI_HI = -1.0, 0.0


@dataclass
class TailFit:
    n: int  # attempts observed
    q0: float  # tail fraction used for the fit
    u: float  # threshold (oriented: larger is better)
    xi: float
    sigma: float
    boot: list[tuple[float, float, float]] = field(default_factory=list)  # (u, xi, sigma) replicates
    best: float = float("nan")  # best oriented attempt

    def p_exceed(self, target: float, params: tuple[float, float, float] | None = None) -> float:
        """P(a single attempt is >= target) under the fitted tail."""
        u, xi, sigma = params or (self.u, self.xi, self.sigma)
        if target <= u:
            return self.q0  # inside the bulk: at least the tail mass (conservative)
        return float(self.q0 * genpareto.sf(target - u, xi, loc=0.0, scale=sigma))

    def p_exceed_samples(self, target: float) -> np.ndarray:
        if not self.boot:
            return np.array([self.p_exceed(target)])
        return np.array([self.p_exceed(target, b) for b in self.boot])

    def quantile_of_best(self, n_attempts: float, level: float = 0.5) -> float:
        """Level-quantile of the best of `n_attempts` future attempts (oriented)."""
        if n_attempts <= 0:
            return float("nan")
        # P(max <= x) = (1 - p(x))^n = level  ->  p(x) = 1 - level^(1/n)
        p = 1.0 - level ** (1.0 / n_attempts)
        if p >= self.q0:
            return self.u
        return float(self.u + genpareto.isf(p / self.q0, self.xi, loc=0.0, scale=self.sigma))


XI_PRIOR = -0.2  # tail shape assumed when a sample is too small to estimate it (bounded, as for the large samples we measured)
MIN_EXCEEDANCES_FOR_SHAPE = 30


def _fit_gpd(exc: np.ndarray, xi_prior: float = XI_PRIOR) -> tuple[float, float]:
    """Fit GPD(xi, sigma) to positive exceedances.

    The shape xi is only estimated from 30+ exceedances. Smaller samples borrow the shape (from
    better-sampled strategies on the same problem when available) and fit the scale by moments:
    E[excess] = sigma / (1 - xi).
    """
    exc = exc[exc > 0]
    if exc.size == 0:
        return xi_prior, 1e-12
    if exc.size < MIN_EXCEEDANCES_FOR_SHAPE:
        xi = float(np.clip(xi_prior, XI_LO, XI_HI))
        sigma = float(exc.mean()) * (1.0 - xi)
    else:
        try:
            xi, _, sigma = genpareto.fit(exc, floc=0.0)
        except Exception:
            xi, sigma = xi_prior, float(exc.mean()) * (1.0 - xi_prior)
        if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
            xi, sigma = xi_prior, float(exc.mean()) * (1.0 - xi_prior)
        xi = float(np.clip(xi, XI_LO, XI_HI))
    if xi < 0:
        # the fitted endpoint u + sigma/|xi| must not sit below the largest observation
        sigma = max(float(sigma), float(exc.max()) * (-xi) * 1.0000001)
    return xi, float(max(sigma, 1e-300))


def fit_tail(attempts, maximize: bool, q0: float | None = None, n_boot: int = 200, seed: int = 0, xi_prior: float = XI_PRIOR) -> TailFit:
    x = np.asarray(attempts, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not maximize:
        x = -x
    n = x.size
    if n == 0:
        raise ValueError("no attempts to fit")
    if q0 is None:
        # aim for ~120 exceedances, never more than the top 20% nor fewer than the top 2%
        q0 = float(np.clip(120.0 / n, 0.02, 0.25))
    if np.ptp(x) == 0:
        return TailFit(n=n, q0=q0, u=float(x[0]), xi=0.0, sigma=1e-12, boot=[], best=float(x[0]))
    # integer-valued outcomes (e.g. code sizes): jitter so the continuous tail model is identifiable
    if np.all(x == np.round(x)):
        x = x + np.random.default_rng(seed).uniform(-0.5, 0.5, size=n)
    u = float(np.quantile(x, 1.0 - q0))
    xi, sigma = _fit_gpd(x[x > u] - u, xi_prior)
    rng = np.random.default_rng(seed + 1)
    boot = []
    for _ in range(n_boot):
        xb = x[rng.integers(0, n, n)]
        ub = float(np.quantile(xb, 1.0 - q0))
        xib, sb = _fit_gpd(xb[xb > ub] - ub, xi_prior)
        boot.append((ub, xib, sb))
    return TailFit(n=n, q0=q0, u=u, xi=xi, sigma=sigma, boot=boot, best=float(x.max()))


@dataclass
class Price:
    """What a pilot says about a strategy, in dollars."""

    cost_per_attempt: float
    p_attempt: float  # posterior-mean P(one attempt hits the record target)
    p_attempt_lo: float  # 10th percentile over bootstrap replicates
    p_attempt_hi: float  # 90th percentile
    frac_reachable: float  # share of replicates whose fitted support includes the target
    usd_to_record: float  # cost_per_attempt / p_attempt (inf if p == 0)

    def p_record_within(self, usd: float, p: float | None = None) -> float:
        p = self.p_attempt if p is None else p
        if p <= 0 or usd <= 0:
            return 0.0
        n = usd / self.cost_per_attempt
        return float(-math.expm1(n * math.log1p(-min(p, 1 - 1e-12))))


def price(fit: TailFit, target_oriented: float, cost_per_attempt: float) -> Price:
    ps = fit.p_exceed_samples(target_oriented)
    p = float(ps.mean())
    return Price(
        cost_per_attempt=cost_per_attempt,
        p_attempt=p,
        p_attempt_lo=float(np.quantile(ps, 0.1)),
        p_attempt_hi=float(np.quantile(ps, 0.9)),
        frac_reachable=float((ps > 0).mean()),
        usd_to_record=(cost_per_attempt / p) if p > 0 else float("inf"),
    )


def p_record_samples(fit: TailFit, target_oriented: float, cost_per_attempt: float, usd: float) -> np.ndarray:
    """P(record within `usd`) for each bootstrap replicate (used for Thompson sampling)."""
    ps = np.clip(fit.p_exceed_samples(target_oriented), 0.0, 1 - 1e-12)
    n = usd / max(cost_per_attempt, 1e-12)
    return -np.expm1(n * np.log1p(-ps))


# ---------------------------------------------------------------------------------------------
# Quotes: one interface over the two pricing models
# ---------------------------------------------------------------------------------------------


@dataclass
class Quote:
    """The current price of a strategy against one target."""

    model: str  # "gpd" (tail extrapolation) | "beta" (success counting) | "none" (no data yet)
    n_attempts: int
    cost_per_attempt: float
    p_samples: np.ndarray  # plausible values of P(one attempt reaches the target)
    fit: TailFit | None = None

    @property
    def p_attempt(self) -> float:
        return float(self.p_samples.mean()) if self.p_samples.size else 0.0

    @property
    def p_lo(self) -> float:
        return float(np.quantile(self.p_samples, 0.1)) if self.p_samples.size else 0.0

    @property
    def p_hi(self) -> float:
        return float(np.quantile(self.p_samples, 0.9)) if self.p_samples.size else 0.0

    @property
    def usd_to_target(self) -> float:
        p = self.p_attempt
        return self.cost_per_attempt / p if p > 0 else float("inf")

    def p_within_samples(self, usd: float) -> np.ndarray:
        if not self.p_samples.size or usd <= 0 or self.cost_per_attempt <= 0:
            return np.zeros(1)
        n = usd / self.cost_per_attempt
        return -np.expm1(n * np.log1p(-np.clip(self.p_samples, 0.0, 1 - 1e-12)))

    def p_within(self, usd: float) -> float:
        return float(self.p_within_samples(usd).mean())


PRIOR_USD_TO_TARGET = 20.0  # before any evidence, no strategy is assumed cheaper than this
MIN_ATTEMPTS_FOR_TAIL_FIT = 30


def quote(attempts, maximize: bool, target: float, cost_per_attempt: float, seed: int = 0, n_samples: int = 200,
          prior_usd: float = PRIOR_USD_TO_TARGET, xi_prior: float = XI_PRIOR, integer: bool = False) -> Quote:
    """Price `target` from the attempt outcomes seen so far.

    Continuous outcomes with real spread and a reasonable sample get a tail fit. Otherwise (few
    attempts, or integer scores, whose distribution is a few atoms set by where the search started)
    nothing can be extrapolated, so we count successes under a sceptical prior: with s successes in
    n attempts,

        P(attempt succeeds) ~ Beta(1/2 + s, prior_usd / (2 * cost_per_attempt) + n - s).

    With no successes this prices the strategy at `prior_usd + 2 * (dollars already spent on it)`:
    every dollar spent without reaching the target makes the next dollar look worse.

    A tail fit is also capped by the success count: it may not claim a higher success probability
    than a draw from Beta(1/2 + s, 1/2 + n - s), the plain posterior after n attempts.
    """
    x = np.asarray(attempts, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return Quote("none", 0, cost_per_attempt, np.zeros(0))
    s = int((x >= target).sum() if maximize else (x <= target).sum())
    rng = np.random.default_rng(seed)
    if integer or np.unique(x).size < 4 or x.size < MIN_ATTEMPTS_FOR_TAIL_FIT:
        b0 = prior_usd / (2.0 * max(cost_per_attempt, 1e-12))
        return Quote("beta", int(x.size), cost_per_attempt, rng.beta(0.5 + s, b0 + x.size - s, size=n_samples))
    fit = fit_tail(x, maximize=maximize, n_boot=n_samples, seed=seed, xi_prior=xi_prior)
    t = target if maximize else -target
    ps = fit.p_exceed_samples(t)
    cap = rng.beta(0.5 + s, 0.5 + x.size - s, size=ps.size)
    return Quote("gpd", int(x.size), cost_per_attempt, np.minimum(ps, cap), fit)


def best_of_n_quantiles(attempts, maximize: bool, n_future: float, levels=(0.1, 0.5, 0.9), fit: TailFit | None = None, seed: int = 0) -> list[float] | None:
    """Predictive quantiles (in the problem's units) of the best outcome among `n_future` new attempts."""
    x = np.asarray(attempts, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 5 or n_future < 1:
        return None
    if fit is not None:
        # levels are for "how good": q10 = pessimistic, q90 = optimistic, in oriented (larger = better) space
        vals = [fit.quantile_of_best(n_future, lv) for lv in levels]
        return [float(v if maximize else -v) for v in vals]
    rng = np.random.default_rng(seed)
    n = int(min(max(n_future, 1), 100_000))
    draws = rng.choice(x, size=(400, n), replace=True)
    best = draws.max(axis=1) if maximize else draws.min(axis=1)
    qs = np.quantile(best, levels if maximize else [1 - lv for lv in levels])
    return [float(v) for v in qs]
