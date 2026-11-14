import math

import numpy as np
import pytest
from scipy import stats

from ante.core import pricing


def _ratio(est, true):
    return max(est, 1e-300) / true


@pytest.mark.parametrize(
    "name,dist,minimize",
    [
        ("normal", stats.norm(loc=1.40, scale=0.02), True),
        ("gumbel", stats.gumbel_l(loc=1.40, scale=0.02), True),
        ("exponential", stats.expon(scale=1.0), False),
    ],
)
def test_tail_extrapolation_two_orders_beyond_the_pilot(name, dist, minimize):
    """A 4000-attempt pilot should price an event of true probability 2.5e-6 within ~1.5 orders of magnitude."""
    p_true = 2.5e-6
    target = dist.ppf(p_true) if minimize else dist.isf(p_true)
    ratios = []
    for seed in range(8):
        x = dist.rvs(size=4000, random_state=seed)
        fit = pricing.fit_tail(x, maximize=not minimize, seed=seed)
        t = -target if minimize else target
        ratios.append(_ratio(pricing.price(fit, t, 1.0).p_attempt, p_true))
    med = float(np.median(ratios))
    assert 1 / 30 < med < 30, (name, ratios)


def test_bounded_distribution_prices_unreachable_target_near_zero():
    """Outcomes with a hard floor (a strategy that plateaus): a target below the floor must look expensive."""
    dist = stats.beta(a=3.0, b=2.0, loc=1.30, scale=0.2)  # support [1.30, 1.50], minimise
    target = 1.26
    ps = []
    for seed in range(6):
        x = dist.rvs(size=4000, random_state=seed)
        fit = pricing.fit_tail(x, maximize=False, seed=seed)
        ps.append(pricing.price(fit, -target, 1.0).p_attempt)
    assert float(np.median(ps)) < 1e-6, ps


def test_target_inside_the_bulk_is_cheap():
    x = stats.norm(0, 1).rvs(size=2000, random_state=0)
    fit = pricing.fit_tail(x, maximize=True)
    pr = pricing.price(fit, 0.0, cost_per_attempt=0.01)
    assert pr.p_attempt >= fit.q0
    assert pr.usd_to_record <= 0.01 / fit.q0 + 1e-12


def test_p_record_within_budget_matches_geometric_formula():
    pr = pricing.Price(cost_per_attempt=0.5, p_attempt=0.01, p_attempt_lo=0.0, p_attempt_hi=0.0, frac_reachable=1.0, usd_to_record=50.0)
    assert pr.p_record_within(0.0) == 0.0
    assert math.isclose(pr.p_record_within(50.0), 1 - 0.99**100, rel_tol=1e-9)
    assert pr.p_record_within(1e9) == pytest.approx(1.0)


def test_integer_outcomes_and_degenerate_inputs():
    rng = np.random.default_rng(0)
    sizes = rng.binomial(30, 0.5, size=3000) + 190  # code sizes around 205
    fit = pricing.fit_tail(sizes, maximize=True)
    p_seen = pricing.price(fit, float(sizes.max()) - 0.5, 1.0).p_attempt
    p_far = pricing.price(fit, float(sizes.max()) + 8, 1.0).p_attempt
    assert p_seen > p_far
    assert 1e-5 < p_seen < 1e-2
    const = pricing.fit_tail([226.0] * 50, maximize=True)
    assert pricing.price(const, 227.0, 1.0).p_attempt == 0.0
    with pytest.raises(ValueError):
        pricing.fit_tail([], maximize=True)


def test_quantile_of_best_is_monotone_in_attempts():
    x = stats.norm(0, 1).rvs(size=3000, random_state=1)
    fit = pricing.fit_tail(x, maximize=True)
    q = [fit.quantile_of_best(n) for n in (1e2, 1e4, 1e6)]
    assert q[0] <= q[1] <= q[2]
    assert 3.0 < q[2] < 7.0  # true median of the max of 1e6 standard normals is about 4.8


def test_small_samples_are_not_optimistic():
    """40 attempts cannot justify a cheap price for a target ten standard deviations away."""
    dist = stats.norm(loc=1.40, scale=0.02)
    ps = []
    for seed in range(8):
        x = dist.rvs(size=40, random_state=seed)
        ps.append(pricing.quote(x, maximize=False, target=1.20, cost_per_attempt=1e-4, seed=seed).p_attempt)
    assert float(np.median(ps)) < 1e-6, ps


def test_unproven_strategy_starts_at_the_sceptical_price_and_gets_dearer_with_failures():
    q0 = pricing.quote([226.0] * 2, maximize=True, target=227.0, cost_per_attempt=0.001, prior_usd=20.0)
    q1 = pricing.quote([226.0] * 20000, maximize=True, target=227.0, cost_per_attempt=0.001, prior_usd=20.0)
    assert q0.model == q1.model == "beta"
    assert 10.0 < q0.usd_to_target < 40.0  # about the opening price
    assert 40.0 < q1.usd_to_target < 90.0  # about opening price + 2 * $20 spent


def test_tail_fit_can_only_make_a_record_look_more_expensive():
    """Outcomes piled up just short of the target (a hard wall) must not be priced below the sceptical prior."""
    rng = np.random.default_rng(0)
    wall = 0.10675139
    x = wall + 1.2e-6 + np.abs(rng.normal(0, 4e-5, size=400))  # minimise; nothing ever gets below wall + 1.2e-6
    q = pricing.quote(x, maximize=False, target=wall - 1e-9, cost_per_attempt=3e-4)
    assert q.model == "gpd"
    assert q.usd_to_target >= 0.9 * (pricing.PRIOR_USD_TO_TARGET + 2 * 400 * 3e-4)
    assert pricing.quote(x, maximize=False, target=wall - 1e-9, cost_per_attempt=3e-4, prior_usd=20.0).usd_to_target >= 0.9 * 20.0
    hopeless = pricing.quote(x + 0.01, maximize=False, target=wall - 1e-9, cost_per_attempt=3e-4)
    assert hopeless.usd_to_target > 100 * q.usd_to_target


def test_own_best_target_is_priced_by_observed_frequency():
    x = np.r_[np.full(90, 1.40), np.full(10, 1.30)]
    q = pricing.quote(x, maximize=False, target=1.35, cost_per_attempt=0.01, sceptical=False)
    assert 0.05 < q.p_attempt < 0.2
