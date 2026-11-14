"""Tests for the kakeya-needle-128 instruments (exact scorer, float kernels, search)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from ante.problems.kakeya import exact, fast

DATA = Path(__file__).resolve().parents[1] / "data" / "arena" / "kakeya-needle-128" / "best_solutions.json"


@pytest.fixture(scope="module")
def solutions():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def exact_areas(solutions):
    return [exact.exact_area(s["data"]["offsets"]) for s in solutions]


@pytest.fixture(scope="module")
def leader(solutions):
    return min(solutions, key=lambda s: s["score"])


def _random_config(rng, n=128):
    j = np.arange(n, dtype=np.float64)
    return -0.12 * j / n + rng.uniform(-0.01, 0.01, n)


# ------------------------------------------------------------------ exact scorer


def test_exact_hand_computed_cases():
    # one triangle with base 1 and height 1
    assert exact.union_area_exact([Fraction(0)]) == Fraction(1, 2)
    # n = 2, both offsets 0: U(y) = 1/2 for y <= 1/2 and 1 - y above -> 3/8
    assert exact.union_area_exact([Fraction(0), Fraction(0)]) == Fraction(3, 8)
    # n = 2, x_2 = e: area = 3/8 + e/2 - e^2/2 for small e >= 0
    e = Fraction(1, 10)
    assert exact.union_area_exact([Fraction(0), e]) == Fraction(3, 8) + e / 2 - e * e / 2
    # pairwise disjoint triangles: total area n * 1/(2n)
    assert exact.union_area_exact([Fraction(0), Fraction(5), Fraction(10)]) == Fraction(1, 2)
    # translation invariance
    xs = [Fraction(0), Fraction(-1, 7), Fraction(2, 9), Fraction(1, 100)]
    assert exact.union_area_exact(xs) == exact.union_area_exact([x + Fraction(3, 11) for x in xs])


def test_parse_rules():
    assert exact.parse_offset("-0.25") == Fraction(-1, 4)
    assert exact.parse_offset("+1.5") == Fraction(3, 2)
    assert exact.parse_offset(" 7 ") == 7
    assert exact.parse_offset(".5") == Fraction(1, 2)
    assert exact.parse_offset("0." + "1" * 25) == Fraction(int("1" * 25), 10**25)
    # "\uff11" is a full-width digit: we deliberately accept ASCII digits only
    for bad in ["", "-", ".", "1e-3", "0.1.2", "0x10", "--1", "1/2", "0." + "1" * 26, "12345", "nan", "\uff11"]:
        with pytest.raises(ValueError):
            exact.parse_offset(bad)
    with pytest.raises(ValueError):
        exact.parse_offset(0.5)
    with pytest.raises(ValueError):
        exact.parse_offsets(["0"] * 127)
    with pytest.raises(ValueError):
        exact.parse_offsets(["0"] * 129)
    with pytest.raises(ValueError):
        exact.exact_area("0")
    assert len(exact.parse_offsets(["0"] * 128)) == 128


def test_exact_reproduces_leaderboard(solutions, exact_areas):
    assert len(solutions) == 5
    for s, a in zip(solutions, exact_areas):
        assert isinstance(a, Fraction)
        assert float(a) == s["score"], (s["agentName"], float(a), s["score"])


# ------------------------------------------------------------------ float kernels


def test_fast_matches_exact_on_leaderboard(solutions, exact_areas):
    for s, a in zip(solutions, exact_areas):
        x = fast.to_array(s["data"]["offsets"])
        assert abs(fast.area(x) - float(a)) < 1e-13, s["agentName"]
        f, g = fast.area_and_grad(x)
        assert f == fast.area(x)
        assert abs(g.sum()) < 1e-12


def test_fast_matches_exact_on_small_random_cases():
    rng = np.random.default_rng(12345)
    for n in (1, 2, 3, 5, 8, 13, 21):
        for _ in range(6):
            vals = rng.uniform(-2.5 / n, 2.5 / n, n)
            strings = [f"{v:.6f}" for v in vals]
            fr = [exact.parse_offset(s) for s in strings]
            ref = float(exact.union_area_exact(fr))
            x = np.array([float(v) for v in fr])
            assert abs(fast.area(x) - ref) < 1e-13, (n, strings)


def test_fast_hand_computed_cases():
    assert abs(fast.area(np.array([0.0, 0.0])) - 0.375) < 1e-15
    f, g, h = fast.area_grad_hess(np.array([0.0, 0.0]))
    # area(0, e) = 3/8 + e/2 - e^2/2
    np.testing.assert_allclose(g, [-0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(h, [[-1.0, 1.0], [1.0, -1.0]], atol=1e-13)
    assert abs(fast.area(np.arange(128) * 10.0) - 0.5) < 1e-13


def test_translation_and_mirror_invariance(solutions):
    rng = np.random.default_rng(7)
    xs = [_random_config(rng), fast.to_array(solutions[0]["data"]["offsets"])]
    for x in xs:
        f = fast.area(x)
        assert abs(fast.area(x + 0.3141) - f) < 1e-13
        assert abs(fast.area(fast.mirror(x)) - f) < 1e-13


def test_gradient_matches_finite_differences(solutions):
    rng = np.random.default_rng(2024)
    best = fast.to_array(solutions[0]["data"]["offsets"])
    points = [_random_config(rng), _random_config(rng), best + rng.normal(0.0, 2e-4, best.size)]
    h = 1e-7
    for x in points:
        f, g = fast.area_and_grad(x)
        assert abs(g.sum()) < 1e-12  # translation invariance
        fd = np.empty_like(x)
        for j in range(x.size):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            fd[j] = (fast.area(xp) - fast.area(xm)) / (2 * h)
        # the area is piecewise quadratic and C^1: central differences are exact inside a
        # cell and off by about (Hessian jump) * h / 4 when a cell boundary is crossed
        # (Hessian entries are at most a few hundred)
        assert np.max(np.abs(g)) > 1e-3  # non-trivial gradient at these points
        assert np.max(np.abs(fd - g)) < 1e-4
        # a random direction as well
        d = rng.normal(size=x.size)
        d /= np.linalg.norm(d)
        fd_dir = (fast.area(x + h * d) - fast.area(x - h * d)) / (2 * h)
        assert abs(fd_dir - g @ d) < 1e-4


def test_hessian_is_consistent(solutions):
    rng = np.random.default_rng(99)
    x = _random_config(rng)
    f, g, h = fast.area_grad_hess(x)
    assert np.max(np.abs(h.sum(axis=1))) < 1e-8  # translation invariance
    assert np.max(np.abs(h - h.T)) < 1e-8  # symmetric at a generic point
    # H d versus a finite difference of the gradient; a direction is only spoiled when a
    # cell boundary lies within eps of x, so require a clear majority to agree
    eps = 1e-9
    agree = 0
    for _ in range(5):
        d = rng.normal(size=x.size)
        d /= np.linalg.norm(d)
        _, gp = fast.area_and_grad(x + eps * d)
        _, gm = fast.area_and_grad(x - eps * d)
        if np.max(np.abs((gp - gm) / (2 * eps) - h @ d)) < 1e-3:
            agree += 1
    assert agree >= 4


# ------------------------------------------------------------------ search


def test_polish_returns_to_the_leader(leader):
    from ante.problems.kakeya import search

    x = fast.to_array(leader["data"]["offsets"])
    res0 = search.polish(x)
    assert res0.area <= leader["score"] + 1e-12
    rng = np.random.default_rng(3)
    y = x + rng.normal(0.0, 1e-6, x.size)
    assert fast.area(y) > leader["score"] + 1e-10  # the perturbation is not negligible
    res = search.polish(y)
    assert res.area <= leader["score"] + 1e-10
    assert res.x[0] == 0.0


def test_polish_decreases_area_from_a_rough_start():
    from ante.problems.kakeya import search

    rng = np.random.default_rng(17)
    x = _random_config(rng)
    f0 = fast.area(x)
    res = search.polish(x, max_iter=60)
    assert res.area < f0
    assert abs(fast.area(res.x) - res.area) < 1e-13


def test_moves_produce_valid_configurations(solutions, leader):
    from ante.problems.kakeya import search

    rng = np.random.default_rng(5)
    x0 = fast.to_array(leader["data"]["offsets"])
    ctx = search.Context(donors=[fast.to_array(s["data"]["offsets"]) for s in solutions])
    for name, move in search.MOVES.items():
        produced = 0
        for _ in range(6):
            y = move(x0, rng, ctx)
            if y is None:
                continue
            produced += 1
            assert y.shape == x0.shape and np.all(np.isfinite(y)), name
            assert y is not x0
            assert 0.0 < fast.area(y) <= 0.5 + 1e-12
        if name != "scan_shift":  # a scan may legitimately find no alternative minimum
            assert produced > 0, name
    # the moves must not modify their input
    np.testing.assert_array_equal(x0, fast.to_array(leader["data"]["offsets"]))


def test_offsets_to_strings_respects_input_rules():
    from ante.problems.kakeya import search

    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 0.05, 128)
    strings = search.offsets_to_strings(x)
    assert strings[0] == "0"
    parsed = exact.parse_offsets(strings)
    assert max(abs(float(p) - (v - x[0])) for p, v in zip(parsed, x)) < 1e-15


def test_search_cli_writes_exactly_scored_solution(leader, tmp_path):
    from ante.problems.kakeya import search

    out = tmp_path / "run.json"
    assert search.main(["--seconds", "3", "--seed", "0", "--start", "best", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert set(payload) == {"offsets", "area_float", "exact_area", "meta"}
    area = exact.exact_area(payload["offsets"])  # also enforces the input rules
    assert payload["exact_area"] == f"{area.numerator}/{area.denominator}"
    assert payload["area_float"] == float(area)
    # starting from the leader the search can only keep or improve it
    assert payload["area_float"] <= leader["score"] + 1e-12
