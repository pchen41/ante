import json
from pathlib import Path

import numpy as np
import pytest

from ante.problems.iso import exact, kernel

DATA = Path(__file__).resolve().parents[1] / "data" / "alphaevolve" / "59"


@pytest.mark.parametrize("n,size", [(64, 112), (100, 164)])
def test_check_accepts_the_published_records(n, size):
    obj = json.loads((DATA / f"record_{n}.json").read_text())
    assert exact.check(obj["points"], n) == size


def test_check_rejects_bad_sets():
    with pytest.raises(ValueError):
        exact.check([(0, 0), (1, 0), (2, 0)], 5)  # the middle point is equally far from the other two
    with pytest.raises(ValueError):
        exact.check([(0, 0), (3, 4), (5, 0)], 6)  # (0,0) is at distance 5 from both others
    with pytest.raises(ValueError):
        exact.check([(0, 0), (0, 0)], 5)
    with pytest.raises(ValueError):
        exact.check([(0, 0), (5, 1)], 5)
    assert exact.check([(0, 0), (1, 0), (3, 1)], 5) == 3


@pytest.mark.parametrize("sym", [0, 1])
def test_search_returns_a_valid_set_that_cannot_be_extended(sym):
    n = 10
    cptr, cdx, cdy = kernel.circle_table(n)
    best, _ = kernel.run(n, np.zeros(0, dtype=np.int64), 400, 3, sym, 2, 0.3, 1, 4, True, 0.5, 2, cptr, cdx, cdy, 10**9)
    pts = [(int(c) // n, int(c) % n) for c in best]
    assert exact.check(pts, n) == len(pts) >= 12
    for q in range(n * n):
        cell = (q // n, q % n)
        if cell not in pts:
            with pytest.raises(ValueError):
                exact.check(pts + [cell], n)


def test_orbit_search_and_plugin_return_checked_sets():
    from ante import problems
    from ante.problems.iso import sym

    n = 16
    cptr, cdx, cdy = kernel.circle_table(n)
    orb, osize = sym.orbit_table(n, n - 2, n - 1)  # vertical axis moved half a cell: the proven optimum 28 is of this type
    prio = sym.edge_priority(n, orb)
    best = 0
    for seed in range(6):
        ids, k, _ = sym.run_sym(n, orb, osize, prio, np.zeros(0, dtype=np.int64), 3000, seed, 0.7, 1, 4, 0.6, 4, 0.4, 0.3, 200, cptr, cdx, cdy, 10**9)
        pts = sym.orbits_to_points(n, orb, ids)
        assert exact.check(pts, n) == k
        ids, k2, _ = sym.run_tabu(n, orb, osize, prio, np.zeros(0, dtype=np.int64), 3000, seed, 6, 0.3, 500, 2, cptr, cdx, cdy, 10**9)
        assert exact.check(sym.orbits_to_points(n, orb, ids), n) == k2
        best = max(best, k, k2)
    assert best == 28

    p = problems.get("isosceles-grid-64")
    assert p.record() == 112 and p.target() == 113
    rec = p.public_solutions()[0]["data"]
    assert p.exact_score(rec) == 112 and p.is_public(rec) is not None
    mirrored = {"points": [[63 - x, y] for x, y in rec["points"]]}
    assert p.is_public(mirrored) is not None  # provenance check is modulo the symmetries of the square
    b = p.run("orbits", {"start": "scratch", "chain_seconds": 4}, 1, 6.0, [])
    assert b.best_solution is not None and p.exact_score(b.best_solution) == b.best_score >= 80


def test_priority_expressions_are_whitelisted():
    from ante.core import safe_code

    v = ("x", "y", "n")
    assert safe_code.check("-min(x, y) + (2 if x % 7 == 3 else 0)", v) is None
    for bad in ("__import__('os')", "x.real", "[x][0]", "open('f')", "9**9**9", "(x**2)**2", "z + 1", "lambda: 1", "x; y", ""):
        assert safe_code.check(bad, v) is not None
    spec = __import__("ante.problems", fromlist=["get"]).get("isosceles-grid-64").kernels()["orbits"]
    assert spec.resolve({"priority": "__import__('os').system('true')"})["priority"] == "-min(x, y)"
    assert spec.resolve({"priority": "x * y"})["priority"] == "x * y"


def test_expressions_that_would_raise_are_rejected_or_harmless():
    from ante.core import safe_code

    v = ("x", "y", "n")
    for bad in ("min(x)", "max()", "sqrt()", "abs(x, y)"):
        assert safe_code.check(bad, v) is not None
    f = safe_code.evaluator("1 / (x - 3) + log(y - 5)", v)  # divides by zero at x = 3: scores 0 there, never raises
    assert f(3, 9, 64) == 0.0 and f(4, 9, 64) != 0.0
