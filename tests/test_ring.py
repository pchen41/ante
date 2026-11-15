import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from ante.problems.ring import dp, exact

PUBLIC = json.loads((Path(__file__).resolve().parents[1] / "data" / "arena" / "ring-loading-15" / "best_solutions.json").read_text())


def test_exact_score_reproduces_the_public_instances():
    by_id = {s["id"]: s["data"]["pairs"] for s in PUBLIC}
    assert exact.score(by_id[2576]) == Fraction(9, 8)  # JSAgent-Ring15, listed 1.125
    assert exact.score_fast(by_id[2576]) == (Fraction(9, 8), 46)
    assert exact.score_fast(by_id[2574])[0] == Fraction(47, 42)  # GPT-Explorer, listed 1.119047619047619


def test_dynamic_programme_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(60):
        m, D = int(rng.integers(3, 12)), int(rng.integers(2, 18))
        a = rng.integers(0, D + 1, m).astype(np.int64)
        b = np.array([int(rng.integers(0, D - x + 1)) for x in a], dtype=np.int64)
        A, count = dp.score_int(a, b, 0)
        pairs = [[f"{x}/{D}" if x else "0", f"{y}/{D}" if y else "0"] for x, y in zip(a.tolist(), b.tolist())]
        value, n_min = exact.score_fast(pairs)
        assert Fraction(int(A), D) == value and int(count) == n_min
