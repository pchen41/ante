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
