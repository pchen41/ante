"""Independent exact scoring for the ring loading problem (EinsteinArena `ring-loading-15`; AlphaEvolve problem 61).

An instance is a list of m pairs (u_i, v_i) of nonnegative rationals with u_i + v_i <= 1. An adversary
picks, for every pair, either z_i = v_i or z_i = -u_i. The score is

    A(u, v) = min over the 2^m choices of  max over cuts k = 1..m of  | z_1 + ... + z_k - (z_{k+1} + ... + z_m) |.

Checking a claimed score needs no mathematics: try all 2^m choices in exact fractions. `score` below does
exactly that in about 20 lines; `score_fast` is the same computation in integer arithmetic (numba) for search.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import lcm

import numpy as np
from numba import njit

RATIONAL = re.compile(r"^(?:0|[1-9]\d*)(?:(?:\.\d+)|(?:/[1-9]\d*))?$")
MAX_PAIRS = 24


def parse(pairs) -> list[tuple[Fraction, Fraction]]:
    """Same input rules as the arena verifier (nonnegative decimal or fraction strings, 64-bit numerator and denominator)."""
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS:
        raise ValueError(f"expected 1 to {MAX_PAIRS} pairs")
    out = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each entry must be a pair [u, v]")
        vals = []
        for s in pair:
            if not isinstance(s, str) or not 1 <= len(s) <= 80 or RATIONAL.fullmatch(s) is None:
                raise ValueError("values must be nonnegative decimal or fraction strings")
            f = Fraction(s)
            if f.numerator.bit_length() > 64 or f.denominator.bit_length() > 64:
                raise ValueError("numerator and denominator must fit in 64 bits")
            vals.append(f)
        if vals[0] + vals[1] > 1:
            raise ValueError("every pair must satisfy u + v <= 1")
        out.append((vals[0], vals[1]))
    return out


def score(pairs) -> Fraction:
    """Exact score by exhaustive enumeration in fractions (slow, obviously correct)."""
    uv = parse(pairs)
    m = len(uv)
    best = None
    for mask in range(1 << m):
        z = [-u if mask >> i & 1 else v for i, (u, v) in enumerate(uv)]
        total = sum(z, Fraction(0))
        prefix = Fraction(0)
        worst = Fraction(0)
        for value in z:
            prefix += value
            worst = max(worst, abs(2 * prefix - total))
        if best is None or worst < best:
            best = worst
    return best


@njit(cache=True)
def _min_max_int(u, v):
    """min over masks of max_k |2 * prefix_k - total| for integer u, v. Returns (value, minimising mask, number of minimisers)."""
    m = u.shape[0]
    best = np.int64(1) << 62
    arg = 0
    count = 0
    for mask in range(1 << m):
        total = 0
        for i in range(m):
            total += -u[i] if (mask >> i) & 1 else v[i]
        prefix = 0
        worst = 0
        for i in range(m):
            prefix += -u[i] if (mask >> i) & 1 else v[i]
            d = 2 * prefix - total
            if d < 0:
                d = -d
            if d > worst:
                worst = d
            if worst > best:
                break
        if worst < best:
            best, arg, count = worst, mask, 1
        elif worst == best:
            count += 1
    return best, arg, count


def score_fast(pairs) -> tuple[Fraction, int]:
    """Exact score via integers over a common denominator. Returns (score, number of adversary choices that attain it)."""
    uv = parse(pairs)
    den = 1
    for u, v in uv:
        den = lcm(den, u.denominator, v.denominator)
    if den.bit_length() > 40:
        raise ValueError("common denominator too large for the integer kernel; use score()")
    ui = np.array([int(u * den) for u, _ in uv], dtype=np.int64)
    vi = np.array([int(v * den) for _, v in uv], dtype=np.int64)
    best, _, count = _min_max_int(ui, vi)
    return Fraction(int(best), den), int(count)
