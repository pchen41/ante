"""Independent exact scorer for the EinsteinArena problem ``kakeya-needle-128``.

For j = 1..n the triangle T_j = conv{(x_j, 0), (x_j + 1/n, 0), (x_j + j/n, 1)} has the
horizontal section

    [ x_j + (j/n) y ,  x_j + (j/n) y + (1 - y)/n ]            (0 <= y <= 1)

i.e. at a fixed height every section has the same width (1 - y)/n.  The union length
U(y) is continuous and affine between consecutive heights at which two endpoint lines
cross, so the trapezoid rule on the sorted crossing heights integrates it exactly.

Everything is done in integer arithmetic on one common denominator:

* offsets (after subtracting x_1) are X_j / den,
* every crossing height is p / (den * dk) with 1 <= dk <= n, hence an integer multiple
  of 1 / (den * L) with L = lcm(1..n); heights are stored as the integer ``key`` with
  y = key / unit, unit = den * L,
* at y = key / unit the left endpoints are (n X_j L + j key) / (n unit) and the common
  width is (unit - key) / (n unit).

This file was written from the problem statement; it does not import or execute the
downloaded arena verifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import lcm

N = 128
MAX_FRAC_DIGITS = 25
MAX_INT_DIGITS = 4

_DIGITS = frozenset("0123456789")


def parse_offset(s: object) -> Fraction:
    """Parse one base-10 decimal string under the arena's input rules."""
    if not isinstance(s, str):
        raise ValueError("each offset must be a base-10 decimal string, got " + repr(s))
    t = s.strip()
    negative = False
    if t[:1] == "-":
        negative = True
        t = t[1:]
    elif t[:1] == "+":
        t = t[1:]
    if t.count(".") > 1:
        raise ValueError("malformed decimal string: " + repr(s))
    int_part, _, frac_part = t.partition(".")
    digits = int_part + frac_part
    if not digits or any(ch not in _DIGITS for ch in digits):
        raise ValueError("malformed decimal string: " + repr(s))
    if len(frac_part) > MAX_FRAC_DIGITS:
        raise ValueError(f"more than {MAX_FRAC_DIGITS} fractional digits: " + repr(s))
    if len(int_part) > MAX_INT_DIGITS:
        raise ValueError(f"more than {MAX_INT_DIGITS} integer digits: " + repr(s))
    value = Fraction(int(digits), 10 ** len(frac_part))
    return -value if negative else value


def parse_offsets(offsets: Sequence[str], n: int = N) -> list[Fraction]:
    """Validate a submission's ``offsets`` list and return exact rationals."""
    if isinstance(offsets, (str, bytes)) or not hasattr(offsets, "__len__"):
        raise ValueError("offsets must be a list of decimal strings")
    if len(offsets) != n:
        raise ValueError(f"expected exactly {n} offsets, got {len(offsets)}")
    return [parse_offset(s) for s in offsets]


def union_area_exact(xs: Sequence[Fraction | int]) -> Fraction:
    """Exact area of the union of the n = len(xs) triangles with rational offsets."""
    fr = [Fraction(x) for x in xs]
    n = len(fr)
    if n == 0:
        raise ValueError("need at least one offset")
    x0 = fr[0]
    rel = [x - x0 for x in fr]  # translation invariance; mirrors the arena convention
    den = 1
    for r in rel:
        den = lcm(den, r.denominator)
    big_l = lcm(*range(1, n + 1))
    unit = den * big_l  # heights are key / unit

    # Endpoint lines scaled by n * den: value = c + den * k * y with integer slope k.
    c = [n * (r.numerator * (den // r.denominator)) for r in rel]
    lines = [(j + 1, c[j]) for j in range(n)]  # left edges: slope j (1-based)
    lines += [(j, c[j] + den) for j in range(n)]  # right edges: slope j - 1

    keys: set[int] = set()
    m = len(lines)
    for a in range(m):
        ka, ca = lines[a]
        for b in range(a + 1, m):
            kb, cb = lines[b]
            dk = ka - kb
            if dk == 0:
                continue  # parallel lines never cross
            p = cb - ca
            if dk < 0:
                dk = -dk
                p = -p
            # crossing height y = p / (den * dk); keep only 0 < y < 1
            if 0 < p < den * dk:
                keys.add(p * (big_l // dk))
    heights = sorted(keys)
    heights.append(unit)  # y = 1

    base = [cj * big_l for cj in c]  # left endpoints at y = 0, units 1 / (n * unit)
    slopes = range(1, n + 1)
    order = list(range(n))  # kept between calls: nearly sorted -> cheap re-sort

    def union_length(key: int) -> int:
        left = [b0 + k * key for b0, k in zip(base, slopes)]
        width = unit - key
        order.sort(key=left.__getitem__)
        total = width
        prev = left[order[0]]
        for idx in order[1:]:
            cur = left[idx]
            gap = cur - prev
            total += gap if gap < width else width
            prev = cur
        return total

    acc = 0
    prev_key = 0
    prev_u = union_length(0)
    for key in heights:
        u = union_length(key)
        acc += (prev_u + u) * (key - prev_key)
        prev_key = key
        prev_u = u
    area = Fraction(acc, 2 * n * unit * unit)
    if area <= 0 or area > Fraction(1, 2):
        raise ValueError("internal area check failed: " + str(float(area)))
    return area


def exact_area(offsets: Sequence[str]) -> Fraction:
    """Exact union area for a submission (exactly 128 decimal strings)."""
    return union_area_exact(parse_offsets(offsets, N))


def exact_area_float(offsets: Sequence[str]) -> float:
    """The arena score: the exact area converted to float as the very last step."""
    return float(exact_area(offsets))
