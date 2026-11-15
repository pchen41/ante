"""Independent exact check for AlphaEvolve problem 59: points of the n x n grid with no isosceles triangle.

A set is valid when no point is equally far from two others (this also forbids a point midway between
two others). Equivalently: from every point, the squared distances to all the other points are distinct.
Everything is integer arithmetic; anyone can re-run it.
"""

from __future__ import annotations


def check(points, n: int) -> int:
    """Return the number of points if the set is valid; raise ValueError otherwise."""
    pts = [(int(x), int(y)) for x, y in points]
    if any(not (0 <= x < n and 0 <= y < n) for x, y in pts):
        raise ValueError(f"every point must lie in the grid 0..{n - 1}")
    if len(set(pts)) != len(pts):
        raise ValueError("points must be distinct")
    for bx, by in pts:
        seen = set()
        for ax, ay in pts:
            if (ax, ay) == (bx, by):
                continue
            d2 = (ax - bx) ** 2 + (ay - by) ** 2
            if d2 in seen:
                raise ValueError(f"point ({bx}, {by}) is equally far from two other points (squared distance {d2})")
            seen.add(d2)
    return len(pts)
