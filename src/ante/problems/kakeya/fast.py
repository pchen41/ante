"""Float64 kernels for the finite Kakeya needle problem (event-based, no y-sampling).

Formulation ("edge visibility").  At height y the union of the n sections is a set of
disjoint intervals; its length is

    U(y) = sum_j  R_j(y) [R_j exposed at y]  -  L_j(y) [L_j exposed at y]

where L_j(y) = x_j + (j/n) y and R_j(y) = x_j + 1/n + ((j-1)/n) y are the endpoint
lines and a line is *exposed* at y when its point is not inside any other section.
For a fixed line l and another triangle i the set {y : L_i(y) <= l(y) <= R_i(y)} is
one interval (two linear inequalities in y), so the exposed set of l is [0, 1] minus
127 intervals.  It is maintained as a short list of gaps; no sorting is needed.

With E_l the exposed set of line l:

    area        = sum_l  sign_l * integral_{E_l} l(y) dy          (sign: R = +1, L = -1)
    d area/dx_j = |E_{R_j}| - |E_{L_j}|                            (shape derivative)
    Hessian     = derivatives of the gap endpoints (each is a crossing height of two
                  endpoint lines, y* = (c_m - c_l) / (s_l - s_m)).

The area is C^1 and piecewise quadratic away from the hyperplanes where two parallel
edges (L_j and R_{j+1}) coincide; the Hessian returned is the one of the current cell.

Cost is O(n^2 * gaps) per evaluation, independent of the number of crossings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

from .exact import parse_offset


@njit(cache=True, nogil=True)
def _kernel(x, grad, hess, want_hess):
    n = x.shape[0]
    fn = float(n)
    half = 0.5 * fn
    mean = 0.0
    for j in range(n):
        mean += x[j]
    mean /= fn
    # u = n * (x - mean): in these units section j is [u_j + j y, u_j + 1 + (j - 1) y]
    u = np.empty(n)
    for j in range(n):
        u[j] = (x[j] - mean) * fn

    cap = n + 2
    g0 = np.empty(cap)  # gap lower ends
    g1 = np.empty(cap)  # gap upper ends
    w0 = np.empty(cap)  # d(g0)/du_source
    w1 = np.empty(cap)  # d(g1)/du_source
    s0 = np.empty(cap, np.int64)  # triangle that created g0 (-1: the boundary y = 0)
    s1 = np.empty(cap, np.int64)  # triangle that created g1 (-1: the boundary y = 1)

    total = 0.0
    for j in range(n):
        gj = 0.0
        for side in range(2):
            if side == 0:  # left edge of triangle j: c + s y with slope index j + 1
                c = u[j]
                s = j + 1
                sgn = -1.0
            else:  # right edge
                c = u[j] + 1.0
                s = j
                sgn = 1.0

            ng = 1
            g0[0] = 0.0
            g1[0] = 1.0
            s0[0] = -1
            s1[0] = -1
            w0[0] = 0.0
            w1[0] = 0.0

            for i in range(n):
                if i == j:
                    continue
                # hidden by triangle i where L_i(y) <= c + s y <= R_i(y):
                #   [A]  ((i + 1) - s) y <= c - u_i
                #   [B]  (s - i) y       <= u_i + 1 - c
                a = 0.0
                b = 1.0
                wa = 0.0
                wb = 0.0

                al = (i + 1) - s
                be = c - u[i]
                if al > 0:
                    t = be / al
                    if t < b:
                        b = t
                        wb = -1.0 / al
                elif al < 0:
                    t = be / al
                    if t > a:
                        a = t
                        wa = -1.0 / al
                elif be < 0.0:
                    continue

                al = s - i
                be = u[i] + 1.0 - c
                if al > 0:
                    t = be / al
                    if t < b:
                        b = t
                        wb = 1.0 / al
                elif al < 0:
                    t = be / al
                    if t > a:
                        a = t
                        wa = 1.0 / al
                elif be < 0.0:
                    continue

                if not (b > a):
                    continue

                # remove [a, b] from every gap
                k = 0
                while k < ng:
                    lo = g0[k]
                    hi = g1[k]
                    if b <= lo or a >= hi:
                        k += 1
                        continue
                    if a <= lo:
                        if b >= hi:
                            ng -= 1
                            if k != ng:
                                g0[k] = g0[ng]
                                g1[k] = g1[ng]
                                s0[k] = s0[ng]
                                s1[k] = s1[ng]
                                w0[k] = w0[ng]
                                w1[k] = w1[ng]
                            continue
                        g0[k] = b
                        s0[k] = i
                        w0[k] = wb
                    else:
                        if b < hi:
                            g0[ng] = b
                            g1[ng] = hi
                            s0[ng] = i
                            w0[ng] = wb
                            s1[ng] = s1[k]
                            w1[ng] = w1[k]
                            ng += 1
                        g1[k] = a
                        s1[k] = i
                        w1[k] = wa
                    k += 1
                if ng == 0:
                    break

            e = 0.0  # exposed measure
            q = 0.0  # integral of y over the exposed set
            for k in range(ng):
                lo = g0[k]
                hi = g1[k]
                d = hi - lo
                e += d
                q += 0.5 * d * (hi + lo)
                if want_hess:
                    i1 = s1[k]
                    if i1 >= 0:
                        v = sgn * fn * w1[k]
                        hess[j, i1] += v
                        hess[j, j] -= v
                    i0 = s0[k]
                    if i0 >= 0:
                        v = sgn * fn * w0[k]
                        hess[j, i0] -= v
                        hess[j, j] += v
            # (s - half): a common shear of all lines leaves U(y) unchanged and keeps the
            # summands small.
            total += sgn * (c * e + (s - half) * q)
            gj += sgn * e
        grad[j] = gj
    return total / fn


_NO_HESS = np.zeros((1, 1))


def _as_x(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def area(x) -> float:
    """Union area for float offsets ``x`` (any length n >= 1)."""
    xa = _as_x(x)
    return float(_kernel(xa, np.empty(xa.shape[0]), _NO_HESS, False))


def area_and_grad(x) -> tuple[float, np.ndarray]:
    """Area and its gradient with respect to the offsets."""
    xa = _as_x(x)
    g = np.empty(xa.shape[0])
    f = _kernel(xa, g, _NO_HESS, False)
    return float(f), g


def area_grad_hess(x) -> tuple[float, np.ndarray, np.ndarray]:
    """Area, gradient and the Hessian of the quadratic piece containing ``x``."""
    xa = _as_x(x)
    n = xa.shape[0]
    g = np.empty(n)
    h = np.zeros((n, n))
    f = _kernel(xa, g, h, True)
    return float(f), g, h


def mirror(x) -> np.ndarray:
    """Exact symmetry of the problem: x'_{n+1-j} = -x_j - 1/n has the same area.

    (Reflect x -> -x, then shear x -> x + y; triangle j becomes triangle n + 1 - j.)
    """
    xa = _as_x(x)
    return np.ascontiguousarray((-xa - 1.0 / xa.shape[0])[::-1])


def to_array(offsets: Sequence[str]) -> np.ndarray:
    """Decimal strings -> correctly rounded float64 offsets (first offset moved to 0)."""
    fr = [parse_offset(s) for s in offsets]
    x0 = fr[0]
    return np.array([float(v - x0) for v in fr], dtype=np.float64)
