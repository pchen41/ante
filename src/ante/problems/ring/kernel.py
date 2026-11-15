"""Search kernel for ring loading: sequential linear programming on witnesses.

For a fixed adversary choice (mask), its worst cut W_mask(x) = max_k |signed cut k| is a maximum of 2m
linear functions of x = (u, v), so the score A(x) = min_mask W_mask(x) is a minimum of convex functions.
Fixing one linear piece (a "witness": a cut k and a sign) per mask gives a linear lower bound on each
W_mask, and maximising the smallest of those lower bounds is a linear program. Its optimum is a valid
instance whose true score is at least the LP value on every mask included. We alternate: pick witnesses
at the current point, solve the LP over the masks that are close to binding, add any mask the new point
violates, repeat until nothing improves. Every step is checked against all 2^m masks.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import linprog


@njit(cache=True)
def evaluate(u, v, W, wk, ws):
    """For every mask: W[mask] = worst cut value, with its witness cut wk[mask] and sign ws[mask]. Returns min W."""
    m = u.shape[0]
    best = 1e300
    for mask in range(1 << m):
        total = 0.0
        for i in range(m):
            total += -u[i] if (mask >> i) & 1 else v[i]
        prefix = 0.0
        worst = -1.0
        k = 0
        s = 1
        for i in range(m):
            prefix += -u[i] if (mask >> i) & 1 else v[i]
            d = 2.0 * prefix - total
            if d > worst:
                worst, k, s = d, i, 1
            if -d > worst:
                worst, k, s = -d, i, -1
        W[mask] = worst
        wk[mask] = k
        ws[mask] = s
        if worst < best:
            best = worst
    return best


def _rows(masks, wk, ws, m):
    """LP rows `t - L_mask(x) <= 0` for x = (u_1..u_m, v_1..v_m, t)."""
    n = masks.shape[0]
    bits = (masks[:, None] >> np.arange(m)[None, :]) & 1  # 1 = adversary takes -u_i
    side = np.where(np.arange(m)[None, :] <= wk[masks][:, None], 1.0, -1.0) * ws[masks][:, None]
    A = np.zeros((n, 2 * m + 1))
    A[:, :m] = np.where(bits == 1, side, 0.0)  # -(coefficient of u_i in L) = +side
    A[:, m : 2 * m] = np.where(bits == 0, -side, 0.0)  # -(coefficient of v_i in L) = -side
    A[:, 2 * m] = 1.0
    return A


def ascend(u0, v0, max_rounds: int = 60, active_cap: int = 6000, slack: float = 0.08):
    """Witness-LP ascent from (u0, v0). Returns (u, v, score, LP solves)."""
    m = u0.shape[0]
    u, v = u0.copy(), v0.copy()
    n_masks = 1 << m
    W = np.empty(n_masks)
    wk = np.empty(n_masks, dtype=np.int64)
    ws = np.empty(n_masks, dtype=np.int64)
    cur = evaluate(u, v, W, wk, ws)
    order = np.argsort(W)[:active_cap]
    active = set(order[W[order] <= cur + slack].tolist())
    cons = np.zeros((m, 2 * m + 1))
    for i in range(m):
        cons[i, i] = cons[i, m + i] = 1.0
    bounds = [(0.0, 1.0)] * (2 * m) + [(None, None)]
    c = np.zeros(2 * m + 1)
    c[-1] = -1.0
    solves = 0
    for _ in range(max_rounds):
        masks = np.fromiter(active, dtype=np.int64)
        A = np.vstack([_rows(masks, wk, ws, m), cons])
        b = np.r_[np.zeros(masks.shape[0]), np.ones(m)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        solves += 1
        if res.status != 0:
            break
        un, vn = np.clip(res.x[:m], 0.0, 1.0), np.clip(res.x[m : 2 * m], 0.0, 1.0)
        W2 = np.empty(n_masks)
        k2 = np.empty(n_masks, dtype=np.int64)
        s2 = np.empty(n_masks, dtype=np.int64)
        new = evaluate(un, vn, W2, k2, s2)
        t = -res.fun
        if new >= t - 1e-10:  # no mask outside the active set was violated: the step is valid
            if new <= cur + 1e-11:
                break  # converged: the LP cannot improve on the current witnesses
            u, v, cur, W, wk, ws = un, vn, new, W2, k2, s2
            order = np.argsort(W)[:active_cap]
            active |= set(order[W[order] <= cur + slack].tolist())
        else:
            # masks we left out are violated at the LP point: add them, with witnesses taken at the current valid point
            viol = np.nonzero(W2 < t - 1e-10)[0]
            viol = viol[np.argsort(W2[viol])[:2000]]
            before = len(active)
            active |= set(viol.tolist())
            if len(active) == before:
                break
    return u, v, cur, solves
