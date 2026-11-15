"""Fast exact scoring of integer ring-loading instances by dynamic programming.

With u_i = a_i / D and v_i = b_i / D (integers a_i, b_i >= 0, a_i + b_i <= D), the adversary's prefix
sums are integers. A choice has worst cut <= A exactly when, for its final total T (|T| <= A), every
prefix sum P_1..P_m stays in the window [ceil((T - A) / 2), floor((T + A) / 2)]. So "is there a choice
with worst cut <= A?" is a reachability question on a strip of width about A/2 + 1, answered in
O(m * A) per final total instead of trying all 2^m choices. The same recursion counts the choices.

This is what makes instances with more than 15 pairs searchable. Any score reported from it is
re-checked by brute force over all 2^m choices (`exact.score_fast`) before it is believed.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def count_within(a, b, A):
    """Number of adversary choices whose worst cut is <= A (in units of 1/D)."""
    m = a.shape[0]
    total = 0
    for T in range(-A, A + 1):
        lo = -((A - T) // 2)  # ceil((T - A) / 2)
        hi = (T + A) // 2  # floor((T + A) / 2)
        w = hi - lo + 1
        if w <= 0:
            continue
        cur = np.zeros(w, dtype=np.int64)
        for p in (b[0], -a[0]):
            if lo <= p <= hi:
                cur[p - lo] += 1
        if a[0] == 0 and b[0] == 0:
            pass  # both choices land on 0: two distinct choices, counted twice above (same as brute force)
        for i in range(1, m):
            nxt = np.zeros(w, dtype=np.int64)
            any_ = False
            for j in range(w):
                c = cur[j]
                if c == 0:
                    continue
                j1 = j + b[i]
                if j1 < w:
                    nxt[j1] += c
                    any_ = True
                j2 = j - a[i]
                if j2 >= 0:
                    nxt[j2] += c
                    any_ = True
            cur = nxt
            if not any_:
                break
        if lo <= T <= hi:
            total += cur[T - lo]
    return total


@njit(cache=True)
def score_int(a, b, hint=0):
    """(A*, count): the smallest A with a feasible choice, and how many choices attain it."""
    A = max(hint, 0)
    c = count_within(a, b, A)
    if c > 0:
        while A > 0:
            c2 = count_within(a, b, A - 1)
            if c2 == 0:
                break
            A -= 1
            c = c2
        return A, c
    while True:
        A += 1
        c = count_within(a, b, A)
        if c > 0:
            return A, c


@njit(cache=True)
def anneal(a0, b0, D, iters, seed, t0, t1, target):
    """Simulated annealing on integer instances. Objective: A* (maximise), ties broken by fewer minimising choices.

    Moves: +-1 on one a_i or b_i, transfer one unit between a_i and b_i, copy a neighbour, swap two pairs.
    Stops early when A* >= target. Returns (best a, best b, best A*, count at best, steps done).
    """
    np.random.seed(seed)
    m = a0.shape[0]
    a, b = a0.copy(), b0.copy()
    A, c = score_int(a, b, 0)
    f = A - np.log2(max(c, 1)) / (m + 1.0)
    ba, bb, bA, bc, bf = a.copy(), b.copy(), A, c, f
    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / max(iters - 1, 1))
        i = np.random.randint(m)
        oa, ob = a[i], b[i]
        j = -1
        oja = ojb = 0
        r = np.random.randint(6)
        if r == 0:
            a[i] += 1 if np.random.random() < 0.5 else -1
        elif r == 1:
            b[i] += 1 if np.random.random() < 0.5 else -1
        elif r == 2:
            d = 1 if np.random.random() < 0.5 else -1
            a[i] += d
            b[i] -= d
        elif r == 3:
            j = (i + (1 if np.random.random() < 0.5 else -1)) % m
            a[i], b[i] = a[j], b[j]
            j = -1
        elif r == 4:
            j = np.random.randint(m)
            oja, ojb = a[j], b[j]
            a[i], b[i], a[j], b[j] = oja, ojb, oa, ob
        else:
            a[i], b[i] = b[i], a[i]
        if a[i] < 0 or b[i] < 0 or a[i] + b[i] > D:
            a[i], b[i] = oa, ob
            if j >= 0:
                a[j], b[j] = oja, ojb
            continue
        A2, c2 = score_int(a, b, A)
        f2 = A2 - np.log2(max(c2, 1)) / (m + 1.0)
        if f2 >= f or np.random.random() < np.exp((f2 - f) / temp):
            A, c, f = A2, c2, f2
            if f > bf:
                ba[:], bb[:], bA, bc, bf = a, b, A, c, f
                if bA >= target:
                    return ba, bb, bA, bc, it + 1
        else:
            a[i], b[i] = oa, ob
            if j >= 0:
                a[j], b[j] = oja, ojb
    return ba, bb, bA, bc, iters
