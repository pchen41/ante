"""CPU search kernel for flat polynomials: tabu search over single coefficient flips.

State: c in {+-1}^70 and V[k] = p(e^{i theta_k}) on a K-point grid. Flipping coefficient j changes V by
-2 c_j e^{i j theta_k}, so every one of the 70 neighbours is scored in O(K) without an FFT.

A coarse grid under-estimates the true sup-norm (peaks hide between grid points), so every episode
ends with a fine-grid polish, and an episode's outcome is the fine-grid sup with parabolic peak
interpolation. Anything interesting is still re-scored by `exact.sup_score` / `exact.grid_score`.

Objective modes (what the tabu walk descends on the coarse grid):
  0: max_k |V_k|^2                      (the true objective on the grid; plateau-prone)
  1: sum_k (|V_k|^2 / 71)^p             (smooth L_{2p} surrogate)
  2: sum_k max(0, |V_k|^2 - T^2)^2      (excess over a moving target level T just under the best max)
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

N = 70
NORM2 = 71.0


def make_tables(K: int, offset: float = 0.0):
    theta = 2.0 * np.pi * (np.arange(K) + offset) / K
    ang = np.outer(np.arange(N), theta)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


@njit(cache=True, fastmath=True)
def _recompute(c, wc, ws, vr, vi):
    K = vr.shape[0]
    for k in range(K):
        vr[k] = 0.0
        vi[k] = 0.0
    for j in range(N):
        s = np.float32(c[j])
        for k in range(K):
            vr[k] += s * wc[j, k]
            vi[k] += s * ws[j, k]


@njit(cache=True, fastmath=True)
def _grid_max(vr, vi):
    m = 0.0
    for k in range(vr.shape[0]):
        q = vr[k] * vr[k] + vi[k] * vi[k]
        if q > m:
            m = q
    return m


@njit(cache=True, fastmath=True)
def _peak_sup(vr, vi, q):
    """max |V|^2 with 3-point parabolic interpolation at local maxima of the periodic grid."""
    K = vr.shape[0]
    for k in range(K):
        q[k] = vr[k] * vr[k] + vi[k] * vi[k]
    m = 0.0
    for k in range(K):
        a = q[k - 1] if k > 0 else q[K - 1]
        b = q[k]
        c = q[k + 1] if k < K - 1 else q[0]
        if b >= a and b >= c:
            den = 2.0 * b - a - c
            v = b
            if den > 0.0:
                v = b + (c - a) * (c - a) / (8.0 * den)
            if v > m:
                m = v
    return m


@njit(cache=True, fastmath=True)
def fine_sup(c, wcf, wsf):
    """Sup of |p|^2 estimated on a fine grid with parabolic peak interpolation."""
    K = wcf.shape[1]
    vr = np.zeros(K, dtype=np.float32)
    vi = np.zeros(K, dtype=np.float32)
    q = np.zeros(K, dtype=np.float32)
    _recompute(c, wcf, wsf, vr, vi)
    return _peak_sup(vr, vi, q)


@njit(cache=True, fastmath=True)
def polish_fine(c, wcf, wsf, max_rounds):
    """Greedy best-improvement 1-flip descent on the fine-grid sup. Mutates c; returns sup |p|^2."""
    K = wcf.shape[1]
    vr = np.zeros(K, dtype=np.float32)
    vi = np.zeros(K, dtype=np.float32)
    tr = np.zeros(K, dtype=np.float32)
    ti = np.zeros(K, dtype=np.float32)
    q = np.zeros(K, dtype=np.float32)
    _recompute(c, wcf, wsf, vr, vi)
    cur = _peak_sup(vr, vi, q)
    for _ in range(max_rounds):
        bj = -1
        bv = cur
        for j in range(N):
            d = np.float32(-2.0 * c[j])
            for k in range(K):
                tr[k] = vr[k] + d * wcf[j, k]
                ti[k] = vi[k] + d * wsf[j, k]
            v = _peak_sup(tr, ti, q)
            if v < bv:
                bv = v
                bj = j
        if bj < 0:
            break
        c[bj] = -c[bj]
        _recompute(c, wcf, wsf, vr, vi)
        cur = _peak_sup(vr, vi, q)
    return cur


@njit(cache=True, fastmath=True)
def _flip_values(c, wc, ws, vr, vi, mode, p, t2, out_obj, out_max):
    """Objective and grid max of each single-flip neighbour."""
    K = vr.shape[0]
    for j in range(N):
        d = np.float32(-2.0 * c[j])
        m = np.float32(0.0)
        acc = 0.0
        if mode == 1:
            for k in range(K):
                a = vr[k] + d * wc[j, k]
                b = vi[k] + d * ws[j, k]
                q = a * a + b * b
                if q > m:
                    m = q
                acc += (q / NORM2) ** p
        elif mode == 2:
            for k in range(K):
                a = vr[k] + d * wc[j, k]
                b = vi[k] + d * ws[j, k]
                q = a * a + b * b
                if q > m:
                    m = q
                e = q - t2
                if e > 0.0:
                    acc += e * e
        else:
            for k in range(K):
                a = vr[k] + d * wc[j, k]
                b = vi[k] + d * ws[j, k]
                q = a * a + b * b
                if q > m:
                    m = q
            acc = m
        out_obj[j] = acc
        out_max[j] = m


@njit(cache=True, fastmath=True)
def tabu_episodes(c0, use_c0, wc, ws, iters, seed, mode, p, margin, tenure_lo, tenure_hi, restart_after, kick, ep_c, ep_m):
    """Tabu walk split into episodes; an episode ends after `restart_after` non-improving steps.

    c0/use_c0: optional fixed start for the first episode (otherwise random).
    kick: 0 = each new episode starts from a fresh random sequence; k > 0 = from the previous
          episode's best with k random flips.
    margin: mode 2 only; the target level is (1 - margin) * (best grid max^2 of the episode).
    ep_c[e], ep_m[e]: best sequence and its coarse-grid max^2 for each finished episode.
    Returns (number of finished episodes, steps used).
    """
    np.random.seed(seed)
    K = wc.shape[1]
    max_ep = ep_m.shape[0]
    c = np.empty(N, dtype=np.int8)
    if use_c0:
        c[:] = c0
    else:
        for j in range(N):
            c[j] = 1 if np.random.random() < 0.5 else -1
    vr = np.zeros(K, dtype=np.float32)
    vi = np.zeros(K, dtype=np.float32)
    _recompute(c, wc, ws, vr, vi)
    obj = np.zeros(N, dtype=np.float64)
    mx = np.zeros(N, dtype=np.float32)
    tabu_until = np.zeros(N, dtype=np.int64)
    ep_best_m = _grid_max(vr, vi)
    ep_best_c = c.copy()
    t2 = ep_best_m * (1.0 - margin)
    since_improve = 0
    n_ep = 0
    it = 0
    while it < iters and n_ep < max_ep:
        _flip_values(c, wc, ws, vr, vi, mode, p, t2, obj, mx)
        bj = -1
        bo = 1e300
        nties = 0
        for j in range(N):
            if tabu_until[j] > it and not (mx[j] < ep_best_m):
                continue
            o = obj[j]
            if o < bo:
                bo = o
                bj = j
                nties = 1
            elif o == bo:
                nties += 1
                if np.random.randint(nties) == 0:
                    bj = j
        if bj < 0:
            bj = np.random.randint(N)
        d = np.float32(-2.0 * c[bj])
        for k in range(K):
            vr[k] += d * wc[bj, k]
            vi[k] += d * ws[bj, k]
        c[bj] = -c[bj]
        tabu_until[bj] = it + 1 + tenure_lo + np.random.randint(tenure_hi - tenure_lo + 1)
        m = mx[bj]
        if m < ep_best_m:
            ep_best_m = m
            ep_best_c[:] = c
            since_improve = 0
            t2 = ep_best_m * (1.0 - margin)
        else:
            since_improve += 1
        if (it & 1023) == 1023:
            _recompute(c, wc, ws, vr, vi)  # wash out float32 drift
        it += 1
        if since_improve >= restart_after:
            ep_c[n_ep, :] = ep_best_c
            ep_m[n_ep] = ep_best_m
            n_ep += 1
            since_improve = 0
            if kick > 0:
                c[:] = ep_best_c
                for _ in range(kick):
                    j = np.random.randint(N)
                    c[j] = -c[j]
            else:
                for j in range(N):
                    c[j] = 1 if np.random.random() < 0.5 else -1
            _recompute(c, wc, ws, vr, vi)
            tabu_until[:] = 0
            ep_best_m = _grid_max(vr, vi)
            ep_best_c[:] = c
            t2 = ep_best_m * (1.0 - margin)
    return n_ep, it


@njit(cache=True, fastmath=True)
def polish_all(ep_c, n_ep, wcf, wsf, max_rounds, out_sup):
    for e in range(n_ep):
        out_sup[e] = polish_fine(ep_c[e], wcf, wsf, max_rounds)


def score_from_sq(m) -> float:
    return math.sqrt(float(m) / NORM2)
