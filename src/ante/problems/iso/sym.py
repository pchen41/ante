"""Orbit-level search for isosceles-free grid sets that are symmetric under both mirror reflections.

The best known sets are (almost) invariant under the two reflections x -> ox - x and y -> oy - y:
the published 164-point set for n = 100 is exactly invariant (41 orbits of 4 points), and the
112-point set for n = 64 is 27 orbits plus 4 single points. Searching over whole orbits shrinks the
problem from n^2 cells to about n^2 / 4 generators and from about 16n/9 points to about 4n/9 orbits.

State and bookkeeping are those of `kernel.py` (blk[q] = violations cell q would create); an orbit
can be added when all its cells are unblocked and stay unblocked as its points go in one by one.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import exact
from .kernel import add, remove


def orbit_table(n: int, ox: int | None = None, oy: int | None = None, band: int = 0, mirror_x: bool = True, mirror_y: bool = True):
    """Orbits of grid cells under the reflections x -> ox - x (if mirror_x) and y -> oy - y (if mirror_y).

    Defaults: both reflections, axes through the centre (ox = oy = n - 1): orbits of 4 points. With one
    reflection the orbits are mirror pairs; with none every cell is its own orbit. Only orbits that lie
    inside the grid and contain no isosceles triangle themselves are kept. With band > 0, only orbits
    with a point within `band` cells of the edge of the symmetric region are kept (in the best known
    sets almost every point is within 8 cells of the edge).
    Returns (orb, osize): orb[g] lists the cell ids x * n + y of orbit g, padded with -1.
    """
    ox = n - 1 if ox is None else ox
    oy = n - 1 if oy is None else oy
    x0, x1 = (max(0, ox - (n - 1)), min(n - 1, ox)) if mirror_x else (0, n - 1)  # the region the symmetry maps to itself
    y0, y1 = (max(0, oy - (n - 1)), min(n - 1, oy)) if mirror_y else (0, n - 1)
    seen, rows = set(), []
    for x in range(n):
        for y in range(n):
            xs = {x, ox - x} if mirror_x else {x}
            ys = {y, oy - y} if mirror_y else {y}
            imgs = sorted((a, b) for a in xs for b in ys)
            if imgs[0] != (x, y) or any(not (0 <= a < n and 0 <= b < n) for a, b in imgs):
                continue
            if band > 0 and min(x - x0, x1 - x, y - y0, y1 - y) >= band:
                continue
            if (x, y) in seen:
                continue
            seen.update(imgs)
            try:
                exact.check(imgs, n)
            except ValueError:
                continue  # e.g. the four corners of a square
            rows.append([a * n + b for a, b in imgs])
    orb = -np.ones((len(rows), 4), dtype=np.int64)
    osize = np.zeros(len(rows), dtype=np.int64)
    for g, r in enumerate(rows):
        orb[g, : len(r)] = r
        osize[g] = len(r)
    return orb, osize


def points_to_orbits(n: int, orb, osize, points) -> np.ndarray:
    """Ids of the orbits that lie entirely inside `points` (points outside any complete orbit are dropped)."""
    cells = {int(x) * n + int(y) for x, y in points}
    return np.array([g for g in range(orb.shape[0]) if all(int(orb[g, t]) in cells for t in range(int(osize[g])))], dtype=np.int64)


@njit(cache=True)
def _try_add(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy):
    """Add orbit g if it fits as a whole; returns the new point count (unchanged if it does not fit)."""
    for t in range(osize[g]):
        q = orb[g, t]
        if sel[q] == 1 or blk[q] != 0:
            for u in range(t):  # the orbit's own points blocked a later one: undo
                k = remove(n, orb[g, u], sel, pts, pos, k, blk, cptr, cdx, cdy)
            return k
        k = add(n, q, sel, pts, pos, k, blk, cptr, cdx, cdy)
    return k


@njit(cache=True)
def _drop(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy):
    for t in range(osize[g]):
        k = remove(n, orb[g, t], sel, pts, pos, k, blk, cptr, cdx, cdy)
    return k


@njit(cache=True)
def _oadd(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko):
    """Add orbit g if it fits; returns (point count, orbit count)."""
    k2 = _try_add(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy)
    if k2 > k:
        osel[g] = 1
        olist[ko] = g
        opos[g] = ko
        ko += 1
    return k2, ko


@njit(cache=True)
def _odrop(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko):
    k = _drop(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy)
    i = opos[g]
    last = olist[ko - 1]
    olist[i] = last
    opos[last] = i
    opos[g] = -1
    osel[g] = 0
    return k, ko - 1


@njit(cache=True)
def _is_free(g, orb, osize, sel, blk):
    for t in range(osize[g]):
        q = orb[g, t]
        if sel[q] == 1 or blk[q] != 0:
            return False
    return True


@njit(cache=True)
def run_sym(n, orb, osize, prio, start, iters, seed, p_cluster, rmin, rmax, temp, max_drop, noise, p_exact, exact_nodes, cptr, cdx, cdy, target):
    """Iterated local search over orbits. `start`: orbit ids of a valid symmetric set (may be empty).

    Each step removes rmin..rmax orbits (a random group, or with probability p_cluster the group nearest
    a random selected orbit) and refills the hole:
      - greedy: add every orbit that fits, in order of prio[g] + noise * N(0,1), highest first;
      - exact (with probability p_exact): depth-first search for the largest set of orbits that fits,
        limited to `exact_nodes` search nodes, stopping at the first refill larger than what was removed.
    The orbits just removed are tried last, so a step can walk sideways to a different set of the same
    size. Worse results are kept with Metropolis probability exp(delta / temp), never more than
    max_drop points below the best. Returns (best orbit ids, best point count, steps done).
    """
    np.random.seed(seed)
    N, G = n * n, orb.shape[0]
    sel = np.zeros(N, dtype=np.uint8)
    pos = -np.ones(N, dtype=np.int64)
    pts = np.zeros(N, dtype=np.int64)
    blk = np.zeros(N, dtype=np.int32)
    osel = np.zeros(G, dtype=np.uint8)
    olist = np.zeros(G, dtype=np.int64)
    opos = -np.ones(G, dtype=np.int64)
    gx = np.zeros(G, dtype=np.int64)
    gy = np.zeros(G, dtype=np.int64)
    for g in range(G):
        gx[g], gy[g] = orb[g, 0] // n, orb[g, 0] % n
    k = 0
    ko = 0
    for g in start:
        if osel[g] == 0:
            k, ko = _oadd(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
    cand = np.zeros(G, dtype=np.int64)
    key = np.zeros(G, dtype=np.float64)
    removed = np.zeros(G, dtype=np.int64)
    added = np.zeros(G, dtype=np.int64)
    stack = np.zeros(G, dtype=np.int64)
    bstack = np.zeros(G, dtype=np.int64)
    was_removed = np.zeros(G, dtype=np.uint8)
    best = olist[:ko].copy()
    bk = k
    for it in range(iters):
        old_k = k
        nrem = 0
        nadd = 0
        rem_pts = 0
        if ko > 0:
            r = min(rmin + np.random.randint(rmax - rmin + 1), ko)
            clustered = np.random.random() < p_cluster
            c = olist[np.random.randint(ko)]
            for _ in range(r):
                if clustered:
                    bi, bd = 0, 1 << 60
                    for i in range(ko):
                        h = olist[i]
                        d = (gx[h] - gx[c]) ** 2 + (gy[h] - gy[c]) ** 2
                        if d < bd:
                            bi, bd = i, d
                    g = olist[bi]
                else:
                    g = olist[np.random.randint(ko)]
                rem_pts += osize[g]
                k, ko = _odrop(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
                removed[nrem] = g
                was_removed[g] = 1
                nrem += 1
        # candidates: every orbit that fits on its own; ordered by priority plus noise, the removed ones last
        nc = 0
        for g in range(G):
            if osel[g] == 0 and _is_free(g, orb, osize, sel, blk):
                cand[nc] = g
                key[nc] = -(prio[g] + noise * np.random.standard_normal()) + (1e12 if was_removed[g] == 1 else 0.0)
                nc += 1
        order = np.argsort(key[:nc])
        tmp = cand[:nc][order].copy()
        cand[:nc] = tmp
        if nc > 0 and np.random.random() < p_exact:
            # depth-first search for the best refill
            base_k = k
            d = 0
            i = 0
            nodes = 0
            best_gain = 0
            nb = 0
            while True:
                rem_free = 0
                first = -1
                for j in range(i, nc):
                    if _is_free(cand[j], orb, osize, sel, blk):
                        rem_free += osize[cand[j]]
                        if first < 0:
                            first = j
                if first >= 0 and (k - base_k) + rem_free > best_gain and nodes < exact_nodes:
                    nodes += 1
                    k2, ko = _oadd(n, cand[first], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
                    i = first + 1
                    if k2 > k:
                        k = k2
                        stack[d] = first
                        d += 1
                        if k - base_k > best_gain:
                            best_gain = k - base_k
                            nb = d
                            bstack[:d] = stack[:d]
                            if best_gain > rem_pts:
                                break  # an improvement: take it
                    continue
                if d == 0:
                    break
                d -= 1
                j = stack[d]
                k, ko = _odrop(n, cand[j], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
                i = j + 1
            while d > 0:  # unwind, then install the best refill found
                d -= 1
                k, ko = _odrop(n, cand[stack[d]], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
            for t in range(nb):
                g = cand[bstack[t]]
                k, ko = _oadd(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
                added[nadd] = g
                nadd += 1
        else:
            for i in range(nc):
                g = cand[i]
                k2, ko = _oadd(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
                if k2 > k:
                    k = k2
                    added[nadd] = g
                    nadd += 1
        for i in range(nrem):
            was_removed[removed[i]] = 0
        if k >= old_k or (bk - k <= max_drop and np.random.random() < np.exp((k - old_k) / temp)):
            if k > bk:
                bk = k
                best = olist[:ko].copy()
                if bk >= target:
                    return best, bk, it + 1
            elif k == bk:
                best = olist[:ko].copy()  # the latest set of the best size: the next call continues the sideways walk from here
        else:  # undo the step
            for i in range(nadd):
                k, ko = _odrop(n, added[i], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
            for i in range(nrem):
                k, ko = _oadd(n, removed[i], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
    return best, bk, iters


@njit(cache=True)
def _oforce(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko):
    """Add orbit g without checking (the set may become invalid; blk keeps counting violations exactly)."""
    for t in range(osize[g]):
        k = add(n, orb[g, t], sel, pts, pos, k, blk, cptr, cdx, cdy)
    osel[g] = 1
    olist[ko] = g
    opos[g] = ko
    return k, ko + 1


@njit(cache=True)
def _violations(pts, k, blk):
    """Number of isosceles triples in the current set: every such triple adds 1 to blk of each of its three points."""
    v = 0
    for i in range(k):
        v += blk[pts[i]]
    return v // 3


@njit(cache=True)
def run_tabu(n, orb, osize, prio, start, iters, seed, tenure, prio_weight, stall, kick, cptr, cdx, cdy, target):
    """Fixed-size tabu search over orbits (min-conflicts with a growing target size).

    From a valid set, force in one more orbit (the one that creates the fewest violations), then repair:
    each step swaps out an orbit that takes part in a violation and swaps in the orbit that would create
    the fewest, with a tabu list against undoing recent swaps. When no violation is left the set is a
    valid set one orbit larger, and the target grows again. If `stall` swaps pass without success, the
    search returns to its best valid set, removes `kick` random orbits and continues.
    Insertions are ranked by violations created - prio_weight * prio[g] (ties broken at random).
    Returns (best valid orbit ids, its point count, steps done).
    """
    np.random.seed(seed)
    N, G = n * n, orb.shape[0]
    sel = np.zeros(N, dtype=np.uint8)
    pos = -np.ones(N, dtype=np.int64)
    pts = np.zeros(N, dtype=np.int64)
    blk = np.zeros(N, dtype=np.int32)
    osel = np.zeros(G, dtype=np.uint8)
    olist = np.zeros(G, dtype=np.int64)
    opos = -np.ones(G, dtype=np.int64)
    tabu = np.zeros(G, dtype=np.int64)
    k = 0
    ko = 0
    for g in start:
        if osel[g] == 0:
            k, ko = _oadd(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
    best = olist[:ko].copy()
    bk = k
    V = 0
    since = 0
    for it in range(iters):
        if V == 0:
            if k >= bk:
                grew = k > bk
                bk = k
                best = olist[:ko].copy()
                if grew and bk >= target:
                    return best, bk, it
            since = 0
        elif since > stall:
            # give up on this size: back to the best valid set, minus a few random orbits
            while ko > 0:
                k, ko = _odrop(n, olist[ko - 1], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
            for g in best:
                k, ko = _oforce(n, g, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
            for _ in range(min(kick, ko)):
                k, ko = _odrop(n, olist[np.random.randint(ko)], orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
            for g in range(G):
                tabu[g] = 0
            V = 0
            since = 0
        if V > 0:
            # swap out an orbit that takes part in a violation: usually the worst one, sometimes a random one
            h, hc, nconf = -1, -1.0, 0
            pick_random = np.random.random() < 0.3
            for i in range(ko):
                g = olist[i]
                c = 0
                for t in range(osize[g]):
                    c += blk[orb[g, t]]
                if c == 0 or tabu[g] > it:
                    continue
                nconf += 1
                score = np.random.random() if pick_random else c + np.random.random()
                if score > hc:
                    h, hc = g, score
            if h < 0:  # every conflicted orbit is tabu: take any conflicted one
                for i in range(ko):
                    g = olist[i]
                    c = 0
                    for t in range(osize[g]):
                        c += blk[orb[g, t]]
                    if c > 0 and (h < 0 or np.random.random() < 0.3):
                        h = g
            if h < 0:  # no orbit takes part in a violation: the count was stale
                V = _violations(pts, k, blk)
                since += 1
                continue
            k, ko = _odrop(n, h, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
            tabu[h] = it + tenure + np.random.randint(tenure + 1)
        # swap in (or, from a valid set, grow by) the orbit that creates the fewest violations
        gbest, gc = -1, 1e18
        for g in range(G):
            if osel[g] == 1 or tabu[g] > it:
                continue
            c = 0.0
            for t in range(osize[g]):
                c += blk[orb[g, t]]
            c = c / osize[g] * 4.0 - prio_weight * prio[g] + np.random.random()
            if c < gc:
                gbest, gc = g, c
        if gbest < 0:
            for g in range(G):
                tabu[g] = 0
            V = _violations(pts, k, blk)
            continue
        k, ko = _oforce(n, gbest, orb, osize, sel, pts, pos, k, blk, cptr, cdx, cdy, osel, olist, opos, ko)
        tabu[gbest] = it + tenure // 2 + np.random.randint(tenure // 2 + 1)
        V = _violations(pts, k, blk)
        since += 1
    return best, bk, iters


def edge_priority(n: int, orb) -> np.ndarray:
    """Default priority: orbits nearest the edge of the grid first."""
    x, y = orb[:, 0] // n, orb[:, 0] % n
    return -np.minimum(np.minimum(x, n - 1 - x), np.minimum(y, n - 1 - y)).astype(np.float64)


def orbits_to_points(n: int, orb, ids) -> list[tuple[int, int]]:
    return [(int(q) // n, int(q) % n) for g in ids for q in orb[int(g)] if q >= 0]
