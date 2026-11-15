"""Search kernel for isosceles-free grid sets: iterated local search with exact incremental bookkeeping.

blk[q] = number of violations that adding grid cell q to the current set S would create. For every
pair {a, b} of selected points three kinds of cell are blocked: cells on the circle around a through b,
cells on the circle around b through a, and cells on the perpendicular bisector of a and b. So
"S plus q is valid" is exactly "blk[q] == 0", and adding or removing one point updates blk in
O(|S| * n) integer operations. Moves: greedy refill of free cells (boundary first), a swap that
trades one point for a cell blocked by a single pair, and ruin-and-recreate of a random or
spatially clustered group; optionally everything acts on mirror-image orbits (the best known sets
are nearly symmetric under the two reflections of the grid).
"""

from __future__ import annotations

import numpy as np
from numba import njit


def circle_table(n: int):
    """For each squared distance d2: the integer vectors (dx, dy) with dx^2 + dy^2 = d2."""
    m = 2 * (n - 1) ** 2 + 1
    vecs = [[] for _ in range(m)]
    for dx in range(-(n - 1), n):
        for dy in range(-(n - 1), n):
            if dx or dy:
                vecs[dx * dx + dy * dy].append((dx, dy))
    ptr = np.zeros(m + 1, dtype=np.int64)
    for d in range(m):
        ptr[d + 1] = ptr[d] + len(vecs[d])
    flat = np.array([v for row in vecs for v in row], dtype=np.int64).reshape(-1, 2)
    return ptr, flat[:, 0].copy(), flat[:, 1].copy()


@njit(cache=True)
def _pair(n, blk, px, py, bx, by, s, cptr, cdx, cdy):
    """Add (s=+1) or remove (s=-1) the cells blocked by the pair {p, b}."""
    d2 = (px - bx) ** 2 + (py - by) ** 2
    for t in range(cptr[d2], cptr[d2 + 1]):
        x, y = bx + cdx[t], by + cdy[t]  # circle around b through p
        if 0 <= x < n and 0 <= y < n and not (x == px and y == py):
            blk[x * n + y] += s
        x, y = px + cdx[t], py + cdy[t]  # circle around p through b
        if 0 <= x < n and 0 <= y < n and not (x == bx and y == by):
            blk[x * n + y] += s
    A, B = bx - px, by - py  # bisector: 2A x + 2B y = C
    C = bx * bx + by * by - px * px - py * py
    if C % 2 != 0:
        return
    C //= 2
    if B == 0:
        if C % A == 0:
            x = C // A
            if 0 <= x < n:
                for y in range(n):
                    blk[x * n + y] += s
    else:
        for x in range(n):
            r = C - A * x
            if r % B == 0:
                y = r // B
                if 0 <= y < n:
                    blk[x * n + y] += s


@njit(cache=True)
def _cost(n, q, pts, k, blk, sel, cptr, cdx, cdy):
    """How many currently free cells adding q would block (with multiplicity): the least-constraining refill takes the cheapest."""
    px, py = q // n, q % n
    c = 0
    for i in range(k):
        b = pts[i]
        bx, by = b // n, b % n
        d2 = (px - bx) ** 2 + (py - by) ** 2
        for t in range(cptr[d2], cptr[d2 + 1]):
            x, y = bx + cdx[t], by + cdy[t]
            if 0 <= x < n and 0 <= y < n and blk[x * n + y] == 0 and sel[x * n + y] == 0 and x * n + y != q:
                c += 1
            x, y = px + cdx[t], py + cdy[t]
            if 0 <= x < n and 0 <= y < n and blk[x * n + y] == 0 and sel[x * n + y] == 0:
                c += 1
        A, B = bx - px, by - py
        C = bx * bx + by * by - px * px - py * py
        if C % 2 != 0:
            continue
        C //= 2
        if B == 0:
            if C % A == 0:
                x = C // A
                if 0 <= x < n:
                    for y in range(n):
                        if blk[x * n + y] == 0 and sel[x * n + y] == 0 and x * n + y != q:
                            c += 1
        else:
            for x in range(n):
                r = C - A * x
                if r % B == 0:
                    y = r // B
                    if 0 <= y < n and blk[x * n + y] == 0 and sel[x * n + y] == 0 and x * n + y != q:
                        c += 1
    return c


@njit(cache=True)
def add(n, q, sel, pts, pos, k, blk, cptr, cdx, cdy):
    px, py = q // n, q % n
    for i in range(k):
        b = pts[i]
        _pair(n, blk, px, py, b // n, b % n, 1, cptr, cdx, cdy)
    sel[q] = 1
    pts[k] = q
    pos[q] = k
    return k + 1


@njit(cache=True)
def remove(n, q, sel, pts, pos, k, blk, cptr, cdx, cdy):
    i = pos[q]
    last = pts[k - 1]
    pts[i] = last
    pos[last] = i
    pos[q] = -1
    sel[q] = 0
    k -= 1
    px, py = q // n, q % n
    for j in range(k):
        b = pts[j]
        _pair(n, blk, px, py, b // n, b % n, -1, cptr, cdx, cdy)
    return k


@njit(cache=True)
def _images(n, q, sym, out):
    """Mirror images of cell q under the chosen symmetry (0: none, 1: both reflections). Returns their number."""
    out[0] = q
    if sym == 0:
        return 1
    x, y = q // n, q % n
    c = 1
    for xx, yy in ((n - 1 - x, y), (x, n - 1 - y), (n - 1 - x, n - 1 - y)):
        z = xx * n + yy
        dup = False
        for t in range(c):
            if out[t] == z:
                dup = True
        if not dup:
            out[c] = z
            c += 1
    return c


@njit(cache=True)
def _refill(n, sel, pts, pos, k, blk, cptr, cdx, cdy, sym, greedy_k, jr, jn, free, lcv):
    """Add free cells until none is left. Among `greedy_k` random free cells the one nearest the boundary is taken
    (lcv=0), or the one that blocks the fewest other free cells (lcv=1, least-constraining)."""
    img = np.empty(4, dtype=np.int64)
    while True:
        nf = 0
        for q in range(n * n):
            if blk[q] == 0 and sel[q] == 0:
                free[nf] = q
                nf += 1
        if nf == 0:
            return k, jn
        progressed = False
        for _ in range(8):
            best, bd = -1, 1 << 30
            for _ in range(greedy_k):
                q = free[np.random.randint(nf)]
                if sel[q] == 1 or blk[q] != 0:
                    continue  # taken or blocked since the free list was built
                if lcv:
                    d = _cost(n, q, pts, k, blk, sel, cptr, cdx, cdy)
                else:
                    x, y = q // n, q % n
                    d = min(min(x, n - 1 - x), min(y, n - 1 - y))
                if d < bd:
                    best, bd = q, d
            if best < 0 or blk[best] != 0 or sel[best] == 1:
                continue
            c = _images(n, best, sym, img)
            added = 0
            ok = True
            for t in range(c):
                z = img[t]
                if sel[z] == 0 and blk[z] == 0:
                    k = add(n, z, sel, pts, pos, k, blk, cptr, cdx, cdy)
                    added += 1
                else:
                    ok = False
                    break
            if not ok:  # the orbit does not fit as a whole: undo it
                for t in range(added):
                    k = remove(n, img[t], sel, pts, pos, k, blk, cptr, cdx, cdy)
                if sym != 0 and np.random.random() < 0.15 and sel[best] == 0 and blk[best] == 0:
                    k = add(n, best, sel, pts, pos, k, blk, cptr, cdx, cdy)  # occasionally break the symmetry
                    jr[jn], jn = best + 1, jn + 1
                    progressed = True
                continue
            for t in range(added):
                jr[jn], jn = img[t] + 1, jn + 1
            progressed = True
        if not progressed:
            # nothing fits as a whole orbit: add single cells so that the set is maximal
            for i in range(nf):
                q = free[i]
                if blk[q] == 0 and sel[q] == 0:
                    k = add(n, q, sel, pts, pos, k, blk, cptr, cdx, cdy)
                    jr[jn], jn = q + 1, jn + 1
            return k, jn


@njit(cache=True)
def run(n, start, iters, seed, sym, greedy_k, p_ruin, rmin, rmax, cluster, temp, max_drop, cptr, cdx, cdy, target, lcv=0):
    """Iterated local search. `start`: cell ids of a valid set (may be empty). Returns (best cells, steps done)."""
    np.random.seed(seed)
    N = n * n
    sel = np.zeros(N, dtype=np.uint8)
    pos = -np.ones(N, dtype=np.int64)
    pts = np.zeros(N, dtype=np.int64)
    blk = np.zeros(N, dtype=np.int32)
    free = np.zeros(N, dtype=np.int64)
    jr = np.zeros(8 * N, dtype=np.int64)  # journal of this step: +(q+1) added, -(q+1) removed
    img = np.empty(4, dtype=np.int64)
    k = 0
    for q in start:
        k = add(n, q, sel, pts, pos, k, blk, cptr, cdx, cdy)
    k, _ = _refill(n, sel, pts, pos, k, blk, cptr, cdx, cdy, sym, greedy_k, jr, 0, free, lcv)
    best = pts[:k].copy()
    bk = k
    for it in range(iters):
        old = k
        jn = 0
        if k < 3:
            k, jn = _refill(n, sel, pts, pos, k, blk, cptr, cdx, cdy, sym, greedy_k, jr, jn, free, lcv)
        if (np.random.random() < p_ruin or k < 3) and k > 0:
            r = rmin + np.random.randint(rmax - rmin + 1)
            centre = pts[np.random.randint(k)]
            cx, cy = centre // n, centre % n
            for _ in range(r):
                if k == 0:
                    break
                if cluster:
                    # remove the selected point nearest the centre
                    bi, bd = 0, 1 << 60
                    for i in range(k):
                        d = (pts[i] // n - cx) ** 2 + (pts[i] % n - cy) ** 2
                        if d < bd:
                            bi, bd = i, d
                    q = pts[bi]
                else:
                    q = pts[np.random.randint(k)]
                c = _images(n, q, sym, img)
                for t in range(c):
                    if sel[img[t]] == 1:
                        k = remove(n, img[t], sel, pts, pos, k, blk, cptr, cdx, cdy)
                        jr[jn], jn = -(img[t] + 1), jn + 1
        else:
            # swap: find a cell blocked by exactly one pair, drop one point of that pair, take the cell
            q = -1
            for _ in range(60):
                z = np.random.randint(N)
                if sel[z] == 0 and blk[z] == 1:
                    q = z
                    break
            if q >= 0:
                qx, qy = q // n, q % n
                victim = -1
                for i in range(k):  # apex q: two selected points equally far from q
                    di = (pts[i] // n - qx) ** 2 + (pts[i] % n - qy) ** 2
                    for j in range(i + 1, k):
                        if (pts[j] // n - qx) ** 2 + (pts[j] % n - qy) ** 2 == di:
                            victim = pts[i] if np.random.random() < 0.5 else pts[j]
                    if victim >= 0:
                        break
                if victim < 0:
                    for i in range(k):  # apex b: q and another selected point equally far from b
                        bx, by = pts[i] // n, pts[i] % n
                        dq = (bx - qx) ** 2 + (by - qy) ** 2
                        for j in range(k):
                            if j != i and (pts[j] // n - bx) ** 2 + (pts[j] % n - by) ** 2 == dq:
                                victim = pts[i] if np.random.random() < 0.5 else pts[j]
                        if victim >= 0:
                            break
                if victim >= 0:
                    k = remove(n, victim, sel, pts, pos, k, blk, cptr, cdx, cdy)
                    jr[jn], jn = -(victim + 1), jn + 1
                    if blk[q] == 0:
                        k = add(n, q, sel, pts, pos, k, blk, cptr, cdx, cdy)
                        jr[jn], jn = q + 1, jn + 1
        k, jn = _refill(n, sel, pts, pos, k, blk, cptr, cdx, cdy, sym, greedy_k, jr, jn, free, lcv)
        accept = k >= old or (bk - k <= max_drop and np.random.random() < np.exp((k - old) / temp))
        if accept:
            if k > bk:
                bk = k
                best = pts[:k].copy()
                if bk >= target:
                    return best, it + 1
        else:
            for t in range(jn - 1, -1, -1):  # undo the step
                e = jr[t]
                if e > 0:
                    k = remove(n, e - 1, sel, pts, pos, k, blk, cptr, cdx, cdy)
                else:
                    k = add(n, -e - 1, sel, pts, pos, k, blk, cptr, cdx, cdy)
    return best, iters
