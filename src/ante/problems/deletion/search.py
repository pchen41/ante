"""Iterated local search (ARW style) for the two-deletion code problem.

The problem is treated as set packing: a word "uses" its descendants and each
descendant may be owned by at most one selected word.  The conflict graph is
never materialised; neighbours of a word are enumerated through the CSR tables
of `graph.py` (word -> descendants -> parent words).

State (always a valid code, i.e. an independent set):
  sel[w]    1 if w is selected
  owner[d]  selected word owning descendant d, or -1
  tight[w]  number of distinct selected words that conflict with w
            (0 for selected words; an unselected word with tight 0 is "free")

Moves (one per iteration, followed by greedy refill of free words and a
(1,2)-swap local search around everything that changed):
  plateau   force-insert a 1-tight word (a (1,1)-swap)
  force     force-insert a word with 2..tmax blockers (blockers are removed)
  ruin      remove a cluster of selected words (neighbourhood of a random word,
            or a random weight band) and recreate greedily
Candidates are chosen by aging (longest time outside the solution among
`kappa` samples), as in Andrade-Resende-Werneck.  Worse results are accepted
with the ARW probability 1/(1 + d*d_best), a Metropolis rule, or never; rejected
iterations are rolled back from a journal.

CLI:
  uv run python -m ante.problems.deletion.search --seconds 300 --seed 0 \
      --start scratch|226|225|224|223|222|path.json --out scratch/deletion/run0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from . import exact
from .graph import Tables, ints_to_words, load_tables, weights, words_to_ints

# counters (ctr array)
C_NSOL = 0
C_QN = 1
C_FN = 2
C_JN = 3
C_JOVER = 4
C_EPOCH = 5
C_DEPOCH = 6
C_ITER = 7
C_BEST = 8
C_BEST_ITER = 9
C_ACC = 10
C_REJ = 11
C_IMP2 = 12
C_RESTART = 13
C_PLATEAU = 14
C_FORCE = 15
C_RUIN = 16
C_NOCAND = 17
C_SIZE = 24

# integer parameters
IP_TMAX = 0
IP_KAPPA = 1
IP_ACCEPT = 2  # 0 strict (never accept worse), 1 ARW, 2 Metropolis
IP_MAXDROP = 3
IP_RMIN = 4
IP_RMAX = 5
IP_USE_LS = 6
IP_GREEDY_K = 7
IP_RUIN_GREEDY_K = 8
IP_STALL = 9
IP_RUIN_MODE = 10  # 0 neighbourhood of a random word, 1 weight band
IP_SIZE = 16

# float parameters
FP_PRUIN = 0
FP_PFORCE = 1
FP_TEMP = 2
FP_SIZE = 8

JOURNAL_CAP = 1 << 17

ACCEPT_MODES = {"strict": 0, "arw": 1, "metropolis": 2}
RUIN_MODES = {"neighbourhood": 0, "band": 1}

# G = (dptr, didx, pptr, pidx, wt)
# S = (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf,
#      jw, jop, left, L, B, best, ones, ctr)
# ones[v] (v selected) = number of 1-tight words whose only blocker is v; it lets the
# local search skip codewords that cannot take part in a (1,2)-swap without scanning them.


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _add(v, G, S, log):
    """Select v.  Precondition: v is unselected and free (tight[v] == 0)."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    ep = ctr[C_EPOCH] + 1
    ctr[C_EPOCH] = ep
    stamp[v] = ep
    c1 = 0
    for a in range(dptr[v], dptr[v + 1]):
        d = didx[a]
        owner[d] = v
        for b in range(pptr[d], pptr[d + 1]):
            u = pidx[b]
            if stamp[u] != ep:
                stamp[u] = ep
                t = tight[u] + 1
                tight[u] = t
                if t == 1:
                    c1 += 1  # u is now 1-tight on v
                elif t == 2:
                    # u stops being 1-tight on its previous sole blocker
                    for c in range(dptr[u], dptr[u + 1]):
                        o = owner[didx[c]]
                        if o >= 0 and o != v:
                            ones[o] -= 1
                            break
    ones[v] = c1
    sel[v] = 1
    n = ctr[C_NSOL]
    sol[n] = v
    solpos[v] = n
    ctr[C_NSOL] = n + 1
    if log:
        j = ctr[C_JN]
        if j < jw.shape[0]:
            jw[j] = v
            jop[j] = 1
            ctr[C_JN] = j + 1
        else:
            ctr[C_JOVER] = 1


@njit(cache=True)
def _remove(v, G, S, log, track):
    """Unselect v.  With track=True, newly free words go to the free buffer and
    the sole blocker of every word that becomes 1-tight goes to the LS queue."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    ep = ctr[C_EPOCH] + 1
    ctr[C_EPOCH] = ep
    stamp[v] = ep
    for a in range(dptr[v], dptr[v + 1]):
        owner[didx[a]] = -1
    sel[v] = 0
    p = solpos[v]
    n = ctr[C_NSOL] - 1
    w = sol[n]
    sol[p] = w
    solpos[w] = p
    solpos[v] = -1
    ctr[C_NSOL] = n
    left[v] = ctr[C_ITER]
    if track and inf[v] == 0:
        inf[v] = 1
        fbuf[ctr[C_FN]] = v
        ctr[C_FN] += 1
    for a in range(dptr[v], dptr[v + 1]):
        d = didx[a]
        for b in range(pptr[d], pptr[d + 1]):
            u = pidx[b]
            if stamp[u] != ep:
                stamp[u] = ep
                t = tight[u] - 1
                tight[u] = t
                if t == 0:
                    if track and inf[u] == 0:
                        inf[u] = 1
                        fbuf[ctr[C_FN]] = u
                        ctr[C_FN] += 1
                elif t == 1:
                    # u becomes 1-tight on its remaining blocker (v's ownership is already cleared)
                    for c in range(dptr[u], dptr[u + 1]):
                        o = owner[didx[c]]
                        if o >= 0:
                            ones[o] += 1
                            if track and inq[o] == 0:
                                inq[o] = 1
                                queue[ctr[C_QN]] = o
                                ctr[C_QN] += 1
                            break
    ones[v] = 0
    if log:
        j = ctr[C_JN]
        if j < jw.shape[0]:
            jw[j] = v
            jop[j] = -1
            ctr[C_JN] = j + 1
        else:
            ctr[C_JOVER] = 1


@njit(cache=True)
def _push(v, S):
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    if inq[v] == 0:
        inq[v] = 1
        queue[ctr[C_QN]] = v
        ctr[C_QN] += 1


@njit(cache=True)
def _clear_queue(S):
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    for i in range(ctr[C_QN]):
        inq[queue[i]] = 0
    ctr[C_QN] = 0


@njit(cache=True)
def _clear_free(S):
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    for i in range(ctr[C_FN]):
        inf[fbuf[i]] = 0
    ctr[C_FN] = 0


@njit(cache=True)
def _blockers(u, G, S):
    """Write the distinct selected words that own a descendant of u into B; return how many.
    (For a selected u this is u itself.)"""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    ep = ctr[C_EPOCH] + 1
    ctr[C_EPOCH] = ep
    n = 0
    for a in range(dptr[u], dptr[u + 1]):
        o = owner[didx[a]]
        if o >= 0:
            if stamp[o] != ep:
                stamp[o] = ep
                B[n] = o
                n += 1
    return n


@njit(cache=True)
def _free_degree(w, G, S):
    """Number of free words that conflict with w."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    ep = ctr[C_EPOCH] + 1
    ctr[C_EPOCH] = ep
    stamp[w] = ep
    cnt = 0
    for a in range(dptr[w], dptr[w + 1]):
        d = didx[a]
        for b in range(pptr[d], pptr[d + 1]):
            u = pidx[b]
            if stamp[u] != ep:
                stamp[u] = ep
                if tight[u] == 0 and sel[u] == 0:
                    cnt += 1
    return cnt


@njit(cache=True)
def _refill(G, S, k):
    """Insert free words from the free buffer until none is left.

    k == 1: uniformly random order.  k > 1: among k random free candidates take
    the one with the fewest free neighbours (randomised min-degree greedy).
    Inserted words are pushed on the LS queue.  Empties the buffer."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    n = ctr[C_FN]
    while n > 0:
        best_j = -1
        best_deg = -1
        tries = 0
        while tries < k and n > 0:
            j = np.random.randint(0, n)
            w = fbuf[j]
            if tight[w] != 0 or sel[w] != 0:
                # stale entry: drop it (swap with the last one)
                inf[w] = 0
                n -= 1
                fbuf[j] = fbuf[n]
                if best_j == n:
                    best_j = j
                continue
            if k == 1 or n == 1:
                best_j = j
                break
            deg = _free_degree(w, G, S)
            if best_j < 0 or deg < best_deg:
                best_deg = deg
                best_j = j
            tries += 1
        if best_j < 0:
            break
        w = fbuf[best_j]
        inf[w] = 0
        n -= 1
        fbuf[best_j] = fbuf[n]
        _add(w, G, S, True)
        if inq[w] == 0:
            inq[w] = 1
            queue[ctr[C_QN]] = w
            ctr[C_QN] += 1
    ctr[C_FN] = 0


@njit(cache=True)
def _two_improve(v, G, S):
    """(1,2)-swap: if selected v has two non-conflicting 1-tight neighbours, replace v by them."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    ep = ctr[C_EPOCH] + 1
    ctr[C_EPOCH] = ep
    ln = 0
    for a in range(dptr[v], dptr[v + 1]):
        d = didx[a]
        for b in range(pptr[d], pptr[d + 1]):
            u = pidx[b]
            if stamp[u] != ep:
                stamp[u] = ep
                if tight[u] == 1 and sel[u] == 0:
                    L[ln] = u
                    ln += 1
    if ln < 2:
        return False
    off = np.random.randint(0, ln)
    for i in range(ln):
        u1 = L[(i + off) % ln]
        de = ctr[C_DEPOCH] + 1
        ctr[C_DEPOCH] = de
        for a in range(dptr[u1], dptr[u1 + 1]):
            dmark[didx[a]] = de
        for j in range(i + 1, ln):
            u2 = L[(j + off) % ln]
            ok = True
            for a in range(dptr[u2], dptr[u2 + 1]):
                if dmark[didx[a]] == de:
                    ok = False
                    break
            if ok:
                _remove(v, G, S, True, True)
                _add(u1, G, S, True)
                _add(u2, G, S, True)
                _push(u1, S)
                _push(u2, S)
                return True
    return False


@njit(cache=True)
def _local_search(G, S):
    """Drain the queue applying (1,2)-swaps; refill free words after every swap."""
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    while ctr[C_QN] > 0:
        q = ctr[C_QN] - 1
        ctr[C_QN] = q
        v = queue[q]
        inq[v] = 0
        if sel[v] == 1 and ones[v] >= 2:
            if _two_improve(v, G, S):
                ctr[C_IMP2] += 1
                _refill(G, S, 1)


@njit(cache=True)
def _undo(G, S):
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    j = ctr[C_JN]
    while j > 0:
        j -= 1
        v = jw[j]
        if jop[j] == 1:
            _remove(v, G, S, False, False)
        else:
            _add(v, G, S, False)
    ctr[C_JN] = 0


@njit(cache=True)
def _pick(v, tmin, tmax, kappa, G, S):
    """Among unselected neighbours u of v with tmin <= tight[u] <= tmax, sample kappa
    and return the one that has been outside the solution longest (-1 if none)."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    best_u = -1
    if tmax == 1 and ones[v] == 0:
        return best_u  # v has no 1-tight neighbour: nothing to scan for
    ep = ctr[C_EPOCH] + 1
    ctr[C_EPOCH] = ep
    ln = 0
    for a in range(dptr[v], dptr[v + 1]):
        d = didx[a]
        for b in range(pptr[d], pptr[d + 1]):
            u = pidx[b]
            if stamp[u] != ep:
                stamp[u] = ep
                t = tight[u]
                if t >= tmin and t <= tmax and sel[u] == 0:
                    L[ln] = u
                    ln += 1
    if ln == 0:
        return best_u
    best_left = 0
    for i in range(kappa):
        u = L[np.random.randint(0, ln)]
        if best_u < 0 or left[u] < best_left:
            best_u = u
            best_left = left[u]
    return best_u


@njit(cache=True)
def _force(u, G, S):
    """Remove every blocker of unselected u, then select u."""
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    nb = _blockers(u, G, S)
    for i in range(nb):
        _remove(B[i], G, S, True, True)
    _add(u, G, S, True)
    _push(u, S)


@njit(cache=True)
def _ruin(G, S, rsize, mode):
    """Remove about rsize selected words that are close to each other."""
    dptr, didx, pptr, pidx, wt = G
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    removed = 0
    tries = 0
    if mode == 1:
        # random selected words from the weight band of a random selected word
        w0 = wt[sol[np.random.randint(0, ctr[C_NSOL])]]
        while removed < rsize and tries < 20 * rsize + 50 and ctr[C_NSOL] > 0:
            tries += 1
            v = sol[np.random.randint(0, ctr[C_NSOL])]
            dw = wt[v] - w0
            if dw >= -1 and dw <= 1:
                _remove(v, G, S, True, True)
                removed += 1
        return removed
    # blockers of a random word c and of random neighbours of c
    c = np.random.randint(0, sel.shape[0])
    nb = _blockers(c, G, S)
    for i in range(nb):
        _remove(B[i], G, S, True, True)
        removed += 1
    nd = dptr[c + 1] - dptr[c]
    while removed < rsize and tries < 8 * rsize + 32 and ctr[C_NSOL] > 0:
        tries += 1
        d = didx[dptr[c] + np.random.randint(0, nd)]
        npar = pptr[d + 1] - pptr[d]
        u = pidx[pptr[d] + np.random.randint(0, npar)]
        nb = _blockers(u, G, S)
        for i in range(nb):
            _remove(B[i], G, S, True, True)
            removed += 1
    return removed


@njit(cache=True)
def _load(words, n, G, S):
    """Reset the state and select words[:n] (words that conflict with earlier ones are skipped).
    Returns the number of skipped words."""
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    sel[:] = 0
    owner[:] = -1
    tight[:] = 0
    solpos[:] = -1
    ones[:] = 0
    inq[:] = 0
    inf[:] = 0
    ctr[C_NSOL] = 0
    ctr[C_QN] = 0
    ctr[C_FN] = 0
    ctr[C_JN] = 0
    ctr[C_JOVER] = 0
    bad = 0
    for i in range(n):
        w = words[i]
        if sel[w] == 0 and tight[w] == 0:
            _add(w, G, S, False)
        else:
            bad += 1
    return bad


@njit(cache=True)
def _maximalize(G, S, k):
    """Insert free words until the code is maximal.  Returns the number of free words found."""
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    for w in range(sel.shape[0]):
        if sel[w] == 0 and tight[w] == 0 and inf[w] == 0:
            inf[w] = 1
            fbuf[ctr[C_FN]] = w
            ctr[C_FN] += 1
    nfree = ctr[C_FN]
    _refill(G, S, k)
    ctr[C_JN] = 0
    ctr[C_JOVER] = 0
    return nfree


@njit(cache=True)
def _polish(G, S):
    """Run the (1,2)-swap local search from every selected word.  Returns the number of swaps."""
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    before = ctr[C_IMP2]
    for i in range(ctr[C_NSOL]):
        _push(sol[i], S)
    _local_search(G, S)
    ctr[C_JN] = 0
    ctr[C_JOVER] = 0
    return ctr[C_IMP2] - before


@njit(cache=True)
def _save_best(S):
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    n = ctr[C_NSOL]
    for i in range(n):
        best[i] = sol[i]
    ctr[C_BEST] = n
    ctr[C_BEST_ITER] = ctr[C_ITER]


@njit(cache=True)
def _run(G, S, iparams, fparams, iters):
    """Run `iters` ILS iterations.  The state must be maximal and 2-opt on entry
    (empty queue and free buffer) with the best code saved."""
    (sel, owner, tight, sol, solpos, stamp, dmark, queue, inq, fbuf, inf, jw, jop, left, L, B, best, ones, ctr) = S
    tmax = iparams[IP_TMAX]
    kappa = iparams[IP_KAPPA]
    mode = iparams[IP_ACCEPT]
    maxdrop = iparams[IP_MAXDROP]
    rmin = iparams[IP_RMIN]
    rmax = iparams[IP_RMAX]
    use_ls = iparams[IP_USE_LS]
    gk = iparams[IP_GREEDY_K]
    rgk = iparams[IP_RUIN_GREEDY_K]
    stall = iparams[IP_STALL]
    ruin_mode = iparams[IP_RUIN_MODE]
    p_ruin = fparams[FP_PRUIN]
    p_force = fparams[FP_PFORCE]
    temp = fparams[FP_TEMP]

    for it in range(iters):
        ctr[C_ITER] += 1
        old = ctr[C_NSOL]
        if old == 0:
            break
        ctr[C_JN] = 0
        ctr[C_JOVER] = 0
        k = gk
        r = np.random.random()
        if r < p_ruin:
            rs = rmin + np.random.randint(0, rmax - rmin + 1)
            _ruin(G, S, rs, ruin_mode)
            k = rgk
            ctr[C_RUIN] += 1
        else:
            v = sol[np.random.randint(0, old)]
            u = -1
            if r < p_ruin + p_force:
                u = _pick(v, 2, tmax, kappa, G, S)
                if u >= 0:
                    ctr[C_FORCE] += 1
                else:
                    u = _pick(v, 1, 1, kappa, G, S)
                    if u >= 0:
                        ctr[C_PLATEAU] += 1
            else:
                # prefer a codeword that has a 1-tight neighbour (O(1) test per sample)
                tries = 0
                while ones[v] == 0 and tries < 16:
                    v = sol[np.random.randint(0, old)]
                    tries += 1
                u = _pick(v, 1, 1, kappa, G, S)
                if u >= 0:
                    ctr[C_PLATEAU] += 1
                elif p_force > 0.0:
                    u = _pick(v, 2, tmax, kappa, G, S)
                    if u >= 0:
                        ctr[C_FORCE] += 1
            if u >= 0:
                _force(u, G, S)
            else:
                ctr[C_NOCAND] += 1
        _refill(G, S, k)
        if use_ls != 0:
            _local_search(G, S)
        else:
            _clear_queue(S)

        new = ctr[C_NSOL]
        accept = True
        if new < old and ctr[C_JOVER] == 0:
            gap = ctr[C_BEST] - new
            if gap > maxdrop or mode == 0:
                accept = False
            elif mode == 1:
                accept = np.random.random() < 1.0 / (1.0 + (old - new) * gap)
            else:
                accept = np.random.random() < np.exp(-(old - new) / temp)
        if accept:
            ctr[C_ACC] += 1
            if new > ctr[C_BEST]:
                for i in range(new):
                    best[i] = sol[i]
                ctr[C_BEST] = new
                ctr[C_BEST_ITER] = ctr[C_ITER]
        else:
            _undo(G, S)
            ctr[C_REJ] += 1

        if stall > 0 and ctr[C_ITER] - ctr[C_BEST_ITER] > stall:
            ctr[C_BEST_ITER] = ctr[C_ITER]
            if ctr[C_NSOL] < ctr[C_BEST]:
                _load(best, ctr[C_BEST], G, S)
                ctr[C_RESTART] += 1
    return ctr[C_BEST]


@njit(cache=True)
def _recount(G, words, n):
    """Reference recomputation of owner/tight for a code (used for consistency checks).
    Returns (owner, tight, n_conflicting_descendants)."""
    dptr, didx, pptr, pidx, wt = G
    nw = dptr.shape[0] - 1
    nd = pptr.shape[0] - 1
    owner = np.full(nd, -1, np.int32)
    clashes = 0
    for i in range(n):
        v = words[i]
        for a in range(dptr[v], dptr[v + 1]):
            d = didx[a]
            if owner[d] >= 0:
                clashes += 1
            owner[d] = v
    tight = np.zeros(nw, np.int32)
    ones = np.zeros(nw, np.int32)
    stamp = np.full(nw, -1, np.int64)
    for w in range(nw):
        c = 0
        last = -1
        for a in range(dptr[w], dptr[w + 1]):
            o = owner[didx[a]]
            if o >= 0 and o != w and stamp[o] != w:
                stamp[o] = w
                c += 1
                last = o
        tight[w] = c
        if c == 1:
            ones[last] += 1
    return owner, tight, ones, clashes


# ----------------------------------------------------------------------------
# Python wrapper
# ----------------------------------------------------------------------------

DEFAULTS = dict(
    tmax=3,
    kappa=3,
    accept="arw",
    max_drop=2,
    ruin_min=4,
    ruin_max=16,
    use_ls=True,
    greedy_k=1,
    ruin_greedy_k=2,
    stall=50_000,
    ruin_mode="neighbourhood",
    p_ruin=0.02,
    p_force=0.25,
    temp=0.5,
)


class Searcher:
    """Holds the search state; `run(seconds)` advances it in jitted chunks."""

    def __init__(self, tables: Tables | None = None, seed: int = 0, **params):
        self.t = tables if tables is not None else load_tables()
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown search parameters: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        nw, nd = self.t.n_words, self.t.n_desc
        self.G = (self.t.dptr, self.t.didx, self.t.pptr, self.t.pidx, weights(self.t.n))
        self.ctr = np.zeros(C_SIZE, np.int64)
        self.S = (
            np.zeros(nw, np.uint8),  # sel
            np.full(nd, -1, np.int32),  # owner
            np.zeros(nw, np.int32),  # tight
            np.zeros(nw, np.int32),  # sol
            np.full(nw, -1, np.int32),  # solpos
            np.zeros(nw, np.int64),  # stamp
            np.zeros(nd, np.int64),  # dmark
            np.zeros(nw, np.int32),  # queue
            np.zeros(nw, np.uint8),  # inq
            np.zeros(nw, np.int32),  # fbuf
            np.zeros(nw, np.uint8),  # inf
            np.zeros(JOURNAL_CAP, np.int32),  # jw
            np.zeros(JOURNAL_CAP, np.int8),  # jop
            np.zeros(nw, np.int64),  # left
            np.zeros(nw, np.int32),  # L
            np.zeros(nw, np.int32),  # B
            np.zeros(nw, np.int32),  # best
            np.zeros(nw, np.int32),  # ones
            self.ctr,
        )
        self.seed = int(seed)
        _seed(self.seed)
        self._set_params()
        self.info: dict = {}

    def _set_params(self) -> None:
        p = self.params
        ip = np.zeros(IP_SIZE, np.int64)
        fp = np.zeros(FP_SIZE, np.float64)
        ip[IP_TMAX] = max(1, int(p["tmax"]))
        ip[IP_KAPPA] = max(1, int(p["kappa"]))
        ip[IP_ACCEPT] = ACCEPT_MODES[p["accept"]]
        ip[IP_MAXDROP] = int(p["max_drop"])
        ip[IP_RMIN] = max(1, int(p["ruin_min"]))
        ip[IP_RMAX] = max(int(ip[IP_RMIN]), int(p["ruin_max"]))
        ip[IP_USE_LS] = 1 if p["use_ls"] else 0
        ip[IP_GREEDY_K] = max(1, int(p["greedy_k"]))
        ip[IP_RUIN_GREEDY_K] = max(1, int(p["ruin_greedy_k"]))
        ip[IP_STALL] = int(p["stall"])
        ip[IP_RUIN_MODE] = RUIN_MODES[p["ruin_mode"]]
        fp[FP_PRUIN] = float(p["p_ruin"])
        fp[FP_PFORCE] = float(p["p_force"])
        fp[FP_TEMP] = max(1e-9, float(p["temp"]))
        self.ip, self.fp = ip, fp

    # -- initialisation ------------------------------------------------------

    def start(self, ints=None) -> dict:
        """Load a code (or the empty code), make it maximal and (1,2)-swap optimal.

        Returns counts that answer "was the start code locally improvable?"."""
        if ints is None:
            arr = np.zeros(0, np.int32)
        else:
            arr = np.ascontiguousarray(np.asarray(ints, dtype=np.int32))
        skipped = int(_load(arr, arr.shape[0], self.G, self.S))
        loaded = int(self.ctr[C_NSOL])
        free = int(_maximalize(self.G, self.S, int(self.ip[IP_GREEDY_K])))
        after_fill = int(self.ctr[C_NSOL])
        swaps = int(_polish(self.G, self.S)) if self.params["use_ls"] else 0
        _clear_queue(self.S)
        _clear_free(self.S)
        _save_best(self.S)
        self.info = {
            "loaded": loaded,
            "skipped_conflicting": skipped,
            "free_words_at_start": free if ints is not None else None,
            "size_after_fill": after_fill,
            "swaps12_at_start": swaps,
            "size_after_polish": int(self.ctr[C_NSOL]),
        }
        return self.info

    # -- running ---------------------------------------------------------------

    def run(self, seconds: float, trace_every: float = 30.0, target: int | None = None) -> dict:
        _run(self.G, self.S, self.ip, self.fp, 0)  # compile outside the clock
        t0 = time.perf_counter()
        it0 = int(self.ctr[C_ITER])
        trace = [[0.0, int(self.ctr[C_BEST])]]
        trace_every = max(float(trace_every), 0.1)
        next_trace = trace_every
        chunk = 200
        while True:
            now = time.perf_counter()
            if now - t0 >= seconds:
                break
            before = int(self.ctr[C_BEST])
            _run(self.G, self.S, self.ip, self.fp, chunk)
            end = time.perf_counter()
            dt = end - now
            if dt < 0.1:
                chunk = min(chunk * 2, 5_000_000)
            elif dt > 0.4:
                chunk = max(chunk // 2, 50)
            best = int(self.ctr[C_BEST])
            if best != before or end - t0 >= next_trace:
                trace.append([round(end - t0, 2), best])
                while end - t0 >= next_trace:
                    next_trace += trace_every
            if target is not None and best >= target:
                break
        elapsed = time.perf_counter() - t0
        iters = int(self.ctr[C_ITER]) - it0
        trace.append([round(elapsed, 2), int(self.ctr[C_BEST])])
        return {
            "seconds": round(elapsed, 3),
            "iterations": iters,
            "iters_per_sec": round(iters / elapsed, 1) if elapsed > 0 else 0.0,
            "best": int(self.ctr[C_BEST]),
            "current": int(self.ctr[C_NSOL]),
            "trace": trace,
            "counters": self.counters(),
        }

    def counters(self) -> dict:
        c = self.ctr
        return {
            "accepted": int(c[C_ACC]),
            "rejected": int(c[C_REJ]),
            "swaps12": int(c[C_IMP2]),
            "plateau_moves": int(c[C_PLATEAU]),
            "force_moves": int(c[C_FORCE]),
            "ruin_moves": int(c[C_RUIN]),
            "no_candidate": int(c[C_NOCAND]),
            "restarts": int(c[C_RESTART]),
        }

    # -- results ---------------------------------------------------------------

    def best_ints(self) -> np.ndarray:
        return np.sort(self.S[16][: int(self.ctr[C_BEST])].copy())

    def current_ints(self) -> np.ndarray:
        return np.sort(self.S[3][: int(self.ctr[C_NSOL])].copy())

    def best_words(self) -> list[str]:
        return ints_to_words(self.best_ints(), self.t.n)

    def check_state(self) -> None:
        """Assert that the incremental bookkeeping matches a recomputation from scratch."""
        sel, owner, tight, sol, solpos = self.S[0], self.S[1], self.S[2], self.S[3], self.S[4]
        n = int(self.ctr[C_NSOL])
        cur = np.ascontiguousarray(sol[:n])
        assert len(set(cur.tolist())) == n, "duplicate words in solution list"
        assert int(sel.sum()) == n, "sel does not match solution list"
        assert bool(np.all(sel[cur] == 1))
        assert bool(np.all(solpos[cur] == np.arange(n)))
        owner2, tight2, ones2, clashes = _recount(self.G, cur, n)
        assert clashes == 0, "current code is not valid"
        assert bool(np.array_equal(owner, owner2)), "owner table out of sync"
        assert bool(np.array_equal(tight, tight2)), "tightness out of sync"
        assert bool(np.array_equal(self.S[17], ones2)), "1-tight neighbour counts out of sync"
        assert not bool(np.any((tight == 0) & (sel == 0))), "code is not maximal"


def resolve_start(start: str, tables: Tables) -> np.ndarray | None:
    """'scratch' -> None; '226' etc -> public code with that score; otherwise a JSON path."""
    if start == "scratch":
        return None
    if start.isdigit():
        words = exact.load_public_words(int(start))
    else:
        with open(start, encoding="utf-8") as fh:
            obj = json.load(fh)
        words = obj["words"] if isinstance(obj, dict) else obj
    exact.verify(list(words))
    return words_to_ints(words)


def search(
    seconds: float,
    seed: int = 0,
    start: str = "scratch",
    tables: Tables | None = None,
    trace_every: float = 30.0,
    target: int | None = None,
    **params,
) -> dict:
    """Run the search and return `{"words", "size", "meta"}` (size confirmed by exact.verify)."""
    t = tables if tables is not None else load_tables()
    s = Searcher(t, seed=seed, **params)
    init = s.start(resolve_start(start, t))
    res = s.run(seconds, trace_every=trace_every, target=target)
    words = s.best_words()
    size = exact.verify(words)
    if size != res["best"]:
        raise RuntimeError(f"search reported {res['best']} but exact.verify counted {size}")
    meta = {
        "seed": seed,
        "start": start,
        "start_info": init,
        "params": s.params,
        "verified_by": "ante.problems.deletion.exact.verify",
        **res,
    }
    return {"words": words, "size": size, "meta": meta}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="scratch", help="scratch | 226 | 225 | 224 | 223 | 222 | path to JSON")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=int, default=None, help="stop as soon as this size is reached")
    ap.add_argument("--trace-every", type=float, default=30.0)
    ap.add_argument("--tmax", type=int, default=DEFAULTS["tmax"])
    ap.add_argument("--kappa", type=int, default=DEFAULTS["kappa"])
    ap.add_argument("--accept", choices=sorted(ACCEPT_MODES), default=DEFAULTS["accept"])
    ap.add_argument("--max-drop", type=int, default=DEFAULTS["max_drop"])
    ap.add_argument("--temp", type=float, default=DEFAULTS["temp"])
    ap.add_argument("--p-ruin", type=float, default=DEFAULTS["p_ruin"])
    ap.add_argument("--p-force", type=float, default=DEFAULTS["p_force"])
    ap.add_argument("--ruin-min", type=int, default=DEFAULTS["ruin_min"])
    ap.add_argument("--ruin-max", type=int, default=DEFAULTS["ruin_max"])
    ap.add_argument("--ruin-mode", choices=sorted(RUIN_MODES), default=DEFAULTS["ruin_mode"])
    ap.add_argument("--greedy-k", type=int, default=DEFAULTS["greedy_k"])
    ap.add_argument("--ruin-greedy-k", type=int, default=DEFAULTS["ruin_greedy_k"])
    ap.add_argument("--stall", type=int, default=DEFAULTS["stall"], help="restart from best after this many idle iterations (0 = never)")
    ap.add_argument("--no-ls", action="store_true", help="ablation: disable the (1,2)-swap local search")
    args = ap.parse_args(argv)

    out = search(
        seconds=args.seconds,
        seed=args.seed,
        start=args.start,
        trace_every=args.trace_every,
        target=args.target,
        tmax=args.tmax,
        kappa=args.kappa,
        accept=args.accept,
        max_drop=args.max_drop,
        temp=args.temp,
        p_ruin=args.p_ruin,
        p_force=args.p_force,
        ruin_min=args.ruin_min,
        ruin_max=args.ruin_max,
        ruin_mode=args.ruin_mode,
        greedy_k=args.greedy_k,
        ruin_greedy_k=args.ruin_greedy_k,
        stall=args.stall,
        use_ls=not args.no_ls,
    )
    m = out["meta"]
    print(
        f"start={m['start']} seed={m['seed']} size={out['size']} (exact.verify) "
        f"iters={m['iterations']} iters/s={m['iters_per_sec']} start_info={m['start_info']}"
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
