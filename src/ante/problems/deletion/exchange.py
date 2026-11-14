"""Exhaustive small-exchange checks around a given code.

For a valid code C this answers exactly:
  * free words:      is there a word that conflicts with no codeword?
  * (1,2)-swaps:     can one codeword be replaced by two words?
  * (2,3)-swaps:     can two codewords be replaced by three words?

Argument for completeness of the (2,3) check: the three inserted words may only
be blocked by the two removed codewords {a, b}.  If none of them is blocked by
both a and b, two of them are blocked by the same single codeword (or are
free), which is a (1,2)-swap or a free word and is reported there.  So it is
enough to look at pairs {a, b} that are the blocker set of some 2-tight word,
with candidates = 1-tight words of a, 1-tight words of b, 2-tight words of {a,b}.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numba import njit

from .graph import Tables, int_to_word


@njit(cache=True)
def _tight_scan(dptr, didx, nd, words):
    """For every word: number of distinct blocking codewords (capped at 4, -1 for codewords)
    and the first three blockers."""
    nw = dptr.shape[0] - 1
    owner = np.full(nd, -1, np.int32)
    sel = np.zeros(nw, np.uint8)
    for i in range(words.shape[0]):
        v = words[i]
        sel[v] = 1
        for a in range(dptr[v], dptr[v + 1]):
            owner[didx[a]] = v
    t = np.zeros(nw, np.int32)
    blk = np.full((nw, 3), -1, np.int32)
    for w in range(nw):
        if sel[w] == 1:
            t[w] = -1
            continue
        cnt = 0
        x1 = -1
        x2 = -1
        x3 = -1
        for a in range(dptr[w], dptr[w + 1]):
            o = owner[didx[a]]
            if o >= 0 and o != x1 and o != x2 and o != x3:
                if cnt == 0:
                    x1 = o
                elif cnt == 1:
                    x2 = o
                elif cnt == 2:
                    x3 = o
                cnt += 1
                if cnt > 3:
                    break
        t[w] = cnt
        blk[w, 0] = x1
        blk[w, 1] = x2
        blk[w, 2] = x3
    return t, blk


@njit(cache=True)
def _conflict(dptr, didx, u, w):
    """True iff words u and w share a descendant (merge of two sorted CSR rows)."""
    a = dptr[u]
    ea = dptr[u + 1]
    b = dptr[w]
    eb = dptr[w + 1]
    while a < ea and b < eb:
        x = didx[a]
        y = didx[b]
        if x == y:
            return True
        if x < y:
            a += 1
        else:
            b += 1
    return False


@njit(cache=True)
def _find_pair(dptr, didx, cands):
    m = cands.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if not _conflict(dptr, didx, cands[i], cands[j]):
                return i, j
    return -1, -1


@njit(cache=True)
def _find_triple(dptr, didx, cands):
    m = cands.shape[0]
    adj = np.zeros((m, m), np.uint8)
    for i in range(m):
        for j in range(i + 1, m):
            if _conflict(dptr, didx, cands[i], cands[j]):
                adj[i, j] = 1
                adj[j, i] = 1
    for i in range(m):
        for j in range(i + 1, m):
            if adj[i, j] == 0:
                for k in range(j + 1, m):
                    if adj[i, k] == 0 and adj[j, k] == 0:
                        return i, j, k
    return -1, -1, -1


def analyse(ints, t: Tables, max_report: int = 5) -> dict:
    """Exhaustive free-word / (1,2) / (2,3) check for a valid code given as integers."""
    code = np.ascontiguousarray(np.asarray(ints, dtype=np.int32))
    tight, blk = _tight_scan(t.dptr, t.didx, t.n_desc, code)
    free = np.flatnonzero(tight == 0)
    one = np.flatnonzero(tight == 1)
    two = np.flatnonzero(tight == 2)

    by_one: dict[int, list[int]] = defaultdict(list)
    for u in one.tolist():
        by_one[int(blk[u, 0])].append(u)
    by_two: dict[tuple[int, int], list[int]] = defaultdict(list)
    for u in two.tolist():
        a, b = int(blk[u, 0]), int(blk[u, 1])
        by_two[(min(a, b), max(a, b))].append(u)

    swaps12 = []
    for v, lst in by_one.items():
        if len(lst) >= 2:
            cands = np.array(lst, dtype=np.int32)
            i, j = _find_pair(t.dptr, t.didx, cands)
            if i >= 0:
                swaps12.append({"remove": [v], "insert": [lst[i], lst[j]]})

    swaps23 = []
    max_cands = 0
    for (a, b), lst in by_two.items():
        cl = by_one.get(a, []) + by_one.get(b, []) + lst
        max_cands = max(max_cands, len(cl))
        if len(cl) >= 3:
            cands = np.array(cl, dtype=np.int32)
            i, j, k = _find_triple(t.dptr, t.didx, cands)
            if i >= 0:
                swaps23.append({"remove": [a, b], "insert": [cl[i], cl[j], cl[k]]})

    def fmt(sw):
        return {key: [int_to_word(x, t.n) for x in val] for key, val in sw.items()}

    hist = {str(k): int(np.count_nonzero(tight == k)) for k in range(0, 4)}
    hist["4+"] = int(np.count_nonzero(tight >= 4))
    return {
        "size": int(code.shape[0]),
        "free_words": int(free.shape[0]),
        "free_examples": [int_to_word(x, t.n) for x in free[:max_report].tolist()],
        "tightness_hist_unselected": hist,
        "codewords_with_1tight_neighbours": len(by_one),
        "pairs_with_2tight_words": len(by_two),
        "max_candidates_per_pair": int(max_cands),
        "improving_12_swaps": len(swaps12),
        "improving_23_swaps": len(swaps23),
        "examples_12": [fmt(s) for s in swaps12[:max_report]],
        "examples_23": [fmt(s) for s in swaps23[:max_report]],
    }
