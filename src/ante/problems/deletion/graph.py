"""Descendant tables for the two-deletion problem (integer / bit-operation kernels).

Conventions
-----------
A word is the integer `int(s, 2)` of its 16-character string `s`, so string
position i (0 = leftmost) is bit `15 - i`.  A descendant is the 14-bit integer
of the string that remains after deleting two positions.

Tables (CSR):
  dptr/didx : word -> sorted distinct descendants          (65,536 rows)
  pptr/pidx : descendant -> sorted words having it ("parents") (16,384 rows)

Two words conflict iff they share a descendant, so the conflict graph never
has to be stored explicitly: N(w) = union of parents(d) for d in desc(w).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

N_BITS = 16
REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / "scratch" / "deletion"


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------


@njit(cache=True)
def _delete_bit(x, p):
    """Remove bit p (0 = least significant) from x, closing the gap."""
    return ((x >> (p + 1)) << p) | (x & ((1 << p) - 1))


@njit(cache=True)
def _build_forward(n):
    nwords = 1 << n
    npairs = n * (n - 1) // 2
    ptr = np.zeros(nwords + 1, np.int64)
    idx = np.empty(nwords * npairs, np.int32)
    buf = np.empty(npairs, np.int64)
    pos = 0
    for x in range(nwords):
        m = 0
        for p in range(1, n):
            y = _delete_bit(x, p)
            for q in range(p):
                buf[m] = _delete_bit(y, q)
                m += 1
        srt = np.sort(buf)
        last = -1
        for k in range(m):
            val = srt[k]
            if val != last:
                last = val
                idx[pos] = val
                pos += 1
        ptr[x + 1] = pos
    return ptr, idx[:pos].copy()


@njit(cache=True)
def _build_inverse(ptr, idx, ndesc):
    nwords = ptr.shape[0] - 1
    cnt = np.zeros(ndesc + 1, np.int64)
    for k in range(idx.shape[0]):
        cnt[idx[k] + 1] += 1
    for d in range(ndesc):
        cnt[d + 1] += cnt[d]
    iptr = cnt.copy()
    fill = cnt[:ndesc].copy()
    iidx = np.empty(idx.shape[0], np.int32)
    for w in range(nwords):
        for a in range(ptr[w], ptr[w + 1]):
            d = idx[a]
            iidx[fill[d]] = w
            fill[d] += 1
    return iptr, iidx


@njit(cache=True)
def _degrees(dptr, didx, pptr, pidx):
    """Exact conflict-graph degree of every word (about 3e8 simple operations for n=16)."""
    n = dptr.shape[0] - 1
    deg = np.zeros(n, np.int64)
    stamp = np.full(n, -1, np.int64)
    for w in range(n):
        stamp[w] = w
        c = 0
        for a in range(dptr[w], dptr[w + 1]):
            d = didx[a]
            for b in range(pptr[d], pptr[d + 1]):
                u = pidx[b]
                if stamp[u] != w:
                    stamp[u] = w
                    c += 1
        deg[w] = c
    return deg


# ----------------------------------------------------------------------------
# Python API
# ----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Tables:
    n: int
    dptr: np.ndarray  # int64, 2**n + 1
    didx: np.ndarray  # int32
    pptr: np.ndarray  # int64, 2**(n-2) + 1
    pidx: np.ndarray  # int32

    @property
    def n_words(self) -> int:
        return 1 << self.n

    @property
    def n_desc(self) -> int:
        return 1 << (self.n - 2)

    def desc(self, w: int) -> np.ndarray:
        return self.didx[self.dptr[w] : self.dptr[w + 1]]

    def parents(self, d: int) -> np.ndarray:
        return self.pidx[self.pptr[d] : self.pptr[d + 1]]

    def conflict(self, a: int, b: int) -> bool:
        return bool(np.intersect1d(self.desc(a), self.desc(b)).size)


def build_tables(n: int = N_BITS) -> Tables:
    dptr, didx = _build_forward(n)
    pptr, pidx = _build_inverse(dptr, didx, 1 << (n - 2))
    return Tables(n=n, dptr=dptr, didx=didx, pptr=pptr, pidx=pidx)


def _cache_path(n: int) -> Path:
    return CACHE_DIR / f"tables_n{n}.npz"


def load_tables(n: int = N_BITS, cache: bool = True) -> Tables:
    """Build the tables, or load them from scratch/deletion/tables_n{n}.npz."""
    path = _cache_path(n)
    if cache and path.exists():
        try:
            with np.load(path) as z:
                t = Tables(
                    n=int(z["n"]),
                    dptr=np.ascontiguousarray(z["dptr"], dtype=np.int64),
                    didx=np.ascontiguousarray(z["didx"], dtype=np.int32),
                    pptr=np.ascontiguousarray(z["pptr"], dtype=np.int64),
                    pidx=np.ascontiguousarray(z["pidx"], dtype=np.int32),
                )
            if (
                t.n == n
                and t.dptr.shape[0] == (1 << n) + 1
                and t.pptr.shape[0] == (1 << (n - 2)) + 1
                and t.didx.shape[0] == t.pidx.shape[0] == int(t.dptr[-1])
            ):
                return t
        except Exception:  # corrupt or stale cache: rebuild below
            pass
    t = build_tables(n)
    if cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_DIR / f"tables_n{n}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp, n=np.int64(n), dptr=t.dptr, didx=t.didx, pptr=t.pptr, pidx=t.pidx)
            os.replace(tmp, path)
        except OSError:
            pass
    return t


def degrees(t: Tables) -> np.ndarray:
    return _degrees(t.dptr, t.didx, t.pptr, t.pidx)


# -- word helpers --------------------------------------------------------------


def word_to_int(s: str) -> int:
    return int(s, 2)


def int_to_word(x: int, n: int = N_BITS) -> str:
    return format(int(x), f"0{n}b")


def words_to_ints(words) -> np.ndarray:
    return np.array([int(s, 2) for s in words], dtype=np.int32)


def ints_to_words(ints, n: int = N_BITS) -> list[str]:
    return [format(int(x), f"0{n}b") for x in ints]


def weights(n: int = N_BITS) -> np.ndarray:
    """Hamming weight of every n-bit word."""
    x = np.arange(1 << n, dtype=np.int64)
    w = np.zeros(1 << n, dtype=np.int32)
    for b in range(n):
        w += ((x >> b) & 1).astype(np.int32)
    return w


def complement(x, n: int = N_BITS):
    return np.asarray(x, dtype=np.int64) ^ ((1 << n) - 1)


def reverse(x, n: int = N_BITS):
    x = np.asarray(x, dtype=np.int64)
    r = np.zeros_like(x)
    for b in range(n):
        r |= ((x >> b) & 1) << (n - 1 - b)
    return r


def symmetry_images(ints, n: int = N_BITS) -> list[np.ndarray]:
    """Images of a code under the order-4 group {id, complement, reversal, both}."""
    x = np.asarray(ints, dtype=np.int64)
    return [x, complement(x, n), reverse(x, n), complement(reverse(x, n), n)]


# -- statistics ----------------------------------------------------------------


def stats(t: Tables, with_degrees: bool = True) -> dict:
    nd = np.diff(t.dptr)
    npar = np.diff(t.pptr)
    out = {
        "n": t.n,
        "n_words": int(t.n_words),
        "n_desc": int(t.n_desc),
        "incidences": int(t.didx.shape[0]),
        "desc_per_word": {
            "min": int(nd.min()),
            "max": int(nd.max()),
            "mean": float(nd.mean()),
            "median": float(np.median(nd)),
            "quantiles_5_25_75_95": [float(q) for q in np.percentile(nd, [5, 25, 75, 95])],
        },
        "words_per_desc": {
            "min": int(npar.min()),
            "max": int(npar.max()),
            "mean": float(npar.mean()),
        },
    }
    if with_degrees:
        deg = degrees(t)
        out["conflict_degree"] = {
            "min": int(deg.min()),
            "max": int(deg.max()),
            "mean": float(deg.mean()),
            "median": float(np.median(deg)),
        }
        out["conflict_edges"] = int(deg.sum() // 2)
    return out


def main() -> None:
    t = load_tables()
    print(json.dumps(stats(t), indent=2))


if __name__ == "__main__":
    main()
