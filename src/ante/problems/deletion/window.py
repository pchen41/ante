"""Exact re-optimisation with a MILP: windows around a code, and recombination pools.

Core model (`solve_packing`): given candidate words, find a largest pairwise
compatible subset,

    maximise  sum_w x_w
    s.t.      sum_{w : d in D(w)} x_w <= 1      for every descendant d
              x binary

One inequality per descendant is a clique inequality of the conflict graph, so
the LP relaxation is much tighter than the edge formulation.  Solved with
scipy.optimize.milp (HiGHS); no extra dependency.

Window search (`solve_window`): release a set R of codewords of a valid code C;
candidates are all words compatible with the fixed part C \\ R (this includes
R).  With `require_gain=True` the constraint sum x >= |R| + 1 is added, so
"infeasible" means (up to solver correctness) that the window admits no
improvement, and any feasible point is an improved code.

Recombination (`recombine_public`): maximum code inside the union of two public
codes (all symmetry images), and inside the union of all of them.

CLI:
  uv run python -m ante.problems.deletion.window --start 226 --mode cluster \
      --size 16 --windows 100 --time-limit 20 --seed 0 --out scratch/deletion/win0.json
  uv run python -m ante.problems.deletion.window --mode recombine --time-limit 300 \
      --out scratch/deletion/recombine.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from . import exact
from .graph import Tables, ints_to_words, load_tables, symmetry_images, weights, words_to_ints
from .search import resolve_start

STATUS = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "other"}
SYM = ["identity", "complement", "reversal", "complement+reversal"]


def compatible_words(t: Tables, fixed) -> np.ndarray:
    """All words that share no descendant with any word in `fixed`."""
    used = np.zeros(t.n_desc, dtype=np.int64)
    for w in np.asarray(fixed, dtype=np.int64).tolist():
        used[t.desc(w)] = 1
    blocked = np.add.reduceat(used[t.didx], t.dptr[:-1])
    return np.flatnonzero(blocked == 0)


def neighbour_mask(t: Tables, w: int) -> np.ndarray:
    """Boolean mask over all words: conflicts with w (w itself included)."""
    mask = np.zeros(t.n_words, dtype=bool)
    for d in t.desc(w).tolist():
        mask[t.parents(d)] = True
    return mask


def cluster_window(t: Tables, code, center: int, size: int, rng=None, pool_factor: float = 1.0) -> np.ndarray:
    """`size` codewords closest to `center` (any word), where closeness of a codeword v is
    |N(v) & N(center)|, the number of words blocked by both.  With pool_factor > 1 the
    window is sampled from the closest pool_factor*size codewords."""
    code = np.asarray(code, dtype=np.int64)
    base = neighbour_mask(t, int(center))
    score = np.array([int(np.count_nonzero(base & neighbour_mask(t, int(v)))) for v in code.tolist()])
    order = np.argsort(-score, kind="stable")
    order = order[score[order] > 0]
    pool = order[: max(size, int(round(pool_factor * size)))]
    if rng is not None and pool.shape[0] > size:
        pool = rng.choice(pool, size=size, replace=False)
    return code[pool[:size]]


def band_window(t: Tables, code, lo: int, hi: int) -> np.ndarray:
    """Codewords with Hamming weight in [lo, hi]."""
    code = np.asarray(code, dtype=np.int64)
    wt = weights(t.n)[code]
    return code[(wt >= lo) & (wt <= hi)]


def solve_packing(t: Tables, cands, min_size: int | None = None, time_limit: float = 30.0, first_feasible: bool = False) -> dict:
    """Largest pairwise compatible subset of `cands` (distinct words).

    Returns {"status", "chosen" (int64 array or None), "upper_bound" (float or None),
    "constraints"}.  With `min_size`, sum x >= min_size is required, so status
    "infeasible" means no compatible subset of that size exists in `cands`."""
    cands = np.asarray(cands, dtype=np.int64)
    m = int(cands.shape[0])
    lens = np.diff(t.dptr)[cands]
    rows = np.concatenate([t.desc(int(w)) for w in cands.tolist()]).astype(np.int64)
    cols = np.repeat(np.arange(m, dtype=np.int64), lens)
    _, inv, counts = np.unique(rows, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    keep = counts[inv] >= 2  # a descendant used by a single candidate constrains nothing
    n_con = int(np.count_nonzero(counts >= 2))
    if not bool(keep.any()):
        # no conflicts at all: take everything
        if min_size is not None and m < min_size:
            return {"status": "infeasible", "chosen": None, "upper_bound": float(m), "constraints": 0}
        return {"status": "optimal", "chosen": cands, "upper_bound": float(m), "constraints": 0}
    _, row_ids = np.unique(inv[keep], return_inverse=True)
    row_ids = np.asarray(row_ids).reshape(-1)
    n_rows = int(row_ids.max()) + 1
    a = sparse.csr_matrix((np.ones(row_ids.shape[0]), (row_ids, cols[keep])), shape=(n_rows, m))
    constraints = [LinearConstraint(a, -np.inf, 1.0)]
    if min_size is not None:
        constraints.append(LinearConstraint(sparse.csr_matrix(np.ones((1, m))), float(min_size), np.inf))
    res = milp(
        c=-np.ones(m),
        constraints=constraints,
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        # with `min_size` set, any feasible point is already a gain: stop at the first one instead of proving optimality
        options={"time_limit": float(time_limit), "disp": False, **({"mip_rel_gap": 0.5} if first_feasible else {})},
    )
    chosen = cands[np.asarray(res.x) > 0.5] if res.x is not None else None
    bound = getattr(res, "mip_dual_bound", None)
    upper = float(-bound) if bound is not None and np.isfinite(bound) else None
    return {
        "status": STATUS.get(int(res.status), str(res.status)),
        "chosen": chosen,
        "upper_bound": upper,
        "constraints": n_con,
    }


def solve_window(
    t: Tables,
    code,
    release,
    time_limit: float = 30.0,
    require_gain: bool = True,
    max_candidates: int = 6000,
) -> dict:
    """Re-optimise the released codewords exactly.  Returns a report; `new_code` (sorted ints)
    is present iff a strictly larger valid code was found."""
    code = np.asarray(code, dtype=np.int64)
    rel = set(np.asarray(release, dtype=np.int64).tolist())
    fixed = np.array([w for w in code.tolist() if w not in rel], dtype=np.int64)
    n_rel = int(code.shape[0] - fixed.shape[0])
    cands = compatible_words(t, fixed)
    m = int(cands.shape[0])
    out = {"released": n_rel, "candidates": m, "status": "skipped", "gain": 0, "proven_no_gain": False}
    if m <= n_rel:
        # only the released words themselves fit (they are pairwise compatible)
        out["status"] = "trivial"
        out["proven_no_gain"] = True
        return out
    if m > max_candidates:
        return out

    sol = solve_packing(t, cands, min_size=(n_rel + 1) if require_gain else None, time_limit=time_limit, first_feasible=require_gain)
    out["status"] = sol["status"]
    out["constraints"] = sol["constraints"]
    if sol["upper_bound"] is not None:
        out["upper_bound"] = sol["upper_bound"]
    chosen = sol["chosen"]
    if chosen is None:
        out["proven_no_gain"] = bool(sol["status"] == "infeasible" and require_gain)
        return out
    out["found"] = int(chosen.shape[0])
    out["gain"] = int(chosen.shape[0]) - n_rel
    out["proven_no_gain"] = bool(sol["status"] == "optimal" and out["gain"] <= 0)
    if out["gain"] > 0:
        new_code = np.sort(np.concatenate([fixed, chosen]))
        try:
            size = exact.verify(ints_to_words(new_code, t.n))
        except ValueError:  # the solver returned something that is not a code: do not trust it
            out["status"] = "invalid_solver_output"
            out["gain"] = 0
            return out
        out["new_code"] = new_code
        out["new_size"] = int(size)
    return out


def solve_fixed(t: Tables, fixed, baseline: int, time_limit: float = 30.0, max_candidates: int = 6000) -> dict:
    """Keep the pairwise compatible words `fixed`, add every word compatible with all of them as a candidate,
    and look for a completion that makes the whole code larger than `baseline` (exact; first feasible gain wins).
    Returns the same report shape as `solve_window`."""
    fixed = np.asarray(sorted(set(np.asarray(fixed, dtype=np.int64).tolist())), dtype=np.int64)
    need = int(baseline) - int(fixed.shape[0]) + 1
    cands = compatible_words(t, fixed)
    cands = cands[~np.isin(cands, fixed)]
    m = int(cands.shape[0])
    out = {"released": need - 1, "candidates": m, "status": "skipped", "gain": 0, "proven_no_gain": False}
    if m < need:
        out["status"], out["proven_no_gain"] = "trivial", True
        return out
    if m > max_candidates:
        return out
    sol = solve_packing(t, cands, min_size=need, time_limit=time_limit, first_feasible=True)
    out["status"], out["constraints"] = sol["status"], sol["constraints"]
    chosen = sol["chosen"]
    if chosen is None:
        out["proven_no_gain"] = bool(sol["status"] == "infeasible")
        return out
    new_code = np.sort(np.concatenate([fixed, chosen]))
    try:
        size = exact.verify(ints_to_words(new_code, t.n))
    except ValueError:
        out["status"] = "invalid_solver_output"
        return out
    out["gain"] = int(size) - int(baseline)
    if out["gain"] > 0:
        out["new_code"], out["new_size"] = new_code, int(size)
    return out


def difference_fixed(code, donor) -> np.ndarray:
    """The words two codes agree on. Re-solving everything else exactly searches the region between two basins."""
    return np.intersect1d(np.asarray(code, dtype=np.int64), np.asarray(donor, dtype=np.int64))


def splice_fixed(t: Tables, code, donor, k: int) -> np.ndarray:
    """Words of `code` with weight <= k plus words of `donor` with weight >= k + 3. The two parts can never
    conflict (conflicts only join weights at most 2 apart), so the two middle layers k+1, k+2 can be re-solved exactly."""
    wt = weights(t.n)
    code = np.asarray(code, dtype=np.int64)
    donor = np.asarray(donor, dtype=np.int64)
    return np.concatenate([code[wt[code] <= k], donor[wt[donor] >= k + 3]])


def recombine_public(t: Tables, time_limit: float = 300.0, pair_time_limit: float = 30.0) -> dict:
    """Exact recombination of the public codes.

    (1) For every pair (A, g(B)) of public codes and symmetry g: the largest code inside
        A | g(B).  (The union of two codes induces a bipartite conflict graph, so these are easy.)
    (2) The pool of all symmetry images of all public codes, asking for best_public + 1 words."""
    public = exact.load_public_solutions()
    codes = []
    for item in public:
        words = item["data"]["words"]
        codes.append((exact.verify(words), np.sort(words_to_ints(words).astype(np.int64))))
    best_public = max(size for size, _ in codes)
    best_size, best_code = max(codes, key=lambda sc: sc[0])  # replaced only by a strictly larger valid code
    pairs = []
    for i, (si, a) in enumerate(codes):
        for j in range(i, len(codes)):
            sj, b = codes[j]
            images = symmetry_images(b, t.n)
            for g in range(4):
                if i == j and g == 0:
                    continue
                pool = np.union1d(a, images[g])
                sol = solve_packing(t, pool, time_limit=pair_time_limit)
                found = int(sol["chosen"].shape[0]) if sol["chosen"] is not None else None
                pairs.append(
                    {
                        "a": si,
                        "b": sj,
                        "symmetry": SYM[g],
                        "overlap": int(a.shape[0] + b.shape[0] - pool.shape[0]),
                        "pool": int(pool.shape[0]),
                        "status": sol["status"],
                        "max_code": found,
                        "upper_bound": sol["upper_bound"],
                    }
                )
                print(json.dumps(pairs[-1]), flush=True)
                if found is not None and found > best_size and exact.is_valid(ints_to_words(sol["chosen"], t.n)):
                    best_size, best_code = found, sol["chosen"]

    pool = np.unique(np.concatenate([img for _, c in codes for img in symmetry_images(c, t.n)]))
    sol = solve_packing(t, pool, min_size=best_public + 1, time_limit=time_limit)
    found = int(sol["chosen"].shape[0]) if sol["chosen"] is not None else None
    full = {
        "pool": int(pool.shape[0]),
        "asked_for": best_public + 1,
        "status": sol["status"],
        "found": found,
        "upper_bound": sol["upper_bound"],
    }
    print("all images pool:", json.dumps(full), flush=True)
    if found is not None and found > best_size and exact.is_valid(ints_to_words(sol["chosen"], t.n)):
        best_size, best_code = found, sol["chosen"]

    words = ints_to_words(np.sort(best_code), t.n)
    size = exact.verify(words)
    return {
        "words": words,
        "size": size,
        "meta": {"best_public": best_public, "pairs": pairs, "all_images_pool": full},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="226", help="226 | 225 | ... | path to JSON with a words list")
    ap.add_argument("--mode", choices=["cluster", "band", "recombine"], default="cluster")
    ap.add_argument("--size", type=int, default=16, help="cluster mode: codewords released per window")
    ap.add_argument("--pool-factor", type=float, default=1.5)
    ap.add_argument("--band-width", type=int, default=1, help="band mode: number of consecutive weights released")
    ap.add_argument("--windows", type=int, default=50)
    ap.add_argument("--time-limit", type=float, default=20.0, help="MILP time limit per window (seconds)")
    ap.add_argument("--seconds", type=float, default=540.0, help="total wall-time budget")
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t = load_tables()
    if args.mode == "recombine":
        # 55 pair pools at <= 5 s each plus one big pool: stays below ~5 min + time_limit
        result = recombine_public(t, time_limit=args.time_limit, pair_time_limit=min(5.0, args.time_limit))
        print(f"best code inside the recombination pools: {result['size']} (exact.verify); "
              f"best public {result['meta']['best_public']}")
    else:
        result = _window_search(t, args)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        print(f"wrote {path}")


def _window_search(t: Tables, args) -> dict:
    code = np.sort(resolve_start(args.start, t).astype(np.int64))
    start_size = int(code.shape[0])
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    log = []
    n = t.n
    for i in range(args.windows):
        if time.time() - t0 > args.seconds:
            break
        if args.mode == "cluster":
            center = int(rng.integers(0, t.n_words))
            release = cluster_window(t, code, center, args.size, rng=rng, pool_factor=args.pool_factor)
            label = f"cluster@{center}"
        else:
            lo = int(rng.integers(0, n + 2 - args.band_width))
            release = band_window(t, code, lo, lo + args.band_width - 1)
            label = f"band[{lo},{lo + args.band_width - 1}]"
        if release.shape[0] == 0:
            continue
        rep = solve_window(t, code, release, time_limit=args.time_limit, max_candidates=args.max_candidates)
        new_code = rep.pop("new_code", None)
        rep["window"] = label
        log.append(rep)
        print(f"[{time.time() - t0:7.1f}s] {i:4d} {label:>16} {json.dumps(rep)}", flush=True)
        if new_code is not None:
            code = new_code
            print(f"  improved to {code.shape[0]} (exact.verify)", flush=True)

    words = ints_to_words(code, n)
    size = exact.verify(words)
    summary = {
        "start": args.start,
        "start_size": start_size,
        "windows": len(log),
        "proven_no_gain": sum(1 for r in log if r["proven_no_gain"]),
        "unresolved": sum(1 for r in log if not r["proven_no_gain"] and r["gain"] <= 0),
        "improving": sum(1 for r in log if r["gain"] > 0),
        "seconds": round(time.time() - t0, 1),
        "args": vars(args),
        "log": log,
    }
    print(f"final size {size} (exact.verify); {summary['proven_no_gain']} windows proven optimal, "
          f"{summary['unresolved']} unresolved, {summary['improving']} improving")
    return {"words": words, "size": size, "meta": summary}


if __name__ == "__main__":
    main()
