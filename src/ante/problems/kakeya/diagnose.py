"""Feasibility diagnostics for ``kakeya-needle-128`` on the leaderboard snapshot.

    uv run python -m ante.problems.kakeya.diagnose --json scratch/kakeya/diagnose.json

Prints (and optionally writes as JSON):

* timings of the float kernels and of the exact scorer,
* experiment (a): for every top solution the gradient norm (translation removed), the
  spectrum of the in-cell Hessian, the result of ``polish`` and the *exact* change of
  the score after re-rounding to 25 fractional digits,
* experiment (b): pairwise distances between the top solutions after translation
  alignment (also against mirror images) and their group boundaries.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .exact import exact_area
from .fast import area, area_and_grad, area_grad_hess, mirror
from .search import TARGET, W, load_leaderboard, offsets_to_strings, polish


def kernel_timings(x: np.ndarray) -> dict:
    area_grad_hess(x)  # compile / warm up
    out = {}
    for name, fn, reps in (("area", area, 300), ("area_and_grad", area_and_grad, 300), ("area_grad_hess", area_grad_hess, 100)):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        out[name + "_ms"] = 1e3 * (time.perf_counter() - t0) / reps
    return out


def _complement_basis(n: int) -> np.ndarray:
    """Orthonormal basis (n x (n-1)) of the subspace orthogonal to translations."""
    m = np.eye(n)
    m[:, 0] = 1.0
    q, _ = np.linalg.qr(m)
    return q[:, 1:]


def neighbourhood_probe(
    x: np.ndarray,
    f0: float,
    rng: np.random.Generator,
    scales: Sequence[float] = (1e-8, 1e-6, 1e-5, 1e-4),
    samples: int = 400,
    hess_samples: int = 10,
) -> list[dict]:
    """Direct local-minimum test that does not rely on one cell's Hessian.

    The leaderboard solutions contain exactly repeated spacings (three edges through one
    point), i.e. they sit on creases between quadratic pieces, so the Hessian at the
    point itself depends on floating-point tie-breaking.  Here random unit directions
    (translation removed) are probed at several radii: ``lower_count`` counts probes with
    a smaller area, and the smallest Hessian eigenvalue is recorded in the neighbouring
    cells that the probes land in.
    """
    n = x.size
    basis = _complement_basis(n)
    out = []
    for scale in scales:
        lower = 0
        min_delta = float("inf")
        for _ in range(samples):
            d = rng.normal(size=n)
            d -= d.mean()
            d /= np.linalg.norm(d)
            delta = area(x + scale * d) - f0
            min_delta = min(min_delta, delta)
            if delta < -1e-14:
                lower += 1
        eig_mins = []
        for _ in range(hess_samples):
            d = rng.normal(size=n)
            d -= d.mean()
            d /= np.linalg.norm(d)
            _, _, h = area_grad_hess(x + scale * d)
            ev = np.linalg.eigvalsh(basis.T @ (0.5 * (h + h.T)) @ basis)
            eig_mins.append(float(ev[0]))
        out.append(
            {
                "radius": scale,
                "samples": samples,
                "lower_count": lower,
                "min_delta_area": min_delta,
                "neighbour_hess_min_eig_lowest": min(eig_mins),
                "neighbour_hess_min_eig_highest": max(eig_mins),
            }
        )
    return out


def local_minimum_report(sol: dict, do_exact: bool = True) -> dict:
    x = sol["x"]
    n = x.size
    f, g, h = area_grad_hess(x)
    g = g - g.mean()
    hs = 0.5 * (h + h.T)
    basis = _complement_basis(n)
    reduced = basis.T @ hs @ basis
    ev = np.linalg.eigvalsh(reduced)
    gr = basis.T @ g
    out = {
        "id": sol["id"],
        "agent": sol["agentName"],
        "leaderboard_score": sol["score"],
        "fast_area": f,
        "fast_minus_leaderboard": f - sol["score"],
        "grad_l2": float(np.linalg.norm(g)),
        "grad_inf": float(np.max(np.abs(g))),
        "hess_asymmetry_max": float(np.max(np.abs(h - h.T))),
        "hess_rowsum_max": float(np.max(np.abs(h.sum(axis=1)))),
        "hess_eig_min": float(ev[0]),
        "hess_eig_smallest5": [float(v) for v in ev[:5]],
        "hess_eig_max": float(ev[-1]),
        "hess_negative_eigs": int(np.sum(ev < -1e-9)),
        "hess_positive_definite": bool(ev[0] > 1e-9),
    }
    if ev[0] > 1e-9:
        step = np.linalg.solve(reduced, gr)
        out["newton_predicted_gain"] = float(0.5 * gr @ step)
        out["newton_step_inf"] = float(np.max(np.abs(basis @ step)))

    t0 = time.perf_counter()
    res = polish(x)
    out["polish_seconds"] = time.perf_counter() - t0
    out["polish_status"] = res.status
    out["polish_iters"] = res.iters
    out["polish_evals"] = res.evals
    out["polish_area"] = res.area
    out["polish_gain_fast"] = f - res.area
    out["polish_grad_inf"] = res.grad_inf
    out["polish_move_inf"] = float(np.max(np.abs(res.x - (x - x[0]))))
    out["probe"] = neighbourhood_probe(x, f, np.random.default_rng(int(sol["id"] or 0)))

    if do_exact:
        t0 = time.perf_counter()
        ex0 = exact_area(sol["offsets"])
        out["exact_seconds"] = time.perf_counter() - t0
        ex1 = exact_area(offsets_to_strings(res.x))
        out["exact_original"] = float(ex0)
        out["exact_reproduces_leaderboard"] = bool(float(ex0) == sol["score"])
        out["fast_minus_exact"] = f - float(ex0)
        out["exact_polished"] = float(ex1)
        out["exact_gain_from_polish"] = float(ex0 - ex1)
        out["polished_beats_target"] = bool(float(ex1) <= TARGET)
    return out


def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    d_mean = d - d.mean()  # L2-optimal translation
    d_med = d - np.median(d)  # robust translation: counts coordinates that really differ
    return {
        "rms": float(np.sqrt(np.mean(d_mean**2))),
        "max": float(np.max(np.abs(d_mean))),
        "max_median_aligned": float(np.max(np.abs(d_med))),
        "n_diff_gt_1e-6": int(np.sum(np.abs(d_med) > 1e-6)),
        "n_diff_gt_1e-4": int(np.sum(np.abs(d_med) > 1e-4)),
        "n_diff_gt_1e-3": int(np.sum(np.abs(d_med) > 1e-3)),
    }


def group_boundaries(x: np.ndarray, thr: float = 0.4 * W) -> list[int]:
    """1-based indices j such that x_j - x_{j-1} > thr (direction j starts a new group)."""
    d = np.diff(x)
    return [int(i) + 2 for i in np.nonzero(d > thr)[0]]


def structure_report(sols: list[dict]) -> dict:
    pairs = []
    for p in range(len(sols)):
        for q in range(p + 1, len(sols)):
            pairs.append(
                {
                    "a": sols[p]["agentName"],
                    "b": sols[q]["agentName"],
                    "direct": _diff_stats(sols[p]["x"], sols[q]["x"]),
                    "a_vs_mirror_b": _diff_stats(sols[p]["x"], mirror(sols[q]["x"])),
                }
            )
    self_mirror = [
        {"agent": s["agentName"], "vs_own_mirror": _diff_stats(s["x"], mirror(s["x"]))} for s in sols
    ]
    groups = [
        {
            "agent": s["agentName"],
            "span": float(s["x"].max() - s["x"].min()),
            "boundaries_gt_0.4W": group_boundaries(s["x"], 0.4 * W),
            "boundaries_gt_2.5W": group_boundaries(s["x"], 2.5 * W),
        }
        for s in sols
    ]
    return {"pairs": pairs, "self_mirror": self_mirror, "groups": groups}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="", help="also write the full report to this path")
    ap.add_argument("--skip-exact", action="store_true", help="skip the exact rational evaluations")
    args = ap.parse_args(argv)

    sols = load_leaderboard()
    report: dict = {"target": TARGET}

    report["timings"] = kernel_timings(sols[0]["x"])
    print("== kernel timings (ms per call) ==")
    for k, v in report["timings"].items():
        print(f"  {k:20s} {v:.4f}")

    print("\n== (a) local-minimum check of every leaderboard solution ==")
    report["local_minimum"] = []
    for s in sols:
        r = local_minimum_report(s, do_exact=not args.skip_exact)
        report["local_minimum"].append(r)
        print(f"- {r['agent']} (id {r['id']}) leaderboard score {r['leaderboard_score']!r}")
        print(f"    fast area {r['fast_area']!r}  (fast - leaderboard = {r['fast_minus_leaderboard']:.3e})")
        print(f"    grad: l2 = {r['grad_l2']:.3e}, inf = {r['grad_inf']:.3e}")
        print(
            f"    in-cell Hessian (translation removed): min eig = {r['hess_eig_min']:.6g}, "
            f"max eig = {r['hess_eig_max']:.6g}, negative = {r['hess_negative_eigs']}, "
            f"PD = {r['hess_positive_definite']}; asym = {r['hess_asymmetry_max']:.2e}, "
            f"rowsum = {r['hess_rowsum_max']:.2e}"
        )
        if "newton_predicted_gain" in r:
            print(
                f"    in-cell Newton step: predicted gain = {r['newton_predicted_gain']:.3e}, "
                f"|step|_inf = {r['newton_step_inf']:.3e}"
            )
        print(
            f"    polish: status = {r['polish_status']}, iters = {r['polish_iters']}, evals = {r['polish_evals']}, "
            f"{r['polish_seconds']:.3f}s, fast gain = {r['polish_gain_fast']:.3e}, "
            f"|dx|_inf = {r['polish_move_inf']:.3e}, grad_inf after = {r['polish_grad_inf']:.3e}"
        )
        for pr in r["probe"]:
            print(
                f"    probe radius {pr['radius']:.0e}: {pr['lower_count']}/{pr['samples']} random directions lower, "
                f"min delta = {pr['min_delta_area']:.3e}, neighbour-cell Hessian min eig in "
                f"[{pr['neighbour_hess_min_eig_lowest']:.4g}, {pr['neighbour_hess_min_eig_highest']:.4g}]"
            )
        if "exact_original" in r:
            print(
                f"    exact: original = {r['exact_original']!r} (== leaderboard: {r['exact_reproduces_leaderboard']}, "
                f"{r['exact_seconds']:.2f}s), fast - exact = {r['fast_minus_exact']:.3e}"
            )
            print(
                f"    exact: polished+rounded = {r['exact_polished']!r}, exact gain = {r['exact_gain_from_polish']:.3e}, "
                f"beats target = {r['polished_beats_target']}"
            )

    print("\n== (b) structure of the top solutions ==")
    report["structure"] = structure_report(sols)
    for p in report["structure"]["pairs"]:
        d, m = p["direct"], p["a_vs_mirror_b"]
        print(
            f"- {p['a']} vs {p['b']}: rms = {d['rms']:.3e}, max = {d['max']:.3e}, "
            f"#coords differing by >1e-6/1e-4/1e-3 = {d['n_diff_gt_1e-6']}/{d['n_diff_gt_1e-4']}/{d['n_diff_gt_1e-3']}"
            f"   | vs mirror: rms = {m['rms']:.3e}, max = {m['max']:.3e}"
        )
    for s in report["structure"]["self_mirror"]:
        d = s["vs_own_mirror"]
        print(f"- {s['agent']} vs its own mirror image: rms = {d['rms']:.3e}, max = {d['max']:.3e}")
    for g in report["structure"]["groups"]:
        print(f"- {g['agent']}: span = {g['span']:.5f}")
        print(f"    group starts (jump > 0.4/128): {g['boundaries_gt_0.4W']}")
        print(f"    group starts (jump > 2.5/128): {g['boundaries_gt_2.5W']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
