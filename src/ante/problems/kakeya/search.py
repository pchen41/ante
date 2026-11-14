"""Local polish and structural exploration for ``kakeya-needle-128``.

* ``polish(x)``: damped (Levenberg-Marquardt) Newton on the piecewise-quadratic area
  using the exact in-cell Hessian from ``fast.area_grad_hess``; steps that cross into a
  worse cell are rejected and re-damped.  Falls back to L-BFGS-B when Newton stalls.
* ``explore(...)``: iterated local search / basin hopping with structural moves
  (block shift, scanned block re-insertion, group resize, block reflection, pattern
  transfer between blocks, crossover with other good solutions and mirror images,
  windowed kicks), each followed by ``polish``.
* CLI::

      uv run python -m ante.problems.kakeya.search --seconds 300 --seed 0 \
          --start best --out scratch/kakeya/run0.json

  The output offsets are rounded to <= 25 fractional digits and the reported score is
  recomputed with ``exact.exact_area`` on those final strings.
"""

from __future__ import annotations

import os

# Keep BLAS single-threaded: the matrices are 128 x 128 and several searches usually run
# side by side on a shared machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import LinAlgError, cho_factor, cho_solve  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from .exact import N, exact_area  # noqa: E402
from .fast import area, area_and_grad, area_grad_hess, mirror, to_array  # noqa: E402

W = 1.0 / N  # base width of every triangle
ARENA_SLUG = "kakeya-needle-128"
# kgrey's #1 score minus the arena's 1e-9 minimum improvement.
TARGET = 0.10675138895168142

# float64 noise of one area evaluation (sum of 256 terms of size O(10), divided by n)
_F_NOISE = 5e-15


# --------------------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------------------


def data_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data" / "arena" / ARENA_SLUG
        if cand.is_dir():
            return cand
    return Path("data") / "arena" / ARENA_SLUG


def load_leaderboard(path: str | Path | None = None) -> list[dict]:
    """Top solutions from the read-only arena snapshot, best first."""
    p = Path(path) if path is not None else data_dir() / "best_solutions.json"
    raw = json.loads(p.read_text())
    out = []
    for item in raw:
        offsets = item["data"]["offsets"]
        out.append(
            {
                "id": item.get("id"),
                "agentName": item.get("agentName"),
                "score": float(item["score"]),
                "offsets": offsets,
                "x": to_array(offsets),
            }
        )
    out.sort(key=lambda s: s["score"])
    return out


def offsets_to_strings(x) -> list[str]:
    """Float offsets -> decimal strings with <= 25 fractional digits, first offset 0."""
    xa = np.asarray(x, dtype=np.float64)
    x0 = float(xa[0])
    out = []
    for v in xa:
        s = "%.25f" % (float(v) - x0)
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if s in ("-0", "-", ""):
            s = "0"
        out.append(s)
    return out


# --------------------------------------------------------------------------------------
# polish
# --------------------------------------------------------------------------------------


@dataclass
class Polished:
    x: np.ndarray  # normalised so that x[0] == 0
    area: float  # float64 area (fast kernel)
    grad_inf: float  # max |gradient| after removing the translation direction
    iters: int
    evals: int
    status: str  # "converged" | "stalled" | "max_iter"


def _newton(x: np.ndarray, max_iter: int, gtol: float):
    n = x.size
    eye = np.eye(n)
    ones = np.full((n, n), 1.0 / n)
    f, g, h = area_grad_hess(x)
    g = g - g.mean()
    evals = 1
    lam = 0.0
    status = "max_iter"
    it = 0
    while it < max_iter:
        gi = float(np.max(np.abs(g)))
        if gi <= gtol:
            status = "converged"
            break
        hs = 0.5 * (h + h.T)
        scale = max(float(np.trace(hs)) / n, 1.0)
        stepped = False
        while lam <= 1e9:
            # ones-term: curvature along the (flat) translation direction
            m = hs + (lam * scale) * eye + scale * ones
            try:
                cf = cho_factor(m, lower=True, check_finite=False)
            except LinAlgError:
                lam = max(10.0 * lam, 1e-5)
                continue
            p = -cho_solve(cf, g, check_finite=False)
            if not np.all(np.isfinite(p)):
                lam = max(10.0 * lam, 1e-5)
                continue
            p -= p.mean()
            xn = x + p
            fn, gn, hn = area_grad_hess(xn)
            evals += 1
            gn = gn - gn.mean()
            gin = float(np.max(np.abs(gn)))
            df = f - fn
            # accept a real decrease, or a step inside the noise floor that shrinks the
            # gradient (Newton converging to the stationary point of the current cell)
            if df > _F_NOISE or (df >= -_F_NOISE and gin < gi):
                x, f, g, h = xn, fn, gn, hn
                lam = lam / 3.0 if lam > 1e-6 else 0.0
                stepped = True
                break
            lam = max(10.0 * lam, 1e-5)
        it += 1
        if not stepped:
            status = "stalled"
            break
    return x, f, float(np.max(np.abs(g))), evals, it, status


def _lbfgs(x: np.ndarray, maxiter: int):
    def fg(z):
        f, g = area_and_grad(z)
        return f, g - g.mean()

    res = minimize(
        fg,
        x,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-16, "gtol": 1e-13},
    )
    return np.asarray(res.x, dtype=np.float64), int(res.nfev)


def polish(x, max_iter: int = 200, gtol: float = 1e-12, lbfgs_fallback: bool = True) -> Polished:
    """Converge to a local minimum of the area starting from ``x``."""
    x0 = np.array(x, dtype=np.float64)
    x0 -= x0[0]
    xr, f, gi, evals, iters, status = _newton(x0, max_iter, gtol)
    if status != "converged" and lbfgs_fallback and gi > 1e-9:
        xl, ev = _lbfgs(xr, 400)
        evals += ev
        fl = area(xl)
        evals += 1
        if fl <= f:
            xr2, f2, gi2, ev2, it2, st2 = _newton(xl - xl[0], max_iter, gtol)
            evals += ev2
            iters += it2
            if f2 <= f:
                xr, f, gi, status = xr2, f2, gi2, st2
    xr = xr - xr[0]
    return Polished(x=xr, area=float(f), grad_inf=float(gi), iters=iters, evals=evals, status=status)


# --------------------------------------------------------------------------------------
# structural moves:  move(x, rng, ctx) -> new x  (or None when not applicable)
# --------------------------------------------------------------------------------------


@dataclass
class Context:
    donors: list[np.ndarray] = field(default_factory=list)  # fixed: other good solutions
    elites: list[tuple[float, np.ndarray]] = field(default_factory=list)  # found in this run
    max_elites: int = 12

    def add_elite(self, f: float, x: np.ndarray) -> None:
        for fe, _ in self.elites:
            if abs(fe - f) < 1e-10:
                return
        self.elites.append((f, x.copy()))
        self.elites.sort(key=lambda t: t[0])
        del self.elites[self.max_elites :]

    def donor_pool(self, x: np.ndarray) -> list[np.ndarray]:
        pool = list(self.donors) + [e for _, e in self.elites]
        pool += [mirror(d) for d in pool]
        pool.append(mirror(x))
        return pool


def _rand_block(rng: np.random.Generator, n: int, lo: int, hi: int) -> tuple[int, int]:
    """Random contiguous block (start, length), length log-uniform in [lo, hi]."""
    hi = max(lo, min(hi, n))
    length = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    length = max(lo, min(hi, length))
    start = int(rng.integers(0, n - length + 1))
    return start, length


def mv_block_shift(x, rng, ctx):
    """Translate a contiguous block (or a whole suffix) of directions."""
    n = x.size
    a, length = _rand_block(rng, n, 1, n // 2)
    if rng.random() < 0.2:
        a = max(a, 1)
        length = n - a
    delta = W * 10.0 ** rng.uniform(-2.0, 0.5)
    if rng.random() < 0.5:
        delta = -delta
    y = x.copy()
    y[a : a + length] += delta
    return y


def mv_kick(x, rng, ctx):
    """Gaussian kick on a contiguous window."""
    n = x.size
    a, length = _rand_block(rng, n, 2, n)
    sigma = W * 10.0 ** rng.uniform(-3.0, -0.3)
    y = x.copy()
    y[a : a + length] += rng.normal(0.0, sigma, length)
    return y


def mv_reflect(x, rng, ctx):
    """Reflect a block in place (the block-local version of the mirror symmetry)."""
    n = x.size
    a, length = _rand_block(rng, n, 3, 48)
    seg = x[a : a + length]
    y = x.copy()
    y[a : a + length] = (2.0 * seg.mean() - seg)[::-1]
    return y


def mv_transfer(x, rng, ctx):
    """Copy the internal pattern of one block onto another block (optionally swap).

    Shearing maps triangles a..a+L-1 onto b..b+L-1 with the same offsets, so a block's
    internal pattern is meaningful at any other position.
    """
    n = x.size
    for _ in range(20):
        length = int(rng.integers(3, 33))
        if 2 * length > n:
            continue
        a = int(rng.integers(0, n - length + 1))
        b = int(rng.integers(0, n - length + 1))
        if abs(a - b) >= length:
            break
    else:
        return None
    strength = float(rng.choice([0.25, 0.5, 1.0]))
    pa = x[a : a + length] - x[a : a + length].mean()
    pb = x[b : b + length] - x[b : b + length].mean()
    y = x.copy()
    y[b : b + length] += strength * (pa - pb)
    if rng.random() < 0.5:
        y[a : a + length] += strength * (pb - pa)
    return y


def mv_resize(x, rng, ctx):
    """Move 1-3 directions across a group boundary (a large upward jump in x)."""
    n = x.size
    d = np.diff(x)
    thr = float(rng.choice([0.15, 0.4, 1.0, 2.5])) * W
    cand = np.nonzero(d > thr)[0]  # boundary between indices cand and cand + 1
    if cand.size == 0:
        return None
    j = int(rng.choice(cand))
    m = int(rng.integers(1, 4))
    neg = d[d < 0]
    typical = float(np.median(neg)) if neg.size else -0.7 * W
    y = x.copy()
    if rng.random() < 0.5:
        # the right group grows to the left: j, j-1, ... continue its pattern backwards
        step = float(d[j + 1]) if (j + 2 < n and d[j + 1] < 0) else typical
        for t in range(m):
            k = j - t
            if k < 0:
                break
            y[k] = y[j + 1] - (t + 1) * step
    else:
        # the left group grows to the right: j+1, j+2, ... continue its pattern forwards
        step = float(d[j - 1]) if (j >= 1 and d[j - 1] < 0) else typical
        for t in range(m):
            k = j + 1 + t
            if k >= n:
                break
            y[k] = y[j] + (t + 1) * step
    return y


def mv_scan_shift(x, rng, ctx):
    """Scan the translation of one direction / one block and jump to another local
    minimum of that 1-D profile (a "re-insertion" of the block)."""
    n = x.size
    if rng.random() < 0.35:
        length = 1
        a = int(rng.integers(0, n))
    else:
        a, length = _rand_block(rng, n, 2, 40)
    span = float(x.max() - x.min())
    reach = float(rng.choice([2.0 * W, 6.0 * W, max(span, 2.0 * W)]))
    m = 81
    deltas = np.linspace(-reach, reach, m)
    vals = np.empty(m)
    y = x.copy()
    seg = x[a : a + length]
    for t in range(m):
        y[a : a + length] = seg + deltas[t]
        vals[t] = area(y)
    cands = [
        t
        for t in range(1, m - 1)
        if vals[t] <= vals[t - 1] and vals[t] <= vals[t + 1] and abs(deltas[t]) > 0.25 * W
    ]
    if not cands:
        return None
    cands.sort(key=lambda t: vals[t])
    pick = cands[min(int(rng.exponential(1.0)), len(cands) - 1)]
    y[a : a + length] = seg + deltas[pick]
    return y


def mv_crossover(x, rng, ctx):
    """Splice a contiguous block from another good solution (or a mirror image)."""
    n = x.size
    pool = ctx.donor_pool(x)
    if not pool:
        return None
    donor = pool[int(rng.integers(len(pool)))]
    if donor.size != n:
        return None
    a, length = _rand_block(rng, n, 4, n // 2)
    seg = donor[a : a + length]
    cur = x[a : a + length]
    if rng.random() < 0.5:
        new = seg - seg.mean() + cur.mean()  # keep the block's mean position
    else:
        new = seg - seg[0] + cur[0]  # keep the block's first offset
    if float(np.max(np.abs(new - cur))) < 1e-9:
        return None
    y = x.copy()
    y[a : a + length] = new
    return y


def group_boundaries(x, thr: float = 0.4) -> np.ndarray:
    """Indices i where a new group starts at i (offset jumps up by more than thr * W)."""
    return np.nonzero(np.diff(x) > thr * W)[0] + 1


def mv_group_crossover(x, rng, ctx):
    """Splice whole groups from another good arrangement: the cut points are the donor's own group
    boundaries, so groups are transplanted intact instead of being cut through the middle."""
    n = x.size
    pool = ctx.donor_pool(x)
    if not pool:
        return None
    donor = pool[int(rng.integers(len(pool)))]
    if donor.size != n:
        return None
    cuts = np.r_[0, group_boundaries(donor, float(rng.choice([0.4, 2.5]))), n]
    if cuts.size < 3:
        return None
    i = int(rng.integers(0, cuts.size - 1))
    j = min(cuts.size - 1, i + 1 + int(rng.integers(0, 4)))  # 1 to 4 consecutive donor groups
    a, b = int(cuts[i]), int(cuts[j])
    if b - a >= n:
        return None
    seg, cur = donor[a:b], x[a:b]
    new = seg - seg.mean() + cur.mean() if rng.random() < 0.5 else seg - seg[0] + cur[0]
    if float(np.max(np.abs(new - cur))) < 1e-9:
        return None
    y = x.copy()
    y[a:b] = new
    return y


MOVES: dict[str, Callable] = {
    "block_shift": mv_block_shift,
    "scan_shift": mv_scan_shift,
    "resize": mv_resize,
    "reflect": mv_reflect,
    "transfer": mv_transfer,
    "crossover": mv_crossover,
    "kick": mv_kick,
    "group_crossover": mv_group_crossover,
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "block_shift": 2.0,
    "scan_shift": 1.0,
    "resize": 2.0,
    "reflect": 1.0,
    "transfer": 1.5,
    "crossover": 1.5,
    "kick": 1.0,
    "group_crossover": 1.5,
}


# --------------------------------------------------------------------------------------
# exploration
# --------------------------------------------------------------------------------------


@dataclass
class ExploreResult:
    best: Polished
    start: Polished
    stats: dict
    history: list  # every new best: {"t", "step", "area", "move"}
    trace: list  # coarse time series: {"t", "step", "best", "current"}
    steps: int
    restarts: int
    seconds: float
    attempts: list = field(default_factory=list)  # (move name, polished area) of every completed hop


def explore(
    x0,
    seconds: float,
    seed: int = 0,
    donors: Sequence[np.ndarray] = (),
    moves: Sequence[str] | None = None,
    temperature: float = 1e-6,
    patience: int = 50,
    polish_iters: int = 150,
    trace_every: float = 5.0,
    move_weights: dict | None = None,
) -> ExploreResult:
    """Basin hopping with structural moves; Metropolis acceptance on polished areas.

    ``temperature`` is in area units (the top-5 leaderboard entries span 2.4e-4 and the
    top two differ by 2.6e-6).  After ``patience`` consecutive steps without a new best
    the walk restarts from the best solution.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    names = list(moves) if moves else list(DEFAULT_WEIGHTS)
    for name in names:
        if name not in MOVES:
            raise ValueError(f"unknown move {name!r}; available: {sorted(MOVES)}")
    mw = {**DEFAULT_WEIGHTS, **(move_weights or {})}
    names = [name for name in names if mw.get(name, 1.0) > 0.0] or names
    weights = np.array([max(mw.get(name, 1.0), 1e-9) for name in names], dtype=np.float64)
    weights /= weights.sum()

    start = polish(x0, max_iter=max(polish_iters, 300))
    cur = start
    best = start
    ctx = Context(donors=[np.asarray(d, dtype=np.float64) for d in donors])
    stats = {
        name: {
            "tries": 0,
            "not_applicable": 0,
            "returned_to_current": 0,
            "accepted": 0,
            "improved_current": 0,
            "new_best": 0,
            "best_gain": 0.0,
            "seconds": 0.0,
        }
        for name in names
    }
    history = [{"t": 0.0, "step": 0, "area": start.area, "move": "start"}]
    trace = [{"t": 0.0, "step": 0, "best": start.area, "current": start.area}]
    step = 0
    since_best = 0
    restarts = 0
    next_trace = trace_every
    attempts: list = []

    while True:
        now = time.perf_counter() - t0
        if now >= seconds:
            break
        if now >= next_trace:
            trace.append({"t": round(now, 3), "step": step, "best": best.area, "current": cur.area})
            next_trace += trace_every

        name = names[int(rng.choice(len(names), p=weights))]
        st = stats[name]
        t1 = time.perf_counter()
        y = MOVES[name](cur.x, rng, ctx)
        step += 1
        st["tries"] += 1
        if y is None:
            st["not_applicable"] += 1
            st["seconds"] += time.perf_counter() - t1
            continue
        res = polish(y, max_iter=polish_iters)
        st["seconds"] += time.perf_counter() - t1

        d = res.area - cur.area
        attempts.append((name, res.area))
        if abs(d) < 1e-12 and float(np.max(np.abs(res.x - cur.x))) < 1e-7:
            st["returned_to_current"] += 1
            since_best += 1
        else:
            if res.area < best.area - 1e-12:
                st["new_best"] += 1
                st["best_gain"] += best.area - res.area
                best = res
                since_best = 0
                history.append(
                    {
                        "t": round(time.perf_counter() - t0, 3),
                        "step": step,
                        "area": res.area,
                        "move": name,
                    }
                )
            else:
                since_best += 1
            accept = False
            if d < 0.0:
                st["improved_current"] += 1
                accept = True
            elif temperature > 0.0 and rng.random() < math.exp(-d / temperature):
                accept = True
            if accept:
                st["accepted"] += 1
                cur = res
            ctx.add_elite(res.area, res.x)

        if since_best >= patience:
            cur = best
            since_best = 0
            restarts += 1

    elapsed = time.perf_counter() - t0
    trace.append({"t": round(elapsed, 3), "step": step, "best": best.area, "current": cur.area})
    return ExploreResult(
        best=best,
        start=start,
        stats=stats,
        history=history,
        trace=trace,
        steps=step,
        restarts=restarts,
        seconds=elapsed,
        attempts=attempts,
    )


# --------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------


def make_payload(x, meta: dict | None = None) -> dict:
    """Round to <= 25 fractional digits and score the *rounded strings* exactly."""
    strings = offsets_to_strings(x)
    ex = exact_area(strings)
    score = float(ex)
    info = dict(meta or {})
    info["fast_area_of_float_offsets"] = area(np.asarray(x, dtype=np.float64))
    info["fast_minus_exact"] = info["fast_area_of_float_offsets"] - score
    info["target_to_take_first_place"] = TARGET
    info["beats_target"] = bool(score <= TARGET)
    return {
        "offsets": strings,
        "area_float": score,
        "exact_area": f"{ex.numerator}/{ex.denominator}",
        "meta": info,
    }


def random_start(rng: np.random.Generator, n: int = N) -> np.ndarray:
    """A crude from-scratch start: linear drift plus noise (no structure)."""
    j = np.arange(n, dtype=np.float64)
    x = -0.12 * j / n + rng.normal(0.0, 1.5 * W, n)
    return x - x[0]


def layout_start(rng: np.random.Generator, n: int = N) -> np.ndarray:
    """A random arrangement with the structure every good solution shares: about 16 small groups of
    3-11 neighbouring directions, nested in 4-5 large groups. Inside a group the offsets step down
    by about 0.7/128; between small groups they jump up by about 1/128, between large groups by
    5-10/128 (measured on the three best public solutions). The group sizes are sampled, so each
    start is a different layout; polishing then finds that layout's local minimum."""
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(rng.integers(3, 12)))
    sizes[-1] -= sum(sizes) - n
    if sizes[-1] < 3 and len(sizes) > 1:
        sizes[-2] += sizes.pop()
    starts = np.cumsum(sizes)[:-1]  # index where each later group starts
    n_large = int(rng.integers(3, 5))  # 3 or 4 large jumps -> 4 or 5 large groups
    large = set()
    if len(starts) > n_large:
        targets = n * (np.arange(1, n_large + 1) / (n_large + 1)) + rng.normal(0.0, 6.0, n_large)
        for t in targets:
            large.add(int(starts[int(np.argmin(np.abs(starts - t)))]))
    step = -rng.uniform(0.6, 0.8) * W
    x = np.zeros(n)
    boundary = set(int(v) for v in starts)
    for i in range(1, n):
        if i in large:
            x[i] = x[i - 1] + rng.uniform(5.5, 9.5) * W
        elif i in boundary:
            x[i] = x[i - 1] + rng.uniform(0.6, 1.9) * W
        else:
            x[i] = x[i - 1] + step * rng.uniform(0.85, 1.15)
    return x - x[0]


def resolve_start(spec: str, leaderboard: list[dict], rng: np.random.Generator):
    """--start: best | rank:K | mirror | random | file:PATH"""
    if spec == "best":
        s = leaderboard[0]
        return s["x"].copy(), f"leaderboard rank 1 ({s['agentName']}, {s['score']!r})"
    if spec.startswith("rank:"):
        k = int(spec.split(":", 1)[1])
        s = leaderboard[k - 1]
        return s["x"].copy(), f"leaderboard rank {k} ({s['agentName']}, {s['score']!r})"
    if spec == "mirror":
        s = leaderboard[0]
        return mirror(s["x"]), f"mirror image of leaderboard rank 1 ({s['agentName']})"
    if spec == "random":
        return random_start(rng), "random linear drift + noise"
    if spec.startswith("file:"):
        path = Path(spec.split(":", 1)[1])
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return to_array(data["offsets"]), f"file {path}"
    raise ValueError(f"unknown --start {spec!r} (best | rank:K | mirror | random | file:PATH)")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0, help="wall-clock budget of the search loop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="best", help="best | rank:K | mirror | random | file:PATH")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--temperature", type=float, default=1e-6, help="Metropolis temperature (area units)")
    ap.add_argument("--patience", type=int, default=50, help="steps without a new best before restarting")
    ap.add_argument("--polish-iters", type=int, default=150)
    ap.add_argument("--moves", default="", help="comma-separated subset of: " + ",".join(MOVES))
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed + 7919)
    leaderboard = load_leaderboard()
    x0, start_desc = resolve_start(args.start, leaderboard, rng)
    moves = [m for m in args.moves.split(",") if m] or None

    result = explore(
        x0,
        seconds=args.seconds,
        seed=args.seed,
        donors=[s["x"] for s in leaderboard],
        moves=moves,
        temperature=args.temperature,
        patience=args.patience,
        polish_iters=args.polish_iters,
    )

    meta = {
        "problem": ARENA_SLUG,
        "seed": args.seed,
        "seconds_requested": args.seconds,
        "seconds_used": round(result.seconds, 3),
        "start": start_desc,
        "start_area_unpolished_fast": area(x0),
        "start_area_polished_fast": result.start.area,
        "start_polish_status": result.start.status,
        "start_polish_grad_inf": result.start.grad_inf,
        "best_area_fast": result.best.area,
        "best_grad_inf": result.best.grad_inf,
        "best_polish_status": result.best.status,
        "steps": result.steps,
        "restarts": result.restarts,
        "temperature": args.temperature,
        "patience": args.patience,
        "polish_iters": args.polish_iters,
        "moves": result.stats,
        "history_new_best": result.history,
        "trace": result.trace,
        "leaderboard_best_score": leaderboard[0]["score"],
    }
    payload = make_payload(result.best.x, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"start: {start_desc}")
    print(f"steps={result.steps} restarts={result.restarts} seconds={result.seconds:.1f}")
    for name, st in result.stats.items():
        print(
            f"  {name:12s} tries={st['tries']:5d} returned={st['returned_to_current']:5d} "
            f"accepted={st['accepted']:5d} improved_cur={st['improved_current']:5d} "
            f"new_best={st['new_best']:3d} gain={st['best_gain']:.3e} sec={st['seconds']:.1f}"
        )
    print(f"fast area (float offsets):   {result.best.area!r}")
    print(f"EXACT area (rounded strings): {payload['area_float']!r}")
    print(f"leaderboard best:             {leaderboard[0]['score']!r}")
    print(f"target for #1:                {TARGET!r}  beats_target={payload['meta']['beats_target']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
