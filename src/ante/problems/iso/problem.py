"""Loop plug-in for AlphaEvolve problem 59: the largest subset of the n x n grid with no isosceles triangle.

Records (AlphaEvolve, arXiv 2511.02864, section 6.39): 112 points for n = 64 and 164 for n = 100. This is
not an EinsteinArena problem; the record and the published sets come from the AlphaEvolve repository of
problems (see data/alphaevolve/59/SOURCE.md) and the "official verifier" is that repository's checker.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ante.core import safe_code
from ante.core.problem import AttemptBatch, KernelSpec, Knob, Problem

from . import exact

DATA59 = Path(__file__).resolve().parents[4] / "data" / "alphaevolve" / "59"
AXES = {"centre": (-0, -0), "offset-x": (-1, 0), "offset-both": (-1, -1)}
PRIORITY_VARS = ("x", "y", "n")
DEFAULT_PRIORITY = "-min(x, y)"


@lru_cache(maxsize=None)
def _circles(n: int):
    from . import kernel as K

    return K.circle_table(n)


@lru_cache(maxsize=None)
def _table(n: int, axes: str, band: int, mirror_x: bool = True, mirror_y: bool = True):
    from . import sym as S

    dx, dy = AXES[axes]
    orb, osize = S.orbit_table(n, n - 1 + dx, n - 1 + dy, band, mirror_x, mirror_y)
    x0 = max(0, dx) if mirror_x else 0  # left and bottom edges of the region the symmetry maps to itself
    y0 = max(0, dy) if mirror_y else 0
    return orb, osize, (orb[:, 0] // n - x0).astype(np.float64), (orb[:, 0] % n - y0).astype(np.float64)


def _priority(n: int, gx, gy, expr: str) -> np.ndarray:
    """prio[g] from the planner's expression, standardised to zero mean and unit spread."""
    try:
        f = safe_code.evaluator(expr, PRIORITY_VARS)
    except ValueError:
        f = safe_code.evaluator(DEFAULT_PRIORITY, PRIORITY_VARS)
    p = np.array([f(float(x), float(y), float(n)) for x, y in zip(gx, gy)], dtype=np.float64)
    sd = float(p.std())
    return (p - p.mean()) / sd if sd > 1e-12 else np.zeros_like(p)


def _timed(fn, start, seconds: float, seed: int):
    """Call fn(start, iters, seed) -> (orbit ids, points, steps) in slices until `seconds` have passed, continuing from the best."""
    t0 = time.time()
    best, bk, steps, iters, i = start, None, 0, 30, 0
    while True:
        t1 = time.time()
        ids, k, s = fn(best, iters, (seed + 7919 * i) % (1 << 31))
        i += 1
        steps += int(s)
        if bk is None or k > bk:
            best, bk = ids, int(k)
        left = seconds - (time.time() - t0)
        if left <= 0:
            return best, bk, steps
        rate = max(int(s), 1) / max(time.time() - t1, 1e-4)
        iters = int(max(10, min(rate * min(1.0, left), 100_000)))


class IsoGrid(Problem):
    maximize = True
    integer_scores = True
    n = 100
    RECORD = {64: 112, 100: 164}
    PREVIOUS = {64: 108, 100: 160}

    # ---- the published record (static; this problem is not on EinsteinArena) -------------------
    def meta(self) -> dict:
        return {"id": None, "minImprovement": 1, "source": "AlphaEvolve repository of problems, problem 59"}

    def leaderboard(self) -> list[dict]:
        return [{"agentName": "AlphaEvolve (arXiv 2511.02864)", "bestScore": self.RECORD[self.n]},
                {"agentName": "previous best, from the AlphaEvolve notebook", "bestScore": self.PREVIOUS[self.n]}]

    def public_solutions(self) -> list[dict]:
        return _public(self.n)

    def verifier_source(self) -> str:
        return (DATA59 / "verifier.py.txt").read_text()

    # ---- kernels -------------------------------------------------------------------------------
    def kernels(self) -> dict[str, KernelSpec]:
        return {
            "orbits": KernelSpec(
                "orbits",
                "Search over mirror-symmetric sets, then break the symmetry. Stage 1 works on whole orbits (the 4 mirror images of a point "
                "under the two reflections of the grid): it looks for the largest set of orbits with no isosceles triangle. Stage 2 keeps that "
                "core and adds mirror pairs under one reflection only, then single points. One attempt = one chain of `chain_seconds`; its "
                "outcome is the number of points of the final set.",
                [
                    Knob("start", "choice", "scratch", choices=["scratch", "elite", "record"], doc="where stage 1 starts: the empty set, the complete orbits of one of our own best sets, or the complete orbits of the published record"),
                    Knob("method", "choice", "ils", choices=["ils", "tabu"], doc="stage 1 search. ils: remove a few orbits and refill the hole (greedy by priority, or exact). tabu: force in one more orbit than fits, then repair by swapping orbits in and out until no violation is left"),
                    Knob("chain_seconds", "float", 20.0, 4.0, 1800.0, doc="length of one chain (one attempt)"),
                    Knob("refine", "float", 0.2, 0.0, 0.8, doc="fraction of the chain spent in stage 2 (mirror pairs, then single points)"),
                    Knob("axes", "choice", "centre", choices=list(AXES), doc="where the two mirror axes sit: through the centre of the grid; the vertical axis moved half a cell (the set then lives in n-1 columns); or both moved"),
                    Knob("band", "int", 0, 0, 50, doc="if > 0, stage 1 only uses orbits within this many cells of the edge (0 = the whole grid)"),
                    Knob("priority", "code", DEFAULT_PRIORITY, variables=PRIORITY_VARS, doc=(
                        "YOU WRITE THIS: an expression that scores every orbit; orbits with a higher value are tried first when a hole is refilled "
                        "(ils) or preferred for insertion (tabu). x and y are the distances of the orbit's lower-left point from the left and "
                        "bottom edges (0 .. n/2 - 1), n is the grid size. Allowed: numbers, x, y, n, + - * / // % **, comparisons, `a if c else b`, "
                        "and, or, not, abs, min, max, sqrt, exp, log, sin, cos, floor, ceil. Values are standardised, so only the ranking matters")),
                    Knob("noise", "float", 0.4, 0.0, 20.0, doc="random spread added to the standardised priority when ordering a refill (0 = strict priority order, large = random order)"),
                    Knob("ruin_max", "int", 6, 1, 24, doc="ils: each step removes between 1 and this many orbits"),
                    Knob("p_cluster", "float", 0.7, 0.0, 1.0, doc="ils: probability that the removed orbits are neighbours rather than random"),
                    Knob("p_exact", "float", 0.2, 0.0, 1.0, doc="ils: probability that a hole is refilled by exhaustive search (the largest refill) instead of greedily"),
                    Knob("exact_nodes", "int", 400, 20, 20000, doc="ils: search-node limit of one exhaustive refill"),
                    Knob("temp", "float", 0.6, 0.05, 8.0, doc="ils: Metropolis temperature in points for accepting a smaller set"),
                    Knob("max_drop", "int", 4, 0, 40, doc="ils: never walk more than this many points below the chain's best"),
                    Knob("tenure", "int", 8, 1, 60, doc="tabu: steps for which a swap may not be undone"),
                    Knob("prio_weight", "float", 0.3, 0.0, 5.0, doc="tabu: how much one unit of standardised priority offsets one violation when choosing the orbit to insert"),
                    Knob("stall", "int", 3000, 100, 200000, doc="tabu: swaps without success before returning to the best valid set"),
                    Knob("kick", "int", 2, 0, 12, doc="tabu: orbits removed at random after a stall"),
                ],
            )
        }

    def warm_params(self) -> list[dict]:
        return [{"chain_seconds": 4.0, "method": "ils", "p_exact": 0.5}, {"chain_seconds": 4.0, "method": "tabu"}]

    # ---- one worker ----------------------------------------------------------------------------
    def _refine(self, core: list, seconds: float, seed: int) -> list:
        """Stage 2: add mirror pairs under one reflection (each tried from the core), then single points."""
        from . import sym as S

        n, ct = self.n, _circles(self.n)
        best = core
        for i, (mx, my) in enumerate(((True, False), (False, True))):
            orb, osize, gx, gy = _table(n, "centre", 0, mx, my)
            prio = _priority(n, np.minimum(gx, n - 1 - gx), np.minimum(gy, n - 1 - gy), DEFAULT_PRIORITY)
            st = S.points_to_orbits(n, orb, osize, core)
            ids, _, _ = _timed(lambda s, it, sd: S.run_sym(n, orb, osize, prio, s, it, sd, 0.7, 1, 3, 0.5, 2, 0.4, 0.3, 300, *ct, 10**9), st, 0.4 * seconds, seed + i)
            cand = S.orbits_to_points(n, orb, ids)
            if len(cand) > len(best):
                best = cand
        orb, osize, gx, gy = _table(n, "centre", 0, False, False)  # single cells: orbit id == cell id
        prio = _priority(n, np.minimum(gx, n - 1 - gx), np.minimum(gy, n - 1 - gy), DEFAULT_PRIORITY)
        st = np.array([x * n + y for x, y in best], dtype=np.int64)
        ids, _, _ = _timed(lambda s, it, sd: S.run_sym(n, orb, osize, prio, s, it, sd, 0.7, 1, 3, 0.5, 2, 0.4, 0.3, 300, *ct, 10**9), st, 0.2 * seconds, seed + 2)
        cand = S.orbits_to_points(n, orb, ids)
        return cand if len(cand) > len(best) else best

    def run(self, kernel_name: str, params: dict, seed: int, seconds: float, elites: list[dict] | None = None) -> AttemptBatch:
        from . import sym as S

        p = self.kernels()[kernel_name].resolve(params)
        n, ct = self.n, _circles(self.n)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        orb, osize, gx, gy = _table(n, p["axes"], p["band"])
        prio = _priority(n, gx, gy, p["priority"])
        if p["method"] == "tabu":
            def stage1(s, it, sd):
                return S.run_tabu(n, orb, osize, prio, s, it, sd, p["tenure"], p["prio_weight"], p["stall"], p["kick"], *ct, 10**9)
        else:
            def stage1(s, it, sd):
                return S.run_sym(n, orb, osize, prio, s, it, sd, p["p_cluster"], 1, p["ruin_max"], p["temp"], p["max_drop"], p["noise"], p["p_exact"], p["exact_nodes"], *ct, 10**9)
        mine = [[tuple(q) for q in e["points"]] for e in (elites or [])]
        scores: list[float] = []
        best: list | None = None
        start_score = None
        steps = started = improved = top_started = top_improved = 0
        ladder: dict[str, list[int]] = {}
        cores: dict[str, int] = {}
        while True:
            left = seconds - (time.time() - t0)
            if left < 3.0 or (scores and left < 0.6 * p["chain_seconds"]):
                break
            chain = min(p["chain_seconds"], left - 0.5)
            start = p["start"]
            if start == "elite" and not mine:
                start = "scratch"
            if start == "record":
                x0 = [tuple(q) for q in self.public_solutions()[0]["data"]["points"]]
            elif start == "elite":
                top = len(mine[0])
                pool = [e for e in mine if len(e) >= top - 4]  # walk broadly over our near-best sets
                x0 = pool[int(rng.integers(len(pool)))]
            else:
                x0 = None
            st = S.points_to_orbits(n, orb, osize, x0) if x0 else np.zeros(0, dtype=np.int64)
            sd = int(rng.integers(1 << 31))
            ids, _, s1 = _timed(stage1, st, chain * (1.0 - p["refine"]), sd)
            steps += s1
            core = S.orbits_to_points(n, orb, ids)
            cores[str(len(ids))] = cores.get(str(len(ids)), 0) + 1
            t2 = chain * p["refine"]
            final = self._refine(core, t2, sd) if t2 >= 0.6 else core
            size = exact.check(final, n)  # never report an unchecked set
            scores.append(float(size))
            if x0:
                start_score = max(start_score or 0.0, float(len(x0)))
                started += 1
                improved += size > len(x0)
                rung = ladder.setdefault(str(len(x0)), [0, 0])
                rung[0] += 1
                rung[1] += int(size > len(x0))
                if len(x0) >= self.PREVIOUS[n]:
                    top_started += 1
                    top_improved += int(size > len(x0))
            if best is None or size > len(best):
                best = final
        meta = {"steps": steps, "params": p, "core_orbits": cores, "started_from_known": int(started), "improved_on_start": int(improved),
                "top_started": int(top_started), "top_improved": int(top_improved)}
        if ladder:
            meta["ladder"] = ladder
        best_solution = best_score = None
        if best is not None:
            best_solution = {"points": [[int(x), int(y)] for x, y in sorted(best)]}
            best_score = float(len(best))
        return AttemptBatch(scores, best_score, best_solution, time.time() - t0, meta, start_score)

    # ---- checking ------------------------------------------------------------------------------
    def exact_score(self, solution: dict) -> float:
        return float(exact.check(solution["points"], self.n))

    def canonical(self, solution: dict) -> str:
        n = self.n
        pts = [(int(x), int(y)) for x, y in solution["points"]]
        maps = (lambda x, y: (x, y), lambda x, y: (n - 1 - x, y), lambda x, y: (x, n - 1 - y), lambda x, y: (n - 1 - x, n - 1 - y),
                lambda x, y: (y, x), lambda x, y: (n - 1 - y, x), lambda x, y: (y, n - 1 - x), lambda x, y: (n - 1 - y, n - 1 - x))
        return min(json.dumps(sorted(f(x, y) for x, y in pts)) for f in maps)

    def lineage(self, kernel: str, params: dict) -> str:
        return "from scratch" if params.get("start") == "scratch" else "starts from the published record or from our own earlier sets"

    def audit(self, solution: dict) -> dict:
        n = self.n
        pts = [(int(x), int(y)) for x, y in solution["points"]]
        s = set(pts)
        full = sum(1 for x, y in pts if (n - 1 - x, y) in s and (x, n - 1 - y) in s and (n - 1 - x, n - 1 - y) in s)
        most = 0
        for bx, by in pts:
            seen: dict[int, int] = {}
            for ax, ay in pts:
                if (ax, ay) != (bx, by):
                    d2 = (ax - bx) ** 2 + (ay - by) ** 2
                    seen[d2] = seen.get(d2, 0) + 1
            most = max(most, max(seen.values(), default=0))
        return {
            "points": len(pts), "distinct": len(s) == len(pts), "all_inside_grid": all(0 <= x < n and 0 <= y < n for x, y in pts),
            "largest_number_of_points_at_one_distance_from_any_point": most, "points_in_complete_mirror_orbits": full,
            "note": "valid iff distinct, inside the grid, and no point has two others at the same distance (the largest count above is 1); integer arithmetic only",
        }

    def brief(self) -> str:
        n, rec, prev = self.n, self.RECORD[self.n], self.PREVIOUS[self.n]
        known = ("the published 112-point set is 27 complete orbits (108 points) plus 2 mirror pairs under one reflection only"
                 if n == 64 else "the published 164-point set is exactly 41 complete orbits, with no extra point at all")
        measured = ("from the empty set, 12-second ils chains end with a core of 24 orbits about 60% of the time, 25 about 25%, 26 about 5%; "
                    "starting stage 2 from the published 27-orbit core reaches 112 within a second, so a 28-orbit core, or a different 27-orbit "
                    "core that admits 3 extra pairs, would beat the record"
                    if n == 64 else
                    "from the empty set, chains of a few minutes end with cores of 36 to 38 orbits (144 to 152 points); stage 2 from the published "
                    "41-orbit core finds nothing, so the record falls to a 42-orbit core or to a 41-orbit core that admits an extra pair or point")
        return (
            f"Choose as many points as possible from the {n} x {n} integer grid so that no point is equally far from two others (no isosceles "
            f"triangle; three equally spaced points on a line are forbidden too). AlphaEvolve problem 59. Best known: {rec} points "
            f"(AlphaEvolve, 2025), previously {prev}. Proven optimal values are 28 for n = 16 and 56 for n = 32, and the best known sizes follow "
            f"about 16n/9, which would be {16 * n / 9:.0f} here. A solution is checked by ten lines of integer arithmetic. "
            f"Structure: the best sets are (nearly) invariant under both mirror reflections of the grid, so they are unions of orbits of 4 points; "
            f"{known}. Almost every orbit of the published set lies within 8 cells of the edge of the grid, in an irregular pattern. "
            f"Measured here: {measured}. The priority expression is the main lever you have over where stage 1 looks: the default prefers "
            "orbits near the edge; it is unknown which positions (corners, mid-edges, particular residues, a few interior orbits) matter."
        )


class IsoGrid64(IsoGrid):
    slug = "isosceles-grid-64"
    title = "Isosceles-free points in the 64 x 64 grid (AlphaEvolve problem 59)"
    n = 64


class IsoGrid100(IsoGrid):
    slug = "isosceles-grid-100"
    title = "Isosceles-free points in the 100 x 100 grid (AlphaEvolve problem 59)"
    n = 100


@lru_cache(maxsize=None)
def _public(n: int) -> list[dict]:
    out = []
    for name, f in (("AlphaEvolve (arXiv 2511.02864)", f"record_{n}.json"), ("previous best, from the AlphaEvolve notebook", f"previous_{n}.json")):
        pts = json.loads((DATA59 / f).read_text())["points"]
        out.append({"agentName": name, "score": len(pts), "data": {"points": pts}})
    return out
