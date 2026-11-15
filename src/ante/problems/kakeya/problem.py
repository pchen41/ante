from __future__ import annotations

import time

import numpy as np

from ante.core.problem import AttemptBatch, KernelSpec, Knob, Problem

from . import exact

MOVE_NAMES = ["block_shift", "scan_shift", "resize", "reflect", "transfer", "crossover", "kick", "group_crossover"]


class KakeyaNeedle(Problem):
    slug = "kakeya-needle-128"
    title = "Discretized Kakeya needle (128 triangles)"
    maximize = False

    def kernels(self) -> dict[str, KernelSpec]:
        return {
            "hop": KernelSpec(
                "hop",
                "Basin hopping over arrangements of the 128 triangles: apply a structural move, slide to the nearest "
                "local minimum (Newton, exact piecewise-quadratic area), accept by Metropolis. One attempt = one chain "
                "of `chain_seconds`; its outcome is the best new local minimum the chain found.",
                [
                    Knob("start", "choice", "random", choices=["random", "layout", "public", "record", "elite"], doc="where each chain starts: random = structureless noise; layout = a freshly sampled group layout (about 16 small groups of 3-11 directions nested in 4-5 large groups, the structure all good solutions share; every start is a different layout); public = a random public top-5 solution or its mirror image; record = the public record; elite = one of our own best arrangements"),
                    Knob("chain_seconds", "float", 60.0, 5.0, 3600.0, doc="length of one chain (one attempt)"),
                    Knob("temperature", "float", 1e-6, 1e-9, 1e-3, doc="Metropolis temperature in area units; neighbouring minima are typically 1e-6 to 1e-4 apart"),
                    Knob("patience", "int", 50, 5, 5000, doc="hops without a new chain best before the chain returns to its best arrangement"),
                    Knob("polish_iters", "int", 150, 20, 400, doc="Newton iteration cap per hop"),
                    Knob("donors", "choice", "all", choices=["all", "own", "none"], doc="which arrangements the crossover move may splice blocks from: public top-5 plus our own best, only our own, or none"),
                ]
                + [Knob(f"w_{m}", "float", w, 0.0, 10.0, doc=f"relative frequency of the `{m}` move") for m, w in zip(MOVE_NAMES, [2.0, 1.0, 2.0, 1.0, 1.5, 1.5, 1.0, 1.5])],
            )
        }

    def _arrays(self, sols: list[dict]):
        from .fast import to_array

        return [to_array(s["offsets"]) for s in sols]

    def run(self, kernel_name: str, params: dict, seed: int, seconds: float, elites: list[dict] | None = None) -> AttemptBatch:
        from . import search as S
        from .fast import mirror

        p = self.kernels()[kernel_name].resolve(params)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        public = self._arrays([s["data"] for s in self.public_solutions()])
        mine = self._arrays(elites or [])
        donors = {"all": public + mine, "own": mine, "none": []}[p["donors"]]
        weights = {m: p[f"w_{m}"] for m in MOVE_NAMES}
        public_areas = [float(S.area(x)) for x in public]
        scores: list[float] = []
        best = None
        hops = 0
        start_score = None
        started = improved = top_started = top_improved = 0
        top_bar = sorted(public_areas)[2]  # area of the third-best public solution
        second_best = sorted(public_areas)[1]
        refs = [y - y.mean() for x in public for y in (x, mirror(x))]  # public arrangements and their mirror images
        while True:
            left = seconds - (time.time() - t0)
            if left < 2.0 or (scores and left < 0.5 * p["chain_seconds"]):
                break
            start = p["start"]
            if start == "elite" and not mine:
                start = "public"
            if start == "random":
                x0 = S.random_start(rng)
            elif start == "layout":
                x0 = S.layout_start(rng)
            elif start == "record":
                x0 = public[0]
            elif start == "public":
                x0 = public[int(rng.integers(len(public)))]
                x0 = mirror(x0) if rng.random() < 0.5 else x0
            else:
                x0 = mine[int(rng.integers(len(mine)))]
            fresh = start in ("random", "layout")
            r = S.explore(
                x0, min(p["chain_seconds"], left), seed=int(rng.integers(1 << 31)), donors=donors,
                temperature=p["temperature"], patience=p["patience"], polish_iters=p["polish_iters"], move_weights=weights,
                known_areas=public_areas, exclude_start=not fresh,
            )
            hops += r.steps
            if not fresh and (start_score is None or r.start.area < start_score):
                start_score = r.start.area
            # a chain only counts for what it newly found: not its own start, and not a public
            # arrangement that crossover happened to rebuild
            known = public_areas + ([r.start.area] if not fresh else [])
            new = [(a, i) for i, (_, a) in enumerate(r.attempts) if all(abs(a - k) > 1e-12 for k in known)]
            outcome = min(new)[0] if new else None
            cand = r.best_new  # what the worker hands back is its best NEW arrangement, never a rebuilt public one
            if not fresh:
                better = outcome is not None and outcome < r.start.area - 1e-12
                started += 1
                improved += better
                if r.start.area <= top_bar + 1e-12:
                    # climbing = a new arrangement strictly better than the second-best public one, reached from a top start.
                    # (Merely drifting from our own arrangement towards a public one does not count.)
                    top_started += 1
                    top_improved += (outcome is not None and outcome < second_best - 1e-12 and r.best_new is not None
                                     and S.is_distinct(r.best_new.x, refs))
            if outcome is not None:
                scores.append(float(outcome))
            if cand is not None and (best is None or cand.area < best.area):
                best = cand
        best_solution = best_score = None
        if best is not None:
            strings = S.offsets_to_strings(best.x)
            best_solution = {"offsets": strings}
            best_score = float(exact.exact_area(strings))
        return AttemptBatch(scores, best_score, best_solution, time.time() - t0, {"hops": hops, "params": p, "started_from_known": int(started), "improved_on_start": int(improved), "top_started": int(top_started), "top_improved": int(top_improved)}, start_score)

    def exact_score(self, solution: dict) -> float:
        return float(exact.exact_area(solution["offsets"]))

    def lineage(self, kernel: str, params: dict) -> str:
        if params.get("start") in ("random", "layout") and params.get("donors") == "none":
            return "from scratch"
        if params.get("start") in ("random", "layout") and params.get("donors") == "own":
            return "from scratch, recombining our own arrangements"
        return "builds on public solutions (start or crossover donors)"

    def canonical(self, solution: dict) -> str:
        xs = exact.parse_offsets(solution["offsets"])
        return ",".join(str(x - xs[0]) for x in xs)

    def is_public(self, solution: dict) -> str | None:
        """Same arrangement as a public solution?

        Two tests. (1) Offsets equal up to translation, mirror image and 1e-9 rounding. (2) Exactly the
        same area: triangles that are completely covered by the others can slide without changing the
        union, so one arrangement has many offset vectors; an area that matches a public score to the
        last float digit is treated as that public arrangement, not as ours.
        """
        import numpy as np

        from .fast import mirror, to_array

        x = to_array(solution["offsets"])
        x = x - x[0]
        score = float(exact.exact_area(solution["offsets"]))
        for s in self.public_solutions():
            if abs(score - float(s["score"])) <= 1e-13:
                return s["agentName"]
            y = to_array(s["data"]["offsets"])
            for z in (y, mirror(y)):
                if float(np.max(np.abs((z - z[0]) - x))) < 1e-9:
                    return s["agentName"]
        return None

    def audit(self, solution: dict) -> dict:
        from fractions import Fraction

        area = exact.exact_area(solution["offsets"])
        return {
            "exact_area": f"{area.numerator}/{area.denominator}",
            "exact_area_float": float(area),
            "margin_below_target": float(Fraction(self.target()) - area),
            "note": "area is an exact rational computed by our own integer implementation; the margin is exact, not a float difference",
        }

    def brief(self) -> str:
        return (
            "Choose offsets x_1..x_128 for triangles T_j = conv{(x_j,0),(x_j+1/128,0),(x_j+j/128,1)} to minimise the area of "
            "their union (exact rational verifier). Good solutions bundle neighbouring directions into groups (about 15 small "
            "groups nested in 3-5 large ones) whose tops overlap; different group layouts are different local minima. The top "
            "three public solutions are three different layouts, each a strict local minimum; from each of them, thousands of "
            "structural moves found no improvement and the nearest other minimum is about 1.2e-6 higher. From a random "
            "arrangement, 4 minutes of basin hopping reaches about 0.10700 (4th-5th place level); the record is 0.1067514. "
            "Chains whose crossover donors include the public solutions reach 0.10677-0.10679 within 20 s but mostly by rebuilding the "
            "second-place arrangement, which does not count as a discovery. The record needs a group layout nobody has tried: "
            "start=layout samples new layouts (with donors=none or own these are genuinely new arrangements), and group_crossover "
            "transplants whole groups between arrangements. Temperatures of 3e-6 to 3e-5 with patience 300+ let a chain cross the "
            "1.2e-6 ridges between neighbouring minima; block_shift and kick mostly fall back to the same minimum."
        )
