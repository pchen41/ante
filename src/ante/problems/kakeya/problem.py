from __future__ import annotations

import time

import numpy as np

from ante.core.problem import AttemptBatch, KernelSpec, Knob, Problem

from . import exact

MOVE_NAMES = ["block_shift", "scan_shift", "resize", "reflect", "transfer", "crossover", "kick"]


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
                    Knob("start", "choice", "random", choices=["random", "public", "record", "elite"], doc="where each chain starts: a random arrangement, a random public top-5 solution (or its mirror image), the public record, or one of our own best arrangements"),
                    Knob("chain_seconds", "float", 60.0, 5.0, 3600.0, doc="length of one chain (one attempt)"),
                    Knob("temperature", "float", 1e-6, 1e-9, 1e-3, doc="Metropolis temperature in area units; neighbouring minima are typically 1e-6 to 1e-4 apart"),
                    Knob("patience", "int", 50, 5, 5000, doc="hops without a new chain best before the chain returns to its best arrangement"),
                    Knob("polish_iters", "int", 150, 20, 400, doc="Newton iteration cap per hop"),
                ]
                + [Knob(f"w_{m}", "float", w, 0.0, 10.0, doc=f"relative frequency of the `{m}` move") for m, w in zip(MOVE_NAMES, [2.0, 1.0, 2.0, 1.0, 1.5, 1.5, 1.0])],
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
        donors = public + mine
        weights = {m: p[f"w_{m}"] for m in MOVE_NAMES}
        scores: list[float] = []
        best = None
        hops = 0
        start_score = None
        while True:
            left = seconds - (time.time() - t0)
            if left < 2.0 or (scores and left < 0.5 * p["chain_seconds"]):
                break
            start = p["start"]
            if start == "elite" and not mine:
                start = "public"
            if start == "random":
                x0 = S.random_start(rng)
            elif start == "record":
                x0 = public[0]
            elif start == "public":
                x0 = public[int(rng.integers(len(public)))]
                x0 = mirror(x0) if rng.random() < 0.5 else x0
            else:
                x0 = mine[min(int(rng.exponential(1.5)), len(mine) - 1)]
            r = S.explore(
                x0, min(p["chain_seconds"], left), seed=int(rng.integers(1 << 31)), donors=donors,
                temperature=p["temperature"], patience=p["patience"], polish_iters=p["polish_iters"], move_weights=weights,
            )
            hops += r.steps
            if start != "random" and (start_score is None or r.start.area < start_score):
                start_score = r.start.area
            if start == "random":
                outcome, cand = r.best.area, r.best
            else:
                # a chain that starts at a known arrangement only counts for what it newly found
                new = [a for _, a in r.attempts if abs(a - r.start.area) > 1e-12]
                outcome, cand = (min(new) if new else None), r.best
            if outcome is not None:
                scores.append(float(outcome))
            if best is None or cand.area < best.area:
                best = cand
        best_solution = best_score = None
        if best is not None:
            strings = S.offsets_to_strings(best.x)
            best_solution = {"offsets": strings}
            best_score = float(exact.exact_area(strings))
        return AttemptBatch(scores, best_score, best_solution, time.time() - t0, {"hops": hops, "params": p}, start_score)

    def exact_score(self, solution: dict) -> float:
        return float(exact.exact_area(solution["offsets"]))

    def canonical(self, solution: dict) -> str:
        xs = exact.parse_offsets(solution["offsets"])
        return ",".join(str(x - xs[0]) for x in xs)

    def is_public(self, solution: dict) -> str | None:
        """Same arrangement as a public solution (up to translation, mirror image and 1e-9 rounding)?"""
        import numpy as np

        from .fast import mirror, to_array

        x = to_array(solution["offsets"])
        x = x - x[0]
        for s in self.public_solutions():
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
            "arrangement, 4 minutes of basin hopping reaches about 0.10700 (4th-5th place level); the record is 0.1067514."
        )
