from __future__ import annotations

import time

import numpy as np

from ante.core.problem import AttemptBatch, KernelSpec, Knob, Problem

from . import exact, kernel

MODES = {"max": 0, "lp": 1, "excess": 2}


class FlatPolynomials(Problem):
    slug = "flat-polynomials"
    title = "Flat polynomials (degree 69, +-1 coefficients)"
    maximize = False

    def kernels(self) -> dict[str, KernelSpec]:
        return {
            "tabu": KernelSpec(
                "tabu",
                "Tabu search over single coefficient flips, scored on a coarse K-point grid of the unit circle; "
                "each episode ends after `restart_after` non-improving steps and is polished on a fine grid. "
                "One attempt = one episode.",
                [
                    Knob("grid", "choice", 256, choices=[128, 192, 256, 384, 512], doc="coarse grid size K (cost per step is proportional to K; a small K lets peaks hide between grid points)"),
                    Knob("objective", "choice", "excess", choices=list(MODES), doc="what the walk descends: max |p|^2, an L_2p norm, or the squared excess over a moving target level"),
                    Knob("p", "float", 4.0, 2.0, 32.0, doc="exponent for objective=lp"),
                    Knob("margin", "float", 0.005, 0.0, 0.1, doc="objective=excess: target level sits this fraction below the episode's best max |p|^2"),
                    Knob("tenure_lo", "int", 6, 1, 40, doc="minimum tabu tenure"),
                    Knob("tenure_hi", "int", 18, 2, 60, doc="maximum tabu tenure"),
                    Knob("restart_after", "int", 3000, 50, 2_000_000, doc="non-improving steps before an episode ends"),
                    Knob("kick", "int", 0, 0, 35, doc="0: every episode starts from a random sequence; k>0: from the previous episode's best with k random flips"),
                    Knob("start", "choice", "random", choices=["random", "record", "elite"], doc="first episode starts from a random sequence, the public record, or our own best sequence"),
                    Knob("fine_grid", "choice", 1024, choices=[512, 1024, 2048], doc="grid used to polish and score each episode"),
                ],
            )
        }

    def run(self, kernel_name: str, params: dict, seed: int, seconds: float, elites: list[dict] | None = None) -> AttemptBatch:
        p = self.kernels()[kernel_name].resolve(params)
        if p["tenure_hi"] < p["tenure_lo"]:
            p["tenure_hi"] = p["tenure_lo"]
        t0 = time.time()
        rng = np.random.default_rng(seed)
        wc, ws = kernel.make_tables(p["grid"], offset=float(rng.random()))
        wcf, wsf = kernel.make_tables(p["fine_grid"])
        start = p["start"] if (p["start"] != "elite" or elites) else "random"
        use_c0 = start != "random"
        src = self.public_solutions()[0]["data"] if start == "record" else (elites[0] if start == "elite" else None)
        c0 = np.array(src["coefficients"], dtype=np.int8) if use_c0 else np.zeros(kernel.N, dtype=np.int8)
        start_level = kernel.score_from_sq(kernel.fine_sup(c0, wcf, wsf)) if use_c0 else None
        chunk = max(20_000, int(1.2 * p["restart_after"]))  # at least one episode can finish per call
        scores: list[float] = []
        best_sq, best_c, steps = np.inf, None, 0
        ep_c = np.zeros((chunk // p["restart_after"] + 2, kernel.N), dtype=np.int8)
        ep_m = np.zeros(ep_c.shape[0], dtype=np.float32)
        sup = np.zeros(ep_c.shape[0], dtype=np.float32)
        carry_c, carry = c0, use_c0
        while time.time() - t0 < seconds - 1.0:
            n_ep, used = kernel.tabu_episodes(
                carry_c, carry, wc, ws, chunk, int(rng.integers(1 << 31)), MODES[p["objective"]], p["p"], p["margin"],
                p["tenure_lo"], p["tenure_hi"], p["restart_after"], p["kick"], ep_c, ep_m,
            )
            steps += used
            carry = False
            if n_ep == 0:
                continue
            kernel.polish_all(ep_c, n_ep, wcf, wsf, 20, sup)
            # an episode that only returns the sequence it started from found nothing
            scores.extend(v for v in (kernel.score_from_sq(s) for s in sup[:n_ep]) if start_level is None or abs(v - start_level) > 1e-6)
            i = int(np.argmin(sup[:n_ep]))
            if sup[i] < best_sq:
                best_sq, best_c = float(sup[i]), ep_c[i].copy()
            if p["kick"] > 0:
                carry_c, carry = ep_c[n_ep - 1].copy(), True
        best_score, best_solution = None, None
        if best_c is not None:
            best_solution = {"coefficients": [int(v) for v in best_c]}
            best_score = exact.sup_score(best_c)
        start_score = exact.sup_score(c0) if use_c0 else None
        return AttemptBatch(scores, best_score, best_solution, time.time() - t0, {"steps": steps, "params": p, "started_from_known": len(scores) if use_c0 else 0, "improved_on_start": int(sum(v < start_level - 1e-6 for v in scores)) if use_c0 else 0}, start_score)

    def exact_score(self, solution: dict) -> float:
        return exact.grid_score(solution["coefficients"])

    def canonical(self, solution: dict) -> str:
        return "".join("+" if v > 0 else "-" for v in exact.canonical(solution["coefficients"]))

    def audit(self, solution: dict) -> dict:
        c = solution["coefficients"]
        return {
            "official_grid_score": exact.grid_score(c),
            "continuous_sup_score": exact.sup_score(c),
            "continuous_sup_upper_bound": exact.sup_upper_bound(c),
            "certified_below_record": bool(exact.sup_upper_bound(c) < self.record()),
            "note": "the official verifier samples 1e6 points; we also require the certified continuous sup-norm to beat the record",
        }

    def brief(self) -> str:
        return (
            "Minimise max_{|z|=1} |p(z)| / sqrt(71) over polynomials of degree 69 with +-1 coefficients. "
            "The public record (1.25946) is isolated: exhaustive Hamming-ball searches around earlier leaders found nothing, "
            "algebraic families (Legendre, cyclotomic, skew-symmetric) top out near 1.28-1.44, and random-start hill climbing "
            "lands on a plateau near 1.35-1.45. Symmetries: negation, reversal, c_k -> (-1)^k c_k."
        )
