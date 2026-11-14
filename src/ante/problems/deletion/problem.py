from __future__ import annotations

import time

import numpy as np

from ante.core.problem import AttemptBatch, KernelSpec, Knob, Problem

from . import exact


class DeletionCode(Problem):
    slug = "two-deletion-code-16"
    title = "Two-deletion-correcting code (length 16)"
    maximize = True
    integer_scores = True

    def kernels(self) -> dict[str, KernelSpec]:
        start = Knob("start", "choice", "scratch", choices=["scratch", "record", "public", "elite"], doc="start from the empty code, the public record (226), a random public top-5 code (random symmetry image), or one of our own best codes")
        return {
            "ils": KernelSpec(
                "ils",
                "Iterated local search on the conflict graph (ARW style): plateau (1,1)-swaps, forced insertions that evict "
                "up to `tmax` blockers, ruin-and-recreate, each followed by greedy refill and (1,2)-swap local search. "
                "One attempt = one chain of `chain_seconds`; its outcome is the largest code the chain reached.",
                [
                    start,
                    Knob("chain_seconds", "float", 60.0, 5.0, 3600.0, doc="length of one chain (one attempt)"),
                    Knob("tmax", "int", 3, 2, 8, doc="forced insertion may evict up to this many codewords"),
                    Knob("kappa", "int", 3, 1, 10, doc="candidates sampled per move; the one longest outside the code is taken"),
                    Knob("accept", "choice", "arw", choices=["strict", "arw", "metropolis"], doc="rule for accepting a smaller code"),
                    Knob("max_drop", "int", 2, 0, 12, doc="never accept a code more than this many words below the chain's best"),
                    Knob("temp", "float", 0.5, 0.05, 5.0, doc="temperature for accept=metropolis (in codewords)"),
                    Knob("p_force", "float", 0.25, 0.0, 1.0, doc="probability of a forced-insertion move"),
                    Knob("p_ruin", "float", 0.02, 0.0, 0.5, doc="probability of a ruin-and-recreate move"),
                    Knob("ruin_min", "int", 4, 1, 60, doc="smallest ruin size (codewords removed)"),
                    Knob("ruin_max", "int", 16, 2, 120, doc="largest ruin size"),
                    Knob("ruin_mode", "choice", "neighbourhood", choices=["neighbourhood", "band"], doc="ruin a cluster around a random word, or a whole weight band"),
                    Knob("stall", "int", 50_000, 0, 5_000_000, doc="idle iterations before returning to the chain's best (0 = never)"),
                ],
            ),
            "window": KernelSpec(
                "window",
                "Exact large-neighbourhood search: release a cluster of codewords, collect every word compatible with the "
                "rest, and solve the refill to optimality as an integer program (HiGHS). One attempt = one window; its "
                "outcome is the code size after the window (never smaller than the start).",
                [
                    Knob("start", "choice", "record", choices=["record", "public", "elite"], doc="code to improve"),
                    Knob("geometry", "choice", "cluster", choices=["cluster", "band"], doc="cluster: release the codewords nearest a random centre; band: release every codeword whose Hamming weight lies in a random run of `band_width` consecutive weights (conflicts only join weights at most 2 apart)"),
                    Knob("band_width", "int", 2, 1, 5, doc="geometry=band: number of consecutive weights released"),
                    Knob("size", "int", 16, 4, 120, doc="geometry=cluster: codewords released per window"),
                    Knob("pool_factor", "float", 1.5, 1.0, 4.0, doc="the window is sampled from the pool_factor*size codewords closest to a random centre (1.0 = the closest ones exactly)"),
                    Knob("centre", "choice", "codeword", choices=["codeword", "any"], doc="window centre: a random codeword, or a random word of the whole space"),
                    Knob("time_limit", "float", 20.0, 2.0, 1500.0, doc="solver time limit per window, seconds"),
                    Knob("max_candidates", "int", 6000, 500, 40000, doc="skip windows whose refill has more candidate words than this"),
                ],
            ),
        }

    def _pick_start(self, start: str, rng, elites, tables):
        from .graph import symmetry_images, words_to_ints

        if start == "elite" and not elites:
            start = "public"
        if start == "scratch":
            return None
        if start == "record":
            return words_to_ints(self.public_solutions()[0]["data"]["words"])
        if start == "public":
            pub = self.public_solutions()
            code = words_to_ints(pub[int(rng.integers(len(pub)))]["data"]["words"])
            return symmetry_images(code)[int(rng.integers(4))]
        return words_to_ints(elites[min(int(rng.exponential(1.5)), len(elites) - 1)]["words"])

    def run(self, kernel_name: str, params: dict, seed: int, seconds: float, elites: list[dict] | None = None) -> AttemptBatch:
        from . import search as S
        from . import window as W
        from .graph import ints_to_words, load_tables

        p = self.kernels()[kernel_name].resolve(params)
        rng = np.random.default_rng(seed)
        t = load_tables()
        t0 = time.time()
        scores: list[float] = []
        best = None
        start_score = None
        meta: dict = {"params": p}
        if kernel_name == "ils":
            p["ruin_max"] = max(p["ruin_max"], p["ruin_min"])
            kw = {k: p[k] for k in ("tmax", "kappa", "accept", "max_drop", "temp", "p_force", "p_ruin", "ruin_min", "ruin_max", "ruin_mode", "stall")}
            iters = 0
            while True:
                left = seconds - (time.time() - t0)
                if left < 2.0 or (scores and left < 0.5 * p["chain_seconds"]):
                    break
                s = S.Searcher(t, seed=int(rng.integers(1 << 31)), **kw)
                x0 = self._pick_start(p["start"], rng, elites, t)
                if x0 is not None:
                    start_score = max(start_score or 0.0, float(len(x0)))
                s.start(x0)
                r = s.run(min(p["chain_seconds"], left), trace_every=1e9)
                iters += r["iterations"]
                scores.append(float(r["best"]))
                if best is None or r["best"] > len(best):
                    best = s.best_ints()
            meta["iterations"] = iters
        else:
            code = np.asarray(self._pick_start(p["start"], rng, elites, t), dtype=np.int64)
            start_score = float(len(code))
            proven = skipped = timed_out = 0
            while True:
                left = seconds - (time.time() - t0)
                if left < 3.0:
                    break
                if p["geometry"] == "band":
                    lo = int(rng.integers(0, 17 - p["band_width"] + 1))
                    rel = W.band_window(t, code, lo, lo + p["band_width"] - 1)
                    if rel.shape[0] == 0:
                        continue
                else:
                    centre = int(rng.choice(code)) if p["centre"] == "codeword" else int(rng.integers(t.n_words))
                    rel = W.cluster_window(t, code, centre, p["size"], rng=rng, pool_factor=p["pool_factor"])
                out = W.solve_window(t, code, rel, time_limit=min(p["time_limit"], left - 1.0), max_candidates=p["max_candidates"])
                skipped += out["status"] == "skipped"
                timed_out += out["status"] == "limit"
                if skipped > 200 and not scores:
                    break  # every window is too large for max_candidates: report nothing rather than spin
                if out["status"] == "skipped":
                    continue
                proven += bool(out.get("proven_no_gain"))
                if out.get("gain", 0) > 0:
                    code = np.asarray(out["new_code"], dtype=np.int64)
                scores.append(float(len(code)))
            best = code
            meta["windows_proven_no_gain"] = proven
            meta["windows_skipped_too_large"] = int(skipped)
            meta["windows_hit_time_limit"] = int(timed_out)
        best_solution = best_score = None
        if best is not None:
            words = ints_to_words(np.sort(np.asarray(best)), t.n)
            best_solution = {"words": words}
            best_score = float(exact.verify(words))
        return AttemptBatch(scores, best_score, best_solution, time.time() - t0, meta, start_score)

    def exact_score(self, solution: dict) -> float:
        return float(exact.verify(solution["words"]))

    def canonical(self, solution: dict) -> str:
        from .graph import symmetry_images, words_to_ints

        imgs = symmetry_images(words_to_ints(solution["words"]))
        return min(",".join(str(int(v)) for v in np.sort(np.asarray(im))) for im in imgs)

    def audit(self, solution: dict) -> dict:
        words = solution["words"]
        desc = [exact.descendants(w) for w in words]
        return {
            "words": len(words),
            "distinct_words": len(set(words)),
            "descendants_used": sum(len(d) for d in desc),
            "distinct_descendants_used": len(set().union(*desc)),
            "note": "valid iff the two descendant counts are equal (pairwise disjoint descendant sets); all checks are exact string operations",
        }

    def brief(self) -> str:
        return (
            "Find the largest set of 16-bit words whose two-deletion descendant sets are pairwise disjoint (a maximum independent "
            "set in a conflict graph with 65,536 vertices and 81 million edges). Record 226; it moved 216 -> 226 in six weeks. "
            "Conflicts only join words whose Hamming weights differ by at most 2. The public 226 code has no free word and no "
            "improving 1-for-2 or 2-for-3 exchange; other agents proved that around the older 223 codes no exchange of up to 7 "
            "words helps. The 225 code shares 188 words with the 226 code; the 224 code shares 138 after complementing; the 223 "
            "is unrelated. From the empty code, one minute of iterated local search reaches 211."
        )
