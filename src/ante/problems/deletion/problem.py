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
                    Knob("greedy_k", "int", 1, 1, 8, doc="refill after an ordinary move: best of this many sampled free words (fewest new blocks)"),
                    Knob("ruin_greedy_k", "int", 2, 1, 16, doc="refill after a ruin: best of this many sampled free words"),
                ],
            ),
            "window": KernelSpec(
                "window",
                "Exact large-neighbourhood search: release a cluster of codewords, collect every word compatible with the "
                "rest, and solve the refill to optimality as an integer program (HiGHS). One attempt = one window; its "
                "outcome is the code size after the window (never smaller than the start).",
                [
                    Knob("start", "choice", "record", choices=["record", "public", "elite"], doc="code to improve"),
                    Knob("geometry", "choice", "cluster", choices=["cluster", "band", "difference", "splice"], doc=(
                        "cluster: release the codewords nearest a random centre. band: release every codeword whose Hamming weight lies in a random run of "
                        "`band_width` consecutive weights. difference: pick a second good code (a random public top-5 code or one of ours, random symmetry "
                        "image), keep only the words both codes share, and re-solve everything else exactly, with every compatible word of the whole space "
                        "as a candidate (searches between two basins). splice: keep our code's words of weight <= k and the second code's words of weight "
                        ">= k+3 for a random k, and re-solve the two middle weight layers exactly (the two kept parts can never conflict)")),
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
            return symmetry_images(words_to_ints(self.public_solutions()[0]["data"]["words"]))[int(rng.integers(4))]
        if start == "public":
            pub = self.public_solutions()
            code = words_to_ints(pub[int(rng.integers(len(pub)))]["data"]["words"])
            return symmetry_images(code)[int(rng.integers(4))]
        # our own codes: uniformly among those within one word of our best, so that a plateau of equal-size codes is explored broadly
        top = len(elites[0]["words"])
        pool = [e for e in elites if len(e["words"]) >= top - 1]
        return words_to_ints(pool[int(rng.integers(len(pool)))]["words"])

    def _public_images(self) -> list[set]:
        from .graph import symmetry_images, words_to_ints

        return [set(np.asarray(img).tolist()) for s in self.public_solutions() for img in symmetry_images(words_to_ints(s["data"]["words"]))]

    @staticmethod
    def _distinct(code, refs, min_new_frac: float = 0.1) -> bool:
        """True when at least 10% of the code's words are absent from every public code (any symmetry image)."""
        c = set(np.asarray(code).tolist())
        return all(len(c - r) >= min_new_frac * len(c) for r in refs)

    def _donor(self, rng, elites, code):
        """A second good code, different from `code`: a public top-5 code or one of ours, in a random symmetry image."""
        from .graph import symmetry_images, words_to_ints

        pool = [s["data"]["words"] for s in self.public_solutions()] + [e["words"] for e in (elites or [])]
        mine = set(np.asarray(code).tolist())
        for _ in range(20):
            img = symmetry_images(words_to_ints(pool[int(rng.integers(len(pool)))]))[int(rng.integers(4))]
            if len(mine & set(np.asarray(img).tolist())) < len(mine):
                return np.asarray(img, dtype=np.int64)
        return None

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
        started = improved = top_started = top_improved = 0
        top_bar = sorted((float(x["score"]) for x in self.public_solutions()), reverse=True)[2]  # third-best public size
        ladder: dict[str, list[int]] = {}
        meta: dict = {"params": p}
        refs = self._public_images()
        if kernel_name == "ils":
            p["ruin_max"] = max(p["ruin_max"], p["ruin_min"])
            kw = {k: p[k] for k in ("tmax", "kappa", "accept", "max_drop", "temp", "p_force", "p_ruin", "ruin_min", "ruin_max", "ruin_mode", "stall", "greedy_k", "ruin_greedy_k")}
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
                if x0 is not None:
                    started += 1
                    improved += r["best"] > len(x0)
                    rung = ladder.setdefault(str(len(x0)), [0, 0])
                    rung[0] += 1
                    rung[1] += int(r["best"] > len(x0))
                    if len(x0) >= top_bar:
                        # climbing = a larger code that is not a near-copy of a public one
                        top_started += 1
                        top_improved += int(r["best"] > len(x0) and self._distinct(s.best_ints(), refs))
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
                tl = min(p["time_limit"], left - 1.0)
                if p["geometry"] in ("difference", "splice"):
                    donor = self._donor(rng, elites, code)
                    if donor is None:
                        break
                    fixed = W.difference_fixed(code, donor) if p["geometry"] == "difference" else W.splice_fixed(t, code, donor, int(rng.integers(2, 12)))
                    out = W.solve_fixed(t, fixed, len(code), time_limit=tl, max_candidates=p["max_candidates"])
                else:
                    if p["geometry"] == "band":
                        lo = int(rng.integers(0, 17 - p["band_width"] + 1))
                        rel = W.band_window(t, code, lo, lo + p["band_width"] - 1)
                        if rel.shape[0] == 0:
                            continue
                    else:
                        centre = int(rng.choice(code)) if p["centre"] == "codeword" else int(rng.integers(t.n_words))
                        rel = W.cluster_window(t, code, centre, p["size"], rng=rng, pool_factor=p["pool_factor"])
                    out = W.solve_window(t, code, rel, time_limit=tl, max_candidates=p["max_candidates"])
                skipped += out["status"] == "skipped"
                timed_out += out["status"] == "limit"
                if skipped > 200 and not scores:
                    break  # every window is too large for max_candidates: report nothing rather than spin
                if out["status"] == "skipped":
                    continue
                proven += bool(out.get("proven_no_gain"))
                started += 1
                top_started += int(len(code) >= top_bar)
                if out.get("gain", 0) > 0:
                    improved += 1
                    top_improved += int(len(code) >= top_bar and self._distinct(out["new_code"], refs))
                    code = np.asarray(out["new_code"], dtype=np.int64)
                scores.append(float(len(code)))
            best = code
            meta["windows_proven_no_gain"] = proven
            meta["windows_skipped_too_large"] = int(skipped)
            meta["windows_hit_time_limit"] = int(timed_out)
        meta["started_from_known"], meta["improved_on_start"] = int(started), int(improved)
        meta["top_started"], meta["top_improved"] = int(top_started), int(top_improved)
        if ladder:
            meta["ladder"] = ladder
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
            "is unrelated. From the empty code, one minute of iterated local search reaches 211. Exact solves show the public 226 code "
            "cannot be improved inside any single weight layer, any two adjacent layers, or tens of thousands of 40-word clusters, so a "
            "227 needs a coordinated change of 30+ words, a temporary descent below 226, or a splice of two different codes. The last "
            "leaderboard step is reproducible: ILS chains that start from the public 225 code with Metropolis acceptance (temp 1.0-1.5), "
            "max_drop 8, tmax 5-8, weight-band ruins of 8-40 words and p_ruin 0.04-0.08 reached 226-word codes in about 1 of 4 chains of "
            "40-60 s, but every one of them lies within 2 to 7 words of the public 226 code: the walk rediscovers the record holder's "
            "code, it does not find a new one. Climbing now counts only results that are not near-copies (at least 10% of the words "
            "absent from every public code). Nothing has yet improved a 226 start. Chains of 300-900 s are untested."
        )
