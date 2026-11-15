"""The research loop.

Each cycle:
  1. read the live leaderboards (records move);
  2. the PI proposes strategies, each with a hypothesis and its own forecast;
  3. every new strategy runs a small pilot;
  4. every strategy is priced from its attempt outcomes (dollars to record);
  5. the allocator funds the best chances per dollar, and files a forecast *before* each run;
  6. the referee scores every forecast; surprises go back to the PI, which writes lessons;
  7. any new best is re-scored by the independent checker; anything that beats a record goes
     through the full verification gauntlet and is held for a human decision to submit.
"""

from __future__ import annotations

import json
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ante import problems as registry

from . import arena, referee, runner as runner_mod, verify
from .ledger import Ledger
from .market import Market, ProblemState, Strategy, solution_sha

SEEDS = {
    "kakeya-needle-128": [
        ("hop", {"start": "layout", "chain_seconds": 20, "donors": "none"}, "Baseline from scratch: sample a new group layout (16 small groups in 4-5 large ones), polish, and hop. No public material is used, so anything found is a new arrangement. Refuted as a route to the record if the best of many chains stays above 0.1068."),
        ("hop", {"start": "public", "chain_seconds": 20, "temperature": 3e-6}, "The public top-5 are distinct layouts; hopping from them with a temperature above the 1.2e-6 gap to neighbouring minima may cross into a better layout."),
        ("hop", {"start": "elite", "chain_seconds": 20, "w_group_crossover": 4.0, "w_crossover": 0.5, "temperature": 1e-5, "patience": 300}, "Recombination at group boundaries: transplant whole groups between our best arrangements and the public ones. If good layouts share good groups, this should improve our best faster than chains from scratch."),
    ],
    "two-deletion-code-16": [
        ("ils", {"start": "scratch", "chain_seconds": 20}, "Baseline: iterated local search from the empty code. Tests how far independent chains get without public codes."),
        ("ils", {"start": "public", "chain_seconds": 60, "tmax": 8, "accept": "metropolis", "temp": 1.5, "max_drop": 8, "p_force": 0.3, "p_ruin": 0.08, "ruin_min": 12, "ruin_max": 40, "ruin_mode": "band", "stall": 0},
         "Reproduce the last leaderboard step: non-resetting Metropolis walks with weight-band ruins turned the public 225 code into new 226 codes in earlier runs. If the same walk can improve a 226 it gives 227; the ladder (improved/started per starting size) measures how close it is."),
        ("window", {"start": "record", "geometry": "splice", "time_limit": 30}, "Exact splice: keep the record's light words and another good code's heavy words, and solve the two middle weight layers exactly with every compatible word as a candidate. Small exchanges and single or adjacent layers are proven not to help; this tests neighbourhoods that mix two codes."),
        ("window", {"start": "record", "geometry": "difference", "time_limit": 30, "max_candidates": 12000}, "Exact difference window: keep only the words the record shares with a second good code and re-solve everything else exactly. Searches the region between two basins."),
    ],
    "flat-polynomials": [
        ("tabu", {"objective": "excess", "restart_after": 3000}, "Baseline: tabu search on the squared excess over a moving target level, from random sequences."),
    ],
}


@dataclass
class Config:
    run_id: str
    problems: list[str] = field(default_factory=lambda: list(SEEDS))
    compute_budget_usd: float = 2.0
    llm_budget_usd: float = 1.0
    max_cycles: int = 3
    max_wall_seconds: float = 3600.0
    pilot_usd: float = 0.05
    pilot_seconds: float = 60.0
    tranche_usd: float = 0.5
    fund_k: int = 3
    scale_seconds: float = 240.0
    proposals_per_problem: int = 2
    max_card_seconds: float = 1800.0
    min_pilot_workers: int = 8
    pi_model: str = "gpt-6.1-terra"
    pi_effort: str = "medium"
    use_pi: bool = True
    sandbox_verify: bool = True
    live_arena: bool = True
    seed: int = 0
    note: str = ""
    carry_from: list[str] = field(default_factory=list)  # earlier run ids whose lessons this run starts with
    app_id: str = ""  # Modal app that ran this (for reconciling estimated cost with the bill)
    allocator: str = "priced"  # "priced" (Thompson sampling on prices) or "uniform" (ablation: fund k live strategies at random)


class Loop:
    def __init__(self, cfg: Config, runner, out_dir: str | Path, llm=None, on_write=None):
        self.cfg, self.runner = cfg, runner
        self.dir = Path(out_dir)
        (self.dir / "candidates").mkdir(parents=True, exist_ok=True)
        (self.dir / "attempts").mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.dir / "ledger.jsonl", on_write=on_write)
        self.market = Market(seed=cfg.seed)
        self.pi = None
        if cfg.use_pi:
            from .llm import LLM
            from .pi import PI

            self.llm = llm or LLM(self.ledger, budget_usd=cfg.llm_budget_usd)
            self.llm.ledger = self.ledger
            self.pi = PI(self.llm, self.ledger, model=cfg.pi_model, effort=cfg.pi_effort)
        self.spent = 0.0
        self.n_cards = 0
        self.cycle = 0
        self.pending_reflection: dict[str, list[dict]] = {}
        self.carried: list[Strategy] = []
        self.carried_solutions: list[tuple[str, str, dict]] = []  # (run id, problem, solution) from earlier runs
        self.t0 = time.time()

    # ---- helpers ----------------------------------------------------------------------------
    def budget_left(self) -> float:
        return self.cfg.compute_budget_usd - self.spent

    def llm_spent(self) -> float:
        return self.pi.llm.spent_usd if self.pi else 0.0

    def _snapshot(self) -> None:
        for slug in self.cfg.problems:
            p = registry.get(slug)
            meta = p.meta()
            rows = arena.leaderboard(meta["id"]) if self.cfg.live_arena else None
            source = "live" if rows else "snapshot"
            rows = rows or p.leaderboard()
            record = float(rows[0]["bestScore"])
            target = p.target(record)
            st = self.market.problems.get(slug)
            if st is None:
                st = self.market.problems[slug] = ProblemState(slug, p.maximize, record, target, integer=p.integer_scores)
            elif record != st.record:
                self.ledger.append("note", text=f"The {slug} record moved from {st.record!r} to {record!r} (now held by {rows[0]['agentName']}).")
            st.record, st.target = record, target
            self.ledger.append(
                "arena_snapshot", problem=slug, title=p.title, maximize=p.maximize, record=record, target=target, leader=rows[0]["agentName"],
                source=source, top=[{"agentName": r["agentName"], "bestScore": r["bestScore"]} for r in rows[:5]],
            )

    def _seed_strategies(self) -> list[Strategy]:
        new = []
        for slug in self.cfg.problems:
            p = registry.get(slug)
            for kernel, params, hyp in SEEDS.get(slug, []):
                resolved = p.kernels()[kernel].resolve(params)
                if self.market.is_duplicate(slug, kernel, resolved):
                    continue  # already present, carried over from an earlier run
                s = Strategy(self.market.new_sid(), slug, kernel, resolved, hyp, rationale="built-in baseline", source="seed", proposer="seed")
                self.market.add(s)
                self.ledger.append("strategy", **s.public())
                new.append(s)
        return new

    def _carry_over(self) -> None:
        """Start with what earlier runs learned: their lessons, and every strategy that improved on a known solution."""
        for run_id in self.cfg.carry_from:
            path = self.dir.parent / run_id / "ledger.jsonl"
            if not path.exists():
                self.ledger.append("note", text=f"Could not carry lessons over from run {run_id}: no ledger found.")
                continue
            events = list(Ledger(path).events())
            strategies = {e["sid"]: e for e in events if e["type"] == "strategy"}
            n = 0
            for e in events:
                if e["type"] == "lesson" and e.get("problem") in self.cfg.problems and self.pi:
                    self.pi.lessons.setdefault(e["problem"], []).append(f"[run {run_id}] {e['text']}")
                    n += 1
                if e["type"] == "candidate" and e.get("problem") in self.cfg.problems and "public" in str(e.get("lineage", "")):
                    st = strategies.get(e["sid"])
                    if not st or st.get("source") != "pi":
                        continue
                    p = registry.get(e["problem"])
                    params = p.kernels()[st["kernel"]].resolve(st["params"])
                    if self.market.is_duplicate(e["problem"], st["kernel"], params):
                        continue
                    carried = Strategy(self.market.new_sid(), e["problem"], st["kernel"], params, st["hypothesis"],
                                       rationale=f"carried over from run {run_id}, where it turned a known solution into a new one scoring {e['score']!r}",
                                       source="carried", proposer=st.get("proposer", "?"), prior=st.get("prior") or {})
                    self.market.add(carried)
                    self.carried.append(carried)
                    self.ledger.append("strategy", **carried.public())
                    n += 1
            for f in sorted((self.dir.parent / run_id / "candidates").glob("*.json")):
                c = json.loads(f.read_text())
                if c.get("problem") in self.cfg.problems:
                    self.carried_solutions.append((run_id, c["problem"], c["solution"]))
            self.ledger.append("note", text=f"Carried {n} lessons and proven strategies over from run {run_id}.")

    def _adopt_carried_solutions(self) -> None:
        """Continue from the best solutions of earlier runs. Each is re-scored by the independent checker first."""
        for run_id, slug, solution in self.carried_solutions:
            p, st = registry.get(slug), self.market.problems[slug]
            try:
                score = float(p.exact_score(solution))
            except Exception as e:
                self.ledger.append("note", text=f"A solution carried from run {run_id} for {slug} failed the independent checker ({e}); ignored.")
                continue
            if p.is_public(solution) is not None:
                continue
            st.add_elite(score, p.canonical(solution), solution)
            if st.better(score, st.best):
                sha = solution_sha(solution)
                st.best, st.best_sha = score, sha
                path = self.dir / "candidates" / f"{slug}-{sha}.json"
                path.write_text(json.dumps({"problem": slug, "score": score, "sid": "carried", "cid": f"run {run_id}", "seed": None, "solution": solution}))
                self.ledger.append("candidate", problem=slug, score=score, cid=f"run {run_id}", sid="carried", solution_sha=sha,
                                   path=str(path.relative_to(self.dir)), lineage=f"carried over from run {run_id}")
        self.carried_solutions = []

    # ---- cards ------------------------------------------------------------------------------
    def _plan_card(self, s: Strategy, stage: str, budget_usd: float) -> dict:
        chain = float(s.params.get("chain_seconds", 0.0) or 0.0)
        base = self.cfg.pilot_seconds if stage == "pilot" else self.cfg.scale_seconds
        seconds = max(base, 1.05 * chain + 5.0, 3.0 * float(s.params.get("time_limit", 0.0) or 0.0))
        seconds = min(seconds, self.cfg.max_card_seconds)
        workers = runner_mod.workers_for(budget_usd, seconds, self.runner.max_workers)
        if stage == "pilot":
            # a pilot needs several independent seeds to be priced, even if that overshoots its nominal budget
            workers = max(workers, min(self.cfg.min_pilot_workers, self.runner.max_workers))
        workers = max(self.runner.cores_per_container, workers - workers % self.runner.cores_per_container)  # fill whole containers
        est = workers * (seconds + runner_mod.CONTAINER_OVERHEAD_S) * runner_mod.USD_PER_CORE_S
        self.n_cards += 1
        cid = f"c{self.n_cards:04d}"
        forecast = self.market.forecast(s, est)
        if forecast["source"] == "prior" and forecast["p_record"] is not None:
            # the proposer's forecast was "per $1 of compute"; convert to this card's budget
            forecast["p_record"] = 1.0 - (1.0 - min(forecast["p_record"], 1 - 1e-12)) ** est
        card = {"cid": cid, "sid": s.sid, "problem": s.problem, "stage": stage, "workers": workers, "seconds": seconds, "budget_usd": est, "forecast": forecast}
        self.ledger.append("card", **card)
        return card

    def _run_card(self, card: dict) -> list[dict]:
        s = self.market.strategies[card["sid"]]
        st = self.market.problems[s.problem]
        base = zlib.crc32(f"{self.cfg.run_id}/{self.cfg.seed}/{card['cid']}".encode())
        jobs = [
            {"problem": s.problem, "kernel": s.kernel, "params": s.params, "seed": (base + i) % (1 << 31), "seconds": card["seconds"], "elites": st.elite_solutions()}
            for i in range(card["workers"])
        ]
        return self.runner.run(jobs)

    def _settle(self, card: dict, results: list[dict]) -> None:
        """Book a finished card: evidence, cost, candidates, price, referee."""
        s = self.market.strategies[card["sid"]]
        st = self.market.problems[s.problem]
        p = registry.get(s.problem)
        cost, core_s = runner_mod.card_cost(results, self.runner.cores_per_container)
        self.spent += cost
        scores = [x for r in results for x in r["scores"]]
        errors = [r.get("error") for r in results if not r.get("ok")]
        s.batches.append(scores)
        s.record_hits += sum(1 for x in scores if st.beats_record(x))
        for r in results:
            if r.get("start_score") is not None and (s.start_score is None or st.better(r["start_score"], s.start_score)):
                s.start_score = float(r["start_score"])
            s.started += int(r.get("meta", {}).get("started_from_known", 0) or 0)
            s.improved += int(r.get("meta", {}).get("improved_on_start", 0) or 0)
            s.top_started += int(r.get("meta", {}).get("top_started", 0) or 0)
            s.top_improved += int(r.get("meta", {}).get("top_improved", 0) or 0)
            for level, (n_started, n_improved) in (r.get("meta", {}).get("ladder") or {}).items():
                rung = s.ladder.setdefault(level, [0, 0])
                rung[0] += int(n_started)
                rung[1] += int(n_improved)
        s.cost_usd += cost
        s.core_seconds += core_s
        s.cards += 1
        s.funded += card["stage"] == "scale"
        (self.dir / "attempts" / f"{card['cid']}.json").write_text(json.dumps({"cid": card["cid"], "sid": s.sid, "scores": scores}))

        # the workers' own claims are not trusted: re-score the best candidate with the independent checker
        claims = [r for r in results if r.get("best_solution") is not None and r.get("best_score") is not None]
        claims.sort(key=lambda r: -r["best_score"] if st.maximize else r["best_score"])
        card_best, new_best, beat_record = None, False, False
        for r in claims[:3]:
            try:
                score = float(p.exact_score(r["best_solution"]))
            except Exception as e:
                self.ledger.append("note", text=f"Card {card['cid']}: a worker's best solution failed the independent checker ({type(e).__name__}: {e}); discarded.")
                continue
            if card_best is None or st.better(score, card_best):
                card_best = score
            started = r.get("start_score")
            if started is not None and not st.better(score, started):
                continue  # it only returned what it started from: nothing new
            if p.is_public(r["best_solution"]) is not None:
                continue  # a copy of a public solution is not ours
            if s.best is None or st.better(score, s.best):
                s.best = score
            st.add_elite(score, p.canonical(r["best_solution"]), r["best_solution"])
            if st.better(score, st.best):
                st.best, new_best = score, True
                sha = solution_sha(r["best_solution"])
                st.best_sha = sha
                path = self.dir / "candidates" / f"{s.problem}-{sha}.json"
                path.write_text(json.dumps({"problem": s.problem, "score": score, "sid": s.sid, "cid": card["cid"], "seed": r.get("seed"), "solution": r["best_solution"]}))
                self.ledger.append("candidate", problem=s.problem, score=score, cid=card["cid"], sid=s.sid, solution_sha=sha, path=str(path.relative_to(self.dir)),
                                   lineage=p.lineage(s.kernel, s.params))
                if st.beats_record(score):
                    beat_record = True
                    v = verify.gauntlet(p, r["best_solution"], self.ledger, sha, sandbox=self.cfg.sandbox_verify)
                    self.ledger.append("note", text=(
                        f"RECORD CANDIDATE on {s.problem}: {score!r} vs record {st.record!r} (target {st.target!r}). "
                        f"Verification {'PASSED' if v['ok'] else 'FAILED'}: {v['checks']}. Held for a human decision; nothing was submitted."))
            break

        import numpy as np

        q = np.quantile(scores, [0.99, 0.9, 0.5] if st.maximize else [0.01, 0.1, 0.5]).tolist() if scores else [None, None, None]
        self.ledger.append(
            "result", cid=card["cid"], sid=s.sid, problem=s.problem, attempts=len(scores), best=card_best, quantiles={"q01": q[0], "q10": q[1], "q50": q[2]},
            cost_usd=cost, wall_seconds=max((r["seconds"] for r in results), default=0.0), core_seconds=core_s, new_best=new_best, beat_record=beat_record,
            error=(f"{len(errors)}/{len(results)} workers failed; first: {errors[0]}" if errors else None),
        )
        if scores:
            self.ledger.append("price", **self.market.price_event(s))
        elif s.n_attempts() == 0:
            s.empty_cards = getattr(s, "empty_cards", 0) + 1
            if s.empty_cards >= 2:
                s.retired = True
            self.ledger.append("note", text=(
                f"Card {card['cid']} for {s.sid} ({s.problem}) produced no attempt outcomes"
                + (f" ({errors[0]})" if errors else " (its attempts did not finish within the card, or every window was skipped)")
                + ("; second time, so the strategy is retired." if s.retired else "; it will get one more pilot.")))
        outcome = {"new_best": new_best, "beat_record": beat_record, "best": card_best}
        verdict = referee.judge(card["forecast"], outcome, st.maximize)
        self.ledger.append("referee", cid=card["cid"], sid=s.sid, forecast=card["forecast"], outcome=outcome, **verdict)
        f = card["forecast"]
        if f.get("p_new_best") is not None:
            self.market.calibration.update(f["source"], "new_best", f["p_new_best"], new_best)
        if f.get("p_record") is not None:
            self.market.calibration.update(f["source"], "record", f["p_record"], beat_record)
        self.pending_reflection.setdefault(s.problem, []).append(
            {"cid": card["cid"], "sid": s.sid, "stage": card["stage"], "cost_usd": cost, "forecast": f, "outcome": outcome, "surprise": verdict["surprise"]})

    def _run_cards(self, plans: list[tuple[Strategy, str, float]]) -> None:
        cards = [self._plan_card(s, stage, usd) for s, stage, usd in plans]
        if not cards:
            return
        with ThreadPoolExecutor(max_workers=len(cards)) as ex:
            futures = [(c, ex.submit(self._run_card, c)) for c in cards]
            for c, fut in futures:
                try:
                    results = fut.result()
                except Exception as e:
                    results = [{"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "scores": [], "best_score": None, "best_solution": None, "seconds": 0.0, "meta": {}}]
                self._settle(c, results)

    # ---- the loop ---------------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        self.ledger.append("run_start", run_id=cfg.run_id, budget_usd={"compute": cfg.compute_budget_usd, "llm": cfg.llm_budget_usd}, problems=cfg.problems,
                           config={**asdict(cfg), "runner": self.runner.name})
        self._carry_over()
        reason = "max cycles reached"
        failures = 0
        for self.cycle in range(cfg.max_cycles):
            try:
                self._cycle()
                failures = 0
            except Exception as e:  # an unattended run must survive one bad cycle
                failures += 1
                self.ledger.append("note", text=f"Cycle {self.cycle} failed with {type(e).__name__}: {e}. " + traceback.format_exc()[-600:])
                if failures >= 3:
                    reason = "three consecutive cycles failed"
                    break
                time.sleep(20)
            if self.budget_left() < max(cfg.pilot_usd, 0.03 * cfg.compute_budget_usd):
                reason = "compute budget spent"
                break
            if time.time() - self.t0 > cfg.max_wall_seconds:
                reason = "wall-clock limit reached"
                break
        self._finish(reason)

    def _cycle(self) -> None:
        cfg = self.cfg
        self.ledger.append("cycle", n=self.cycle, phase="start", spent_compute_usd=self.spent, spent_llm_usd=self.llm_spent())
        self._snapshot()
        if self.carried_solutions:
            self._adopt_carried_solutions()

        # 2-3. new strategies and their pilots
        new = (self._seed_strategies() + self.carried) if self.cycle == 0 else []
        new += [s for s in self.market.live() if s.n_attempts() == 0 and getattr(s, "empty_cards", 0) == 1]  # one retry for an empty pilot
        if self.pi and self.cycle > 0:
            for slug in cfg.problems:
                new += self.pi.propose(self.market, registry.get(slug), slug, cfg.proposals_per_problem, cfg.pilot_usd)
        pilots = [(s, "pilot", cfg.pilot_usd) for s in new]
        if pilots and self.budget_left() > cfg.pilot_usd * len(pilots):
            self._run_cards(pilots)

        # 4. the PI rates every live strategy on the evidence (its rating is the strategy's opening price)
        if self.pi:
            for slug in cfg.problems:
                self.pi.rate(self.market, registry.get(slug), slug)

        # 4-5. price everything and fund the best chances per dollar
        tranche = min(cfg.tranche_usd, self.budget_left())
        if tranche > 0.2 * cfg.tranche_usd:
            table, chosen = self.market.allocate(tranche, cfg.fund_k, min_share_usd=cfg.pilot_usd, uniform=cfg.allocator == "uniform")
            names = ", ".join(f"{s.sid} ({s.problem})" for s, _ in chosen) or "nothing"
            self.ledger.append("allocation", cycle=self.cycle, budget_left_usd=self.budget_left(), tranche_usd=tranche, table=table, calibration=self.market.calibration.table(),
                               note=(f"Funded {names}: chosen at random with equal shares (uniform-allocation control arm)." if cfg.allocator == "uniform" else
                                     f"Funded {names}: highest sampled chance of a record, plus evidence of improving top known solutions; "
                                     "the tranche is split in proportion to the draws."))
            self._run_cards([(s, "scale", usd) for s, usd in chosen])

        # 6. reflection
        if self.pi:
            for slug, recent in list(self.pending_reflection.items()):
                self.pi.reflect(self.market, registry.get(slug), slug, recent)
            self.pending_reflection = {}

        hours = max((time.time() - self.t0) / 3600.0, 1e-9)
        self.ledger.append("cycle", n=self.cycle, phase="end", spent_compute_usd=self.spent, spent_llm_usd=self.llm_spent(),
                           usd_per_hour=round(self.spent / hours, 3), hours=round(hours, 3))

    def _finish(self, reason: str) -> None:
        cfg = self.cfg
        # 7. final verification of our best on every problem (independent + sandbox + audit + provenance)
        for slug, st in self.market.problems.items():
            if st.best_sha:
                cand = json.loads((self.dir / "candidates" / f"{slug}-{st.best_sha}.json").read_text())
                verify.gauntlet(registry.get(slug), cand["solution"], self.ledger, st.best_sha, sandbox=cfg.sandbox_verify)
        self.ledger.append("run_end", reason=reason, spent_compute_usd=self.spent, spent_llm_usd=self.llm_spent(),
                           best={slug: st.best for slug, st in self.market.problems.items()}, records={slug: st.record for slug, st in self.market.problems.items()})
