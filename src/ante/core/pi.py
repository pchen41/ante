"""The principal investigator: an LLM that proposes strategies, forecasts them, and learns from the referee.

The PI never touches compute directly. It can only (1) file strategies, each with a hypothesis and a
forecast, (2) retire strategies, and (3) record lessons. Everything it says is written to the ledger.
"""

from __future__ import annotations

import json

from .llm import LLM
from .market import Market, Strategy

SYSTEM = """You are the principal investigator of an autonomous research system that tries to beat best-known results on open mathematical optimisation problems.

How the system works:
- You do not run searches. You file STRATEGIES: a search kernel, parameter settings, and a falsifiable hypothesis about why this strategy will reach a better score than existing ones.
- Every new strategy first gets a small pilot. From the pilot's attempt outcomes the system estimates the strategy's PRICE: the expected dollars of compute until it beats the record (tail extrapolation for continuous scores, success counting for integer scores).
- A budget allocator then funds the strategies with the best chance of a record per dollar. Before each run a forecast is written down; afterwards a referee scores it.
- Your p_record for a strategy sets its OPENING PRICE (1 / p_record dollars to a record). The price then rises by $2 for every $1 the strategy spends without a record, and a tail fit can raise it further but never lower it. Strategies that have never beaten a record cannot be told apart by the data alone, so your ratings are what directs the money among them. You may re-rate any live strategy each cycle as evidence comes in.
- Your ratings are scored by the referee against what happens, and a rater whose promised records do not arrive is discounted, so be calibrated: most strategies never produce a record, and probabilities like 0.001 are legitimate answers.
- The strongest evidence short of a record is CLIMBING: attempts that started from a known solution and ended strictly better than it ("improved N of M starts"). A strategy that turns a 225 into a 226 is doing the thing that would turn a 226 into a 227.

What makes a good proposal:
- It tests a distinct mechanism (a different starting point, neighbourhood, acceptance rule, time scale), not a cosmetic tweak, unless the evidence shows a strategy is close and needs tuning.
- It uses the evidence: attempt-outcome quantiles, prices, lessons, and what other agents already ruled out.
- Its hypothesis says what result would refute it.
Use only kernels and parameters that are listed. Parameter values are given as strings."""

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kernel": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}}, "required": ["name", "value"], "additionalProperties": False}},
                    "hypothesis": {"type": "string", "description": "one or two sentences: the mechanism, and what would refute it"},
                    "rationale": {"type": "string", "description": "which evidence from the tables or lessons led to this"},
                    "parent": {"type": ["string", "null"], "description": "sid of the strategy this refines, or null"},
                    "p_record": {"type": "number", "description": "probability that $1 of compute on this strategy beats the record"},
                    "expected_best": {"type": "number", "description": "the best score you expect from the pilot"},
                },
                "required": ["kernel", "params", "hypothesis", "rationale", "parent", "p_record", "expected_best"],
                "additionalProperties": False,
            },
        },
        "retire": {"type": "array", "items": {"type": "string"}, "description": "sids of strategies that should get no more compute"},
        "ratings": {
            "type": "array",
            "description": "your current probability, for each live strategy you have a view on, that $1 more compute on it beats the record",
            "items": {"type": "object", "properties": {"sid": {"type": "string"}, "p_record": {"type": "number"}, "why": {"type": "string"}},
                      "required": ["sid", "p_record", "why"], "additionalProperties": False},
        },
    },
    "required": ["proposals", "retire", "ratings"],
    "additionalProperties": False,
}

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "one sentence a future proposer can act on, with the numbers that support it"},
                    "evidence": {"type": "array", "items": {"type": "string"}, "description": "card ids"},
                },
                "required": ["text", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lessons"],
    "additionalProperties": False,
}


def _fmt(x, digits=6) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


class PI:
    def __init__(self, llm: LLM, ledger, model: str = "gpt-6.1-terra", effort: str = "medium"):
        self.llm, self.ledger, self.model, self.effort = llm, ledger, model, effort
        self.lessons: dict[str, list[str]] = {}

    # ---- context ----------------------------------------------------------------------------
    def _strategy_table(self, market: Market, problem, slug: str) -> str:
        import numpy as np

        rows = []
        mine = [s for s in market.strategies.values() if s.problem == slug]
        retired = [s for s in mine if s.retired]
        if len(retired) > 8:  # keep the prompt bounded on long runs: old retired strategies collapse to one line
            rows.append("- retired earlier (no more compute): " + ", ".join(f"{s.sid} ({s.kernel}, best {_fmt(s.best, 9)})" for s in retired[:-8]))
        for s in mine:
            if s.retired and s in retired[:-8]:
                continue
            spec = problem.kernels()[s.kernel]
            defaults = spec.resolve({})
            diff = {k: v for k, v in s.params.items() if defaults.get(k) != v}
            att = s.attempts()
            if att:
                q = np.quantile(att, [0.99, 0.9, 0.5] if problem.maximize else [0.01, 0.1, 0.5])
                price = market.price_event(s)
                climb = f"improved on its start in {s.improved} of {s.started} attempts; " if s.started else ""
                if s.ladder:
                    rungs = ", ".join(f"from {k}: {v[1]}/{v[0]}" for k, v in sorted(s.ladder.items(), key=lambda kv: float(kv[0])))
                    climb += f"ladder (start score: improved/started) {rungs}; "
                ev = (climb + f"attempts={s.n_attempts()} top1%={_fmt(float(q[0]), 9)} top10%={_fmt(float(q[1]), 9)} median={_fmt(float(q[2]), 9)} "
                      f"best={_fmt(s.best, 12)} spent=${s.cost_usd:.3f} your_rating={_fmt(s.rating if s.rating is not None else s.prior.get('p_record'), 3)} price_usd_to_record={_fmt(price['usd_to_record'], 3) if price['usd_to_record'] is not None else 'unreachable'} "
                      f"P(attempt hits record)={_fmt(price['p_attempt'], 2)} model={price['model']}")
            else:
                ev = "no pilot yet"
            rows.append(f"- {s.sid}{' [retired]' if s.retired else ''} kernel={s.kernel} changed_params={json.dumps(diff)} funded={s.funded}x\n    hypothesis: {s.hypothesis}\n    evidence: {ev}")
        return "\n".join(rows) or "(none yet)"

    def _context(self, market: Market, problem, slug: str) -> str:
        ps = market.problems[slug]
        direction = "maximise" if ps.maximize else "minimise"
        kernels = "\n\n".join(k.describe() for k in problem.kernels().values())
        lessons = "\n".join(f"- {t}" for t in (self.lessons.get(slug, []) + self.lessons.get("global", []))[-24:]) or "(none yet)"
        return (
            f"PROBLEM `{slug}` ({direction}): {problem.brief()}\n\n"
            f"Record to beat: {ps.record!r}. Score needed for first place: {ps.target!r}. Our best so far: {_fmt(ps.best, 15)}.\n\n"
            f"KERNELS AND PARAMETERS\n{kernels}\n\n"
            f"STRATEGIES SO FAR\n{self._strategy_table(market, problem, slug)}\n\n"
            f"LESSONS\n{lessons}"
        )

    # ---- actions ----------------------------------------------------------------------------
    def propose(self, market: Market, problem, slug: str, n: int, pilot_usd: float) -> list[Strategy]:
        user = (
            self._context(market, problem, slug)
            + f"\n\nTASK: propose exactly {n} new strategies for this problem. Each gets a pilot of about ${pilot_usd:.2f} "
            f"(roughly {pilot_usd / 0.0000153:.0f} core-seconds, spread over at least 8 parallel workers), so a strategy whose single "
            "attempt takes minutes yields only a handful of attempts and is priced poorly; prefer attempts short enough to give 30 or more "
            "per pilot unless the evidence says long chains are the point. "
            "Also list any strategies to retire (only ones whose evidence shows they cannot reach the record), and give your current "
            "rating for every live strategy whose evidence has changed your view."
        )
        out = self.llm.json("propose", self.model, SYSTEM, user, PROPOSE_SCHEMA, effort=self.effort)
        new = []
        if not out:
            return new
        specs = problem.kernels()
        for pr in out.get("proposals", [])[:n]:
            if pr["kernel"] not in specs:
                self.ledger.append("note", text=f"PI proposed unknown kernel {pr['kernel']!r} for {slug}; dropped.")
                continue
            params = specs[pr["kernel"]].resolve({kv["name"]: kv["value"] for kv in pr["params"]})
            if market.is_duplicate(slug, pr["kernel"], params):
                self.ledger.append("note", text=f"PI proposed a duplicate of an existing {slug} strategy; dropped.")
                continue
            parent = pr.get("parent") if pr.get("parent") in market.strategies else None
            s = Strategy(
                sid=market.new_sid(), problem=slug, kernel=pr["kernel"], params=params, hypothesis=pr["hypothesis"], rationale=pr["rationale"],
                source="pi", proposer=self.model, parent=parent,
                prior={"p_record": float(min(max(pr["p_record"], 0.0), 1.0)), "expected_best": float(pr["expected_best"])},
            )
            market.add(s)
            self.ledger.append("strategy", **s.public())
            new.append(s)
        for r in out.get("ratings", []):
            t = market.strategies.get(r["sid"])
            if t and t.problem == slug and not t.retired:
                old = t.rating if t.rating is not None else t.prior.get("p_record")
                t.rating = float(min(max(r["p_record"], 0.0), 1.0))
                self.ledger.append("rating", sid=t.sid, problem=slug, p_record=t.rating, previous=old, why=r["why"], proposer=self.model)
        for sid in out.get("retire", []):
            s = market.strategies.get(sid)
            if s and s.problem == slug and not s.retired:
                s.retired = True
                self.ledger.append("note", text=f"PI retired {sid} ({slug}).")
        return new

    def reflect(self, market: Market, problem, slug: str, recent: list[dict]) -> None:
        """`recent`: referee verdicts for this problem's cards since the last reflection."""
        if not recent:
            return
        lines = []
        for r in recent:
            f = r["forecast"]
            lines.append(
                f"- card {r['cid']} strategy {r['sid']} ({r['stage']}, ${r['cost_usd']:.3f}): forecast[{f.get('source')}] P(new best)={_fmt(f.get('p_new_best'), 3)} "
                f"P(record)={_fmt(f.get('p_record'), 3)} expected best in [{_fmt(f.get('best_q10'), 9)}, {_fmt(f.get('best_q90'), 9)}]; "
                f"outcome: best={_fmt(r['outcome']['best'], 12)} new_best={r['outcome']['new_best']} record={r['outcome']['beat_record']} surprise={_fmt(r['surprise'], 3)}"
            )
        user = (
            self._context(market, problem, slug)
            + "\n\nLATEST RESULTS (forecast vs outcome)\n" + "\n".join(lines)
            + "\n\nTASK: write 1 to 3 lessons from these results. A lesson must be specific enough to change what gets proposed next "
            "(what worked, what is ruled out, which forecast was wrong and why). Do not restate the tables."
        )
        out = self.llm.json("reflect", self.model, SYSTEM, user, REFLECT_SCHEMA, effort=self.effort)
        for lesson in (out or {}).get("lessons", [])[:3]:
            self.lessons.setdefault(slug, []).append(lesson["text"])
            self.ledger.append("lesson", problem=slug, text=lesson["text"], evidence=lesson["evidence"], proposer=self.model)
