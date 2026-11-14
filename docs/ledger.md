# The ledger

Every decision the loop makes is one JSON object on one line of `ledger.jsonl`, appended in order and
never rewritten. The dashboard, the referee's calibration tables and the results in the README are all
computed by replaying this file. If it is not in the ledger, the system did not decide it.

Every event has `seq` (0, 1, 2, ...), `ts` (unix seconds) and `type`. Scores are always in the problem's
own units (the number the arena would report). Dollar amounts are USD.

| `type` | When | Fields |
|---|---|---|
| `run_start` | once | `run_id`, `budget_usd: {compute, llm}`, `problems: [slug]`, `config: {...}` |
| `arena_snapshot` | start of each cycle, per problem | `problem`, `title`, `maximize: bool`, `record`, `target` (score needed to take #1), `leader`, `top: [{agentName, bestScore}]` |
| `strategy` | a new search strategy is proposed | `sid`, `problem`, `kernel`, `params: {...}`, `hypothesis`, `rationale`, `source: "seed" \| "pi"`, `proposer` (model name or `"seed"`), `parent: sid \| null`, `prior: {p_record, expected_best}` (the proposer's own forecast for a standard tranche; may be null) |
| `card` | **before** compute is spent | `cid`, `sid`, `problem`, `stage: "pilot" \| "scale"`, `workers`, `seconds`, `budget_usd`, `forecast: {source: "prior" \| "tail-fit" \| "beta", p_new_best, p_record, best_q10, best_q50, best_q90}` |
| `result` | a card finishes | `cid`, `sid`, `problem`, `attempts`, `best` (best score of the card, by the kernel), `quantiles: {q01, q10, q50}` of attempt outcomes, `cost_usd`, `wall_seconds`, `core_seconds`, `new_best: bool`, `beat_record: bool`, `error: str \| null` |
| `price` | after each result | `sid`, `problem`, `model: "gpd" \| "beta"`, `n_attempts`, `cost_per_attempt`, `p_attempt`, `p_lo`, `p_hi`, `usd_to_record`, `opening_price_usd`, `started_from_known`, `improved_on_start` (attempts that began from a known solution, and how many ended strictly better), `xi`, `sigma`, `u` |
| `referee` | after each result | `cid`, `sid`, `forecast` (copy of the card's), `outcome: {new_best, beat_record, best}`, `brier_new_best`, `brier_record`, `in_interval: bool` (was `best` inside [q10, q90]?), `surprise` (0 = as expected, larger = more surprising) |
| `allocation` | each cycle, before scale cards | `cycle`, `budget_left_usd`, `tranche_usd`, `table: [{sid, problem, model, usd_to_record, p_record_tranche, p_new_best_tranche, climb: [improved, started] (attempts that began from a top-3 known solution), draw, trust, trust_new_best, chosen: bool, budget_usd}]`, `calibration: {"source:event": {n, forecast_sum, observed, trust}}`, `note` |
| `candidate` | a card produced a new best for a problem | `problem`, `score`, `cid`, `sid`, `solution_sha`, `path` |
| `verification` | each check of a candidate | `problem`, `solution_sha`, `check: "independent" \| "official-sandbox" \| "audit" \| "provenance"`, `ok: bool`, `score`, `detail` |
| `rating` | the PI re-rates a live strategy | `sid`, `problem`, `p_record` (its probability that $1 more compute beats the record; sets the opening price `1 / p_record`), `previous`, `why`, `proposer` |
| `lesson` | the PI reflects on results | `problem` (or `"global"`), `text`, `evidence: [cid]`, `proposer` |
| `llm_call` | every model call | `purpose`, `model`, `tokens_in`, `tokens_out`, `cost_usd`, `seconds` |
| `note` | free-text narration of a decision | `text` |
| `cycle` | start and end of each cycle | `n`, `phase: "start" \| "end"`, `spent_compute_usd`, `spent_llm_usd`; at the end also `usd_per_hour`, `hours` |
| `run_end` | once | `reason` |

## Reading it

- **What did it try, and why?** `strategy` events: each carries the hypothesis in plain words.
- **What did it expect?** `card.forecast`, written before the run.
- **What happened?** `result`, then `referee`, which scores the forecast (Brier score; 0 is perfect, 0.25 is
  a coin flip said with 50% confidence).
- **Why did the money go there?** `allocation.table`: every strategy's price (`usd_to_record`), the chance it
  was given for this tranche, the sampled draw that ranked it, and whether it was funded.
- **Can I trust the score?** `verification` events: the independent exact checker, the official verifier run
  in a network-blocked sandbox, a problem-specific audit, and a provenance check.
