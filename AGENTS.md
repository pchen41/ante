# AGENTS.md

Project: `ante`, an autonomous research loop targeting open problems on EinsteinArena (einsteinarena.com).

## Commands
- Python 3.12 via uv. Run anything with `uv run python ...`; tests with `uv run pytest -q`.
- Do not `uv add` casually: dependencies are pinned behind `exclude-newer` (>= 7 days old).
- Modal CLI: `uv run modal ...` (account already logged in on this machine).

## Hard rules
- Secrets (`.env`, API keys, Modal tokens) never enter the repo or its history. The repo is public.
- `data/arena/*/verifier.py.txt` is downloaded code. Never import or exec it locally; it only runs
  inside a network-blocked sandbox (`ante.verify`). Local scoring uses our own independent checkers.
- No POST requests to einsteinarena.com (registration, submissions, threads) without explicit human approval.
- Never claim a score that has not been reproduced by an independent exact checker.
- Modal spend beyond small tests needs human approval; every launch records its estimated cost.

## Layout
- `src/ante/problems/<name>/`: per-problem instruments: `exact.py` (independent exact checker),
  fast scoring kernels, search kernels.
- `src/ante/core/`: the loop: ledger, experiment cards, allocator, PI (LLM), referee.
- `data/arena/`: read-only snapshot of public arena data (see `SNAPSHOT.md`).
- `runs/`: ledgers and verified artifacts that are part of the public record.
- `scratch/`: gitignored working space.
