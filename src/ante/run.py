"""Run the loop locally: `uv run python -m ante.run --help`. (The cloud entry point is `modal_app.py`.)"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ante.core.loop import SEEDS, Config, Loop
from ante.core.runner import LocalRunner


def load_env(path: str | Path = Path(__file__).resolve().parents[3] / ".env") -> None:
    """Read KEY=VALUE lines from a .env file that lives outside the repo."""
    path = Path(os.environ.get("ANTE_ENV_FILE", path))
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=time.strftime("local-%m%d-%H%M%S"))
    ap.add_argument("--problems", default=",".join(SEEDS))
    ap.add_argument("--compute-usd", type=float, default=0.05, help="compute budget, accounted at Modal prices")
    ap.add_argument("--llm-usd", type=float, default=0.25)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--pilot-usd", type=float, default=0.004)
    ap.add_argument("--pilot-seconds", type=float, default=20.0)
    ap.add_argument("--tranche-usd", type=float, default=0.012)
    ap.add_argument("--scale-seconds", type=float, default=40.0)
    ap.add_argument("--fund-k", type=int, default=2)
    ap.add_argument("--proposals", type=int, default=1)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--max-card-seconds", type=float, default=90.0)
    ap.add_argument("--min-pilot-workers", type=int, default=2)
    ap.add_argument("--pi-model", default="gpt-6-luna")
    ap.add_argument("--no-pi", action="store_true")
    ap.add_argument("--sandbox", action="store_true", help="also run the official verifier in a Modal Sandbox")
    ap.add_argument("--offline", action="store_true", help="use the arena snapshot instead of live leaderboards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    load_env()
    cfg = Config(
        run_id=a.run_id, problems=a.problems.split(","), compute_budget_usd=a.compute_usd, llm_budget_usd=a.llm_usd, max_cycles=a.cycles,
        pilot_usd=a.pilot_usd, pilot_seconds=a.pilot_seconds, tranche_usd=a.tranche_usd, scale_seconds=a.scale_seconds, fund_k=a.fund_k,
        proposals_per_problem=a.proposals, pi_model=a.pi_model, use_pi=not a.no_pi, max_card_seconds=a.max_card_seconds, min_pilot_workers=a.min_pilot_workers, sandbox_verify=a.sandbox, live_arena=not a.offline, seed=a.seed,
    )
    out = Path(a.out) / a.run_id
    Loop(cfg, LocalRunner(a.processes), out).run()
    print("ledger:", out / "ledger.jsonl")


if __name__ == "__main__":
    main()
