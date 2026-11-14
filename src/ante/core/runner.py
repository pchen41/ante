"""Where compute happens. A runner executes one card: the same kernel on many seeds for a fixed time.

Costs are estimated from Modal's public prices (CPU per physical core-second plus memory) so that a
local run and a cloud run account identically; the estimate is reconciled against Modal's billing page.
"""

from __future__ import annotations

import multiprocessing as mp

from . import worker

CPU_USD_PER_CORE_S = 0.0000131
MEM_USD_PER_GIB_S = 0.00000222
MEM_GIB_PER_CORE = 1.0
USD_PER_CORE_S = CPU_USD_PER_CORE_S + MEM_GIB_PER_CORE * MEM_USD_PER_GIB_S
# Seconds a container is billed for beyond the search itself: image start, imports, scale-down.
# Calibrated against Modal's billing report (see docs/cost.md); re-check it whenever the worker config changes.
CONTAINER_OVERHEAD_S = 12.0


def card_cost(results: list[dict], cores_per_container: int = 1) -> tuple[float, float]:
    """(estimated USD, core-seconds of search) for a finished card.

    A container is billed for all its cores for as long as it lives, so each container costs
    cores * (its longest job + overhead), whether or not every core had a job.
    """
    core_s = sum(r["seconds"] for r in results)
    billed = 0.0
    for i in range(0, len(results), max(cores_per_container, 1)):
        chunk = results[i : i + cores_per_container]
        billed += cores_per_container * (max(r["seconds"] for r in chunk) + CONTAINER_OVERHEAD_S)
    return billed * USD_PER_CORE_S, core_s


def workers_for(budget_usd: float, seconds: float, max_workers: int) -> int:
    return int(max(1, min(max_workers, budget_usd / ((seconds + CONTAINER_OVERHEAD_S) * USD_PER_CORE_S))))


class LocalRunner:
    name = "local"
    cores_per_container = 1

    def __init__(self, processes: int = 4):
        self.processes = processes
        self.max_workers = processes

    def run(self, jobs: list[dict]) -> list[dict]:
        if len(jobs) == 1 or self.processes == 1:
            return [worker.run_job(j) for j in jobs]
        with mp.get_context("spawn").Pool(min(self.processes, len(jobs))) as pool:
            return pool.map(worker.run_job, jobs)


class ModalRunner:
    """Fans a card out over Modal containers; each container runs `cores` seeds in parallel processes."""

    name = "modal"

    def __init__(self, fn, cores: int = 4, max_workers: int = 400):
        self.fn = fn
        self.cores_per_container = cores
        self.max_workers = max_workers

    def run(self, jobs: list[dict]) -> list[dict]:
        chunks = [jobs[i : i + self.cores_per_container] for i in range(0, len(jobs), self.cores_per_container)]
        out: list[dict] = []
        for chunk, res in zip(chunks, self.fn.map(chunks, order_outputs=True, return_exceptions=True)):
            if isinstance(res, Exception):
                # one error per job the container held, so that results stay aligned with containers for costing
                out.extend({"ok": False, "error": f"container failed: {type(res).__name__}: {res}"[:300], "scores": [], "best_score": None,
                            "best_solution": None, "seconds": float(job["seconds"]), "meta": {}, "seed": job["seed"]} for job in chunk)
            else:
                out.extend(res)
        return out
