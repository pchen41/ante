"""The unit of compute: run one kernel with one seed for a fixed time. No LLM, no network."""

from __future__ import annotations

import time
import traceback


def run_job(job: dict) -> dict:
    """job = {problem, kernel, params, seed, seconds, elites}. Never raises."""
    t0 = time.time()
    try:
        from ante import problems

        problem = problems.get(job["problem"])
        b = problem.run(job["kernel"], job["params"], int(job["seed"]), float(job["seconds"]), job.get("elites") or [])
        return {
            "ok": True, "seed": job["seed"], "scores": [float(s) for s in b.scores], "best_score": b.best_score,
            "best_solution": b.best_solution, "seconds": time.time() - t0, "meta": b.meta, "start_score": b.start_score,
        }
    except Exception as e:  # a failed worker must not take the card down
        return {"ok": False, "seed": job.get("seed"), "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:],
                "scores": [], "best_score": None, "best_solution": None, "seconds": time.time() - t0, "meta": {}}


def run_jobs(jobs: list[dict]) -> list[dict]:
    """Run several jobs in parallel processes inside one container (one process per core)."""
    import multiprocessing as mp

    if len(jobs) == 1:
        return [run_job(jobs[0])]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        return pool.map(run_job, jobs)


def warm() -> None:
    """Compile every numba kernel once (used at image build so workers start hot)."""
    from ante import problems

    for slug in problems.slugs():
        p = problems.get(slug)
        for name in p.kernels():
            for params in (p.warm_params() if hasattr(p, "warm_params") else [{"chain_seconds": 3.0, "time_limit": 2.0}]):
                try:
                    p.run(name, params, 0, 5.0, [])
                except Exception as e:
                    print("warm-up failed for", slug, name, repr(e))
