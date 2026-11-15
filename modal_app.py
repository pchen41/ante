"""ante on Modal: search workers, the overnight orchestrator, the sandboxed verifier and the dashboard.

    modal deploy modal_app.py                      # deploy workers + dashboard
    modal run modal_app.py::smoke                  # tiny end-to-end check (a few cents)
    modal run --detach modal_app.py::launch --compute-usd 20 --hours 2 --run-id demo

Layout inside the container mirrors the repo (/root/src/ante, /root/data/arena) so paths resolve identically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("ante")

CORES = 4  # search processes per worker container
ROOT = Path(__file__).parent


def _warm() -> None:
    from ante.core import worker

    worker.warm()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "numba==0.68.1", "openai==2.6.1", "fastapi[standard]==0.162.0")
    # NUMBA_CPU_NAME pins code generation to the x86-64-v3 level (AVX2), so the kernels compiled once at
    # image build are reused on every worker instead of being recompiled per host CPU model.
    .env({"PYTHONPATH": "/root/src", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
          "NUMBA_CPU_NAME": "x86-64-v3",
          "NUMBA_CPU_FEATURES": "+avx,+avx2,+bmi,+bmi2,+f16c,+fma,+lzcnt,+movbe,+popcnt,+sse3,+ssse3,+sse4.1,+sse4.2,+cx16,+sahf,+xsave"})
    .add_local_dir(ROOT / "data" / "arena", "/root/data/arena", copy=True)
    .add_local_dir(ROOT / "data" / "alphaevolve", "/root/data/alphaevolve", copy=True)
    .add_local_dir(ROOT / "src" / "ante", "/root/src/ante", copy=True, ignore=["**/__pycache__"])
    .run_function(_warm)  # compile numba kernels and build lookup tables into the image
)
if (ROOT / "dashboard" / "index.html").exists():
    image = image.add_local_dir(ROOT / "dashboard", "/root/dashboard", copy=True)

runs = modal.Volume.from_name("ante-runs", create_if_missing=True)


# cpu=(CORES, CORES): a hard limit, so a container can never use (and be billed for) more than it asked for;
# scaledown_window=2: a finished container stops within seconds instead of idling for the default minute.
# Both were found by reconciling the ledger with Modal's billing report: idle time was two thirds of the first bill.
@app.function(image=image, cpu=(CORES, CORES), memory=CORES * 1024, timeout=2 * 3600, max_containers=100, scaledown_window=2)
def work(jobs: list[dict]) -> list[dict]:
    """One worker container: run up to CORES seeds of one strategy in parallel processes. No network use, no LLM."""
    from ante.core import worker

    return worker.run_jobs(jobs)


# nonpreemptible: the orchestrator holds the run's state in memory, so it must not be evicted mid-run (workers may be).
@app.function(image=image, cpu=1, memory=2048, timeout=24 * 3600, secrets=[modal.Secret.from_name("ante-keys")], volumes={"/runs": runs}, nonpreemptible=True)
def orchestrate(cfg: dict) -> dict:
    """The research loop itself. Runs in the cloud so nothing depends on a laptop staying awake."""
    from ante.core.loop import Config, Loop
    from ante.core.runner import ModalRunner

    import threading

    dirty, stop = threading.Event(), threading.Event()

    def committer() -> None:  # publish the ledger to the dashboard every few seconds while it changes
        while not stop.is_set():
            time.sleep(5)
            if dirty.is_set():
                dirty.clear()
                try:
                    runs.commit()
                except Exception as e:
                    print("volume commit failed:", repr(e))

    threading.Thread(target=committer, daemon=True).start()
    import os

    cfg = {**cfg, "app_id": cfg.get("app_id") or os.environ.get("MODAL_APP_ID", "")}
    runs.reload()  # see the ledgers of earlier runs (for carry-over)
    config = Config(**cfg)
    loop = Loop(config, ModalRunner(work, cores=CORES, max_workers=100 * CORES), f"/runs/{config.run_id}", on_write=dirty.set)
    try:
        loop.run()
    finally:
        stop.set()
        runs.commit()
    return {"run_id": config.run_id, "spent_compute_usd": loop.spent, "spent_llm_usd": loop.llm_spent(),
            "best": {k: v.best for k, v in loop.market.problems.items()}}


@app.function(image=image, cpu=0.25, memory=512, volumes={"/runs": runs}, min_containers=0, scaledown_window=300)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def dashboard():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    web = FastAPI()

    def run_dirs() -> list[str]:
        runs.reload()
        ds = [d for d in Path("/runs").iterdir() if (d / "ledger.jsonl").exists()]
        return [d.name for d in sorted(ds, key=lambda d: (d / "ledger.jsonl").stat().st_mtime, reverse=True)]

    @web.get("/", response_class=HTMLResponse)
    def index():
        page = Path("/root/dashboard/index.html")
        return page.read_text() if page.exists() else "<p>dashboard not deployed yet; the raw ledger is at /ledger.jsonl</p>"

    @web.get("/runs")
    def list_runs():
        return JSONResponse(run_dirs())

    @web.get("/ledger.jsonl", response_class=PlainTextResponse)
    def ledger(run: str | None = None):
        names = run_dirs()
        name = run if run in names else (names[0] if names else None)
        if name is None:
            return ""
        return (Path("/runs") / name / "ledger.jsonl").read_text()

    @web.get("/candidate")
    def candidate(run: str, path: str):
        f = (Path("/runs") / run / path).resolve()
        if not str(f).startswith(f"/runs/{run}/candidates/") or not f.exists():
            raise HTTPException(404)
        return JSONResponse(json.loads(f.read_text()))

    return web


@app.local_entrypoint()
def launch(
    run_id: str = "",
    compute_usd: float = 2.0,
    llm_usd: float = 1.0,
    hours: float = 0.5,
    cycles: int = 1000,
    pilot_usd: float = 0.05,
    tranche_usd: float = 0.5,
    fund_k: int = 3,
    scale_seconds: float = 240.0,
    pilot_seconds: float = 60.0,
    proposals: int = 2,
    pi_model: str = "gpt-6.1-terra",
    pi_effort: str = "medium",
    problems: str = "",
    seed: int = 0,
    no_sandbox: bool = False,
    wait: bool = False,
    carry_from: str = "",
    max_card_seconds: float = 1800.0,
    allocator: str = "priced",
    no_pi: bool = False,
):
    """Start a run in the cloud. Prints the estimated maximum spend first."""
    from ante.core.loop import SEEDS

    cfg = dict(
        run_id=run_id or time.strftime("run-%m%d-%H%M"), problems=(problems.split(",") if problems else list(SEEDS)),
        compute_budget_usd=compute_usd, llm_budget_usd=llm_usd, max_cycles=cycles, max_wall_seconds=hours * 3600, pilot_usd=pilot_usd,
        pilot_seconds=pilot_seconds, tranche_usd=tranche_usd, fund_k=fund_k, scale_seconds=scale_seconds, proposals_per_problem=proposals,
        pi_model=pi_model, pi_effort=pi_effort, sandbox_verify=not no_sandbox, seed=seed,
        carry_from=[r for r in carry_from.split(",") if r], max_card_seconds=max_card_seconds, app_id=app.app_id or "", allocator=allocator, use_pi=not no_pi,
    )
    print(f"Launching {cfg['run_id']}: compute budget ${compute_usd:.2f}, LLM budget ${llm_usd:.2f}, wall limit {hours:.1f} h")
    if wait:
        print(json.dumps(orchestrate.remote(cfg), indent=1))
    else:
        call = orchestrate.spawn(cfg)
        print("spawned", call.object_id, "- follow it on the dashboard")


@app.function(image=image, cpu=(CORES, CORES), memory=8192, timeout=600, scaledown_window=2)
def bench_container(procs: int, seconds: float = 20.0) -> dict:
    """How much search does one CORES-core container do with `procs` parallel processes? (Chooses PROCS.)"""
    import resource

    from ante import problems
    from ante.core import worker

    out = {}
    for slug, kernel in (("flat-polynomials", "tabu"), ("kakeya-needle-128", "hop"), ("two-deletion-code-16", "ils")):
        params = problems.get(slug).kernels()[kernel].resolve({"chain_seconds": seconds})
        jobs = [{"problem": slug, "kernel": kernel, "params": params, "seed": i, "seconds": seconds, "elites": []} for i in range(procs)]
        t0 = time.time()
        res = worker.run_jobs(jobs)
        work_done = sum(r["meta"].get("steps", 0) + r["meta"].get("hops", 0) + r["meta"].get("iterations", 0) for r in res)
        out[slug] = {"work": work_done, "wall": round(time.time() - t0, 1), "ok": all(r["ok"] for r in res)}
    out["max_rss_mb_children"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    import os

    out["os_cpu_count"] = os.cpu_count()
    return out


@app.local_entrypoint()
def bench():
    calls = {n: bench_container.spawn(n) for n in (4, 8, 12)}
    base = None
    for n, c in calls.items():
        r = c.get()
        base = base or r
        print(n, "processes:", {k: (v, f"x{v['work'] / max(base[k]['work'], 1):.2f}") if isinstance(v, dict) else v for k, v in r.items()})


@app.local_entrypoint()
def smoke():
    """A few cents: 8 worker jobs across 2 containers, then the sandboxed official verifier on a public solution."""
    from ante import problems
    from ante.core import verify

    jobs = [{"problem": slug, "kernel": k, "params": problems.get(slug).kernels()[k].resolve({"chain_seconds": 10, "time_limit": 5}), "seed": i, "seconds": 20.0, "elites": []}
            for i, (slug, k) in enumerate([("kakeya-needle-128", "hop"), ("two-deletion-code-16", "ils"), ("two-deletion-code-16", "window"), ("flat-polynomials", "tabu")] * 2)]
    t0 = time.time()
    out = [r for chunk in work.map([jobs[:4], jobs[4:]]) for r in chunk]
    print(f"workers returned in {time.time() - t0:.1f}s")
    for j, r in zip(jobs, out):
        print(f"  {j['problem']:22s} {j['kernel']:7s} ok={r['ok']} attempts={len(r['scores'])} best={r['best_score']} worker_seconds={r['seconds']:.1f} {r.get('error', '')}")
    for slug in ("kakeya-needle-128", "two-deletion-code-16", "flat-polynomials"):
        p = problems.get(slug)
        top = p.public_solutions()[0]
        t0 = time.time()
        r = verify.official_sandbox(slug, top["data"])
        print(f"sandbox {slug}: {r} | leaderboard {top['score']!r} | independent {p.exact_score(top['data'])!r} | {time.time() - t0:.1f}s")
