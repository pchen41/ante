"""The verification gauntlet. A score is only "ours" after it survives all four checks.

1. independent      : our own exact checker, written from the problem statement.
2. official-sandbox : the arena's downloaded verifier, executed in a Modal Sandbox with networking
                      blocked (it is third-party code; it never runs on the orchestrator or a laptop).
3. audit            : problem-specific evidence that the score is not a verifier artefact.
4. provenance       : the solution is not a copy (modulo symmetry) of a public solution.
"""

from __future__ import annotations

import json
from pathlib import Path

from .problem import DATA

SANDBOX_RUNNER = """
import json, sys
sys.setrecursionlimit(10000)
import verifier
with open("/work/solution.json") as f:
    solution = json.load(f)
try:
    print("ANTE_RESULT " + json.dumps({"ok": True, "score": float(verifier.evaluate(solution))}))
except Exception as e:
    print("ANTE_RESULT " + json.dumps({"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}))
"""


def official_sandbox(slug: str, solution: dict, timeout: int = 900, source: str | None = None) -> dict:
    """Run the official verifier on `solution` inside a network-blocked Modal Sandbox."""
    import modal

    app = modal.App.lookup("ante-sandbox", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.4", "scipy==1.18.1", "sympy==1.14.0")
    source = source if source is not None else (Path(DATA) / slug / "verifier.py.txt").read_text()
    sb = modal.Sandbox.create(app=app, image=image, block_network=True, timeout=timeout, cpu=1.0, memory=2048, workdir="/work")
    try:
        sb.filesystem.write_text(source, "/work/verifier.py")
        sb.filesystem.write_text(json.dumps(solution), "/work/solution.json")
        sb.filesystem.write_text(SANDBOX_RUNNER, "/work/run.py")
        p = sb.exec("python", "/work/run.py", timeout=timeout)
        out = p.stdout.read()
        err = p.stderr.read()
        p.wait()
    finally:
        sb.terminate()
    for line in out.splitlines():
        if line.startswith("ANTE_RESULT "):
            return json.loads(line[len("ANTE_RESULT "):])
    return {"ok": False, "error": ("no result line; stderr: " + err[-300:])}


def scores_agree(a: float, b: float, integer: bool) -> bool:
    if integer:
        return a == b
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def gauntlet(problem, solution: dict, ledger, sha: str, sandbox: bool = True) -> dict:
    """Run all checks, write one `verification` event per check, return a summary."""
    slug = problem.slug
    checks = {}

    try:
        indep = float(problem.exact_score(solution))
        checks["independent"] = (True, indep, "independent exact checker")
    except Exception as e:
        indep = None
        checks["independent"] = (False, None, f"{type(e).__name__}: {e}"[:300])

    if sandbox and indep is not None:
        try:
            r = official_sandbox(slug, solution, source=problem.verifier_source())
            if r.get("ok"):
                agree = scores_agree(r["score"], indep, problem.integer_scores)
                checks["official-sandbox"] = (agree, r["score"], "official verifier in a network-blocked sandbox" + ("" if agree else f"; DISAGREES with independent score {indep!r}"))
            else:
                checks["official-sandbox"] = (False, None, "official verifier rejected the solution: " + r.get("error", "?"))
        except Exception as e:
            checks["official-sandbox"] = (False, None, f"sandbox failed: {type(e).__name__}: {e}"[:300])

    try:
        checks["audit"] = (True, indep, json.dumps(problem.audit(solution), default=str)[:1500])
    except Exception as e:
        checks["audit"] = (False, None, f"{type(e).__name__}: {e}"[:300])

    try:
        owner = problem.is_public(solution)
        checks["provenance"] = (owner is None, indep, "differs from every public top-5 solution (modulo symmetry)" if owner is None else f"identical to the public solution of {owner}")
    except Exception as e:
        checks["provenance"] = (False, None, f"{type(e).__name__}: {e}"[:300])

    for name, (ok, score, detail) in checks.items():
        ledger.append("verification", problem=slug, solution_sha=sha, check=name, ok=bool(ok), score=score, detail=detail)
    return {"ok": all(ok for ok, _, _ in checks.values()), "score": indep, "checks": {k: v[0] for k, v in checks.items()}}
