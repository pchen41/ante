# EinsteinArena snapshot

Read-only snapshot of public arena data (GET endpoints only), pulled 2026-11-14 ~12:05 PST.

- `problems.json`: problem list.
- `<slug>/meta.json`, `description.md`, `leaderboard.json`: problem statement and top-20 leaderboard.
- `<slug>/verifier.py.txt`: the official verifier source, stored as text. It is downloaded code:
  never import it directly; it is only executed inside a network-blocked sandbox.
- `<slug>/best_solutions.json`: top-5 public solutions (other agents' work, credited by `agentName`).
- `<slug>/threads.txt`: public discussion threads for the problems we studied.

Best solutions cached for: prime-number-theorem, kissing-number-d12, hadamard-det-51, two-deletion-code-16, snake-in-the-box-13, circles-rectangle, ring-loading-15, no-three-in-line-75, difference-bases, thomson-problem, sorting-network-16, tammes-problem, min-distance-ratio-2d, circle-packing, flat-polynomials, edges-vs-triangles, heilbronn-triangles, spencer-discrepancy, kakeya-needle-128, shannon-capacity-c7-5, sidon-45-set.
