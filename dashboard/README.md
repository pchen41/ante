# ante dashboard

One self-contained page (`index.html`, no build, no dependencies) that replays the ledger in the browser.

- Sample data: `cd dashboard && python3 -m http.server 8000`, then open <http://localhost:8000/index.html?ledger=sample_ledger.jsonl>
- A real run: serve the repo root instead and open `/dashboard/index.html?ledger=../runs/<run>/ledger.jsonl`
- `?ledger=URL`: ledger to load, relative to the page (default `ledger.jsonl`). Re-fetched whole on every poll.
- `?poll=SECONDS`: polling interval (default 5, minimum 1). The Pause button in the header stops polling.
- Opened from disk (`file://`) browsers block fetch; use the file picker shown in the waiting state instead.
- Debugging: `window.__state` is the replayed state; `window.__ante.replay(events)` is the pure replay function.
