"""Thin, cost-tracked wrapper around an OpenAI-compatible API. Every call is written to the ledger."""

from __future__ import annotations

import json
import os
import time

# USD per 1M tokens (input, cached input, output); OpenAI standard tier, checked 2026-11-14.
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
}


class LLM:
    def __init__(self, ledger=None, budget_usd: float = float("inf"), base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or os.environ.get("ANTE_LLM_BASE_URL") or None, api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def json(self, purpose: str, model: str, system: str, user: str, schema: dict, effort: str = "low", max_output_tokens: int = 6000) -> dict | None:
        """One structured-output call. Returns the parsed object, or None if the budget is exhausted or the call fails."""
        if self.spent_usd >= self.budget_usd:
            return None
        t0 = time.time()
        try:
            r = self.client.responses.create(
                model=model,
                instructions=system,
                input=user,
                reasoning={"effort": effort},
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "json_schema", "name": purpose.replace("-", "_"), "schema": schema, "strict": True}},
            )
            u = r.usage
            cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            pin, pcached, pout = PRICES.get(model, (2.0, 0.2, 10.0))
            cost = ((u.input_tokens - cached) * pin + cached * pcached + u.output_tokens * pout) / 1e6
            self.spent_usd += cost
            out = json.loads(r.output_text)
            err = None
        except Exception as e:
            u, cost, out, err = None, 0.0, None, f"{type(e).__name__}: {e}"[:300]
        if self.ledger is not None:
            self.ledger.append(
                "llm_call", purpose=purpose, model=model, tokens_in=getattr(u, "input_tokens", 0), tokens_out=getattr(u, "output_tokens", 0),
                cost_usd=cost, seconds=round(time.time() - t0, 2), error=err,
            )
        return out
