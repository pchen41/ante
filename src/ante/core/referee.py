"""The referee scores every pre-registered forecast against what happened."""

from __future__ import annotations

import math


def judge(forecast: dict, outcome: dict, maximize: bool) -> dict:
    """forecast: the card's forecast. outcome: {new_best: bool, beat_record: bool, best: float | None}."""
    def brier(p, happened):
        return None if p is None else (float(p) - float(bool(happened))) ** 2

    surprise = 0.0
    for p, happened in ((forecast.get("p_new_best"), outcome["new_best"]), (forecast.get("p_record"), outcome["beat_record"])):
        if p is None:
            continue
        p_event = p if happened else 1.0 - p
        surprise = max(surprise, -math.log10(max(p_event, 1e-12)))
    lo, hi, best = forecast.get("best_q10"), forecast.get("best_q90"), outcome.get("best")
    in_interval = None
    if lo is not None and hi is not None and best is not None:
        a, b = min(lo, hi), max(lo, hi)
        in_interval = a <= best <= b
        if not in_interval:
            width = max(b - a, 1e-300)
            surprise = max(surprise, min((a - best if best < a else best - b) / width, 99.0))
    direction = None
    if in_interval is False:
        better_than_expected = best > max(lo, hi) if maximize else best < min(lo, hi)
        direction = "better" if better_than_expected else "worse"
    return {
        "brier_new_best": brier(forecast.get("p_new_best"), outcome["new_best"]),
        "brier_record": brier(forecast.get("p_record"), outcome["beat_record"]),
        "in_interval": in_interval, "surprise": round(surprise, 3), "direction": direction,
    }
