"""Problem registry."""

from __future__ import annotations

from functools import lru_cache


def _registry() -> dict:
    from .deletion.problem import DeletionCode
    from .flat_poly.problem import FlatPolynomials
    from .kakeya.problem import KakeyaNeedle

    return {p.slug: p for p in (KakeyaNeedle, DeletionCode, FlatPolynomials)}


def slugs() -> list[str]:
    return list(_registry())


@lru_cache(maxsize=None)
def get(slug: str):
    return _registry()[slug]()
