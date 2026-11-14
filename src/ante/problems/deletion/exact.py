"""Independent exact checker for EinsteinArena `two-deletion-code-16`.

A code C of 16-bit binary words is valid iff the sets of length-14
subsequences ("descendants") of its words are pairwise disjoint.  The score is
the number of words.

This module is deliberately slow, tiny and string based: it shares no code
with the fast integer kernels in `graph.py` / `search.py`, so it can be used
to referee them.  It never imports or executes the downloaded arena verifier.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

WORD_LENGTH = 16
DELETIONS = 2
MAX_WORDS = 1 << (WORD_LENGTH - DELETIONS)

REPO_ROOT = Path(__file__).resolve().parents[4]
ARENA_DIR = REPO_ROOT / "data" / "arena" / "two-deletion-code-16"

# All ways to choose the 14 positions that are kept (equivalently, the 2 that
# are deleted).  120 index tuples, computed once.
_KEEP = tuple(combinations(range(WORD_LENGTH), WORD_LENGTH - DELETIONS))


def descendants(word: str) -> frozenset[str]:
    """Distinct length-14 subsequences of a 16-character word."""
    if len(word) != WORD_LENGTH:
        raise ValueError("Every codeword must contain exactly 16 binary digits")
    return frozenset("".join(word[k] for k in keep) for keep in _KEEP)


def _check_word(word: object) -> str:
    if not isinstance(word, str) or len(word) != WORD_LENGTH:
        raise ValueError("Every codeword must contain exactly 16 binary digits")
    for ch in word:
        if ch != "0" and ch != "1":
            raise ValueError("Every codeword must contain exactly 16 binary digits")
    return word


def verify(words: list[str]) -> int:
    """Return len(words) if `words` is a valid two-deletion-correcting code.

    Raises ValueError if the input is not a list of 1..16384 distinct
    16-character strings over {'0','1'} or if two words share a descendant.
    """
    if not isinstance(words, list) or not 1 <= len(words) <= MAX_WORDS:
        raise ValueError("Expected between 1 and 16384 codewords")

    seen: set[str] = set()
    owner: dict[str, str] = {}
    for word in words:
        _check_word(word)
        if word in seen:
            raise ValueError(f"Codewords must be distinct (duplicate {word})")
        seen.add(word)
        for d in descendants(word):
            other = owner.get(d)
            if other is not None:
                raise ValueError(
                    f"Two codewords share a two-deletion descendant: {other} and {word} -> {d}"
                )
            owner[d] = word
    return len(words)


def verify_solution(solution: dict) -> int:
    """Verify an arena-style solution object `{"words": [...]}`."""
    if not isinstance(solution, dict) or "words" not in solution:
        raise ValueError('Expected an object with a "words" list')
    return verify(solution["words"])


def is_valid(words: list[str]) -> bool:
    try:
        verify(words)
    except ValueError:
        return False
    return True


def load_public_solutions() -> list[dict]:
    """The public top-5 snapshot: list of {id, agentName, score, createdAt, data}."""
    with open(ARENA_DIR / "best_solutions.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_public_words(score: int) -> list[str]:
    """Words of the public solution with the given leaderboard score."""
    for item in load_public_solutions():
        if int(item["score"]) == int(score):
            return list(item["data"]["words"])
    raise KeyError(f"no public solution with score {score}")


def main() -> None:
    for item in load_public_solutions():
        n = verify(item["data"]["words"])
        print(f"id={item['id']} agent={item['agentName']} listed={item['score']} verified={n}")


if __name__ == "__main__":
    main()
