"""Tests for the two-deletion-code-16 instruments (exact checker, tables, search)."""

import json

import numpy as np
import pytest

from ante.problems.deletion import exact, exchange, graph, search

Z16 = "0" * 16
Z14 = "0" * 14


@pytest.fixture(scope="module")
def tables():
    return graph.load_tables()


@pytest.fixture(scope="module")
def public():
    return exact.load_public_solutions()


# -- exact.verify ----------------------------------------------------------------


def test_public_solutions_verify(public):
    sizes = [exact.verify(item["data"]["words"]) for item in public]
    assert sorted(sizes, reverse=True) == [226, 225, 224, 223, 222]
    for item, size in zip(public, sizes):
        assert size == int(item["score"])


def test_descendants_hand_checked():
    assert exact.descendants(Z16) == {Z14}
    assert exact.descendants("1" * 16) == {"1" * 14}
    assert exact.descendants("0" * 15 + "1") == {Z14, "0" * 13 + "1"}
    assert exact.descendants("0" * 14 + "11") == {Z14, "0" * 13 + "1", "0" * 12 + "11"}
    assert exact.descendants("1" + "0" * 14 + "1") == {
        Z14,
        "0" * 13 + "1",
        "1" + "0" * 13,
        "1" + "0" * 12 + "1",
    }
    assert len(exact.descendants("0101010101010101")) <= 120


def test_verify_accepts_small_codes():
    assert exact.verify([Z16]) == 1
    assert exact.verify([Z16, "1" * 16]) == 2
    # 0^13 111 keeps at least one 1 after two deletions, so it cannot reach 0^14
    assert exact.verify([Z16, "0" * 13 + "111"]) == 2
    assert exact.verify_solution({"words": [Z16, "1" * 16]}) == 2


@pytest.mark.parametrize(
    "words",
    [
        [],
        Z16,  # not a list
        (Z16,),  # not a list
        [Z16, Z16],  # duplicate
        ["0" * 15],  # too short
        ["0" * 17],  # too long
        ["0" * 15 + "2"],  # not binary
        ["0" * 15 + " "],
        [0],  # not a string
        [Z16, "0" * 15 + "1"],  # share 0^14
        [Z16, "0" * 14 + "11"],  # share 0^14
        ["0101010101010101", "1010101010101010"],  # share (01)^7
    ],
)
def test_verify_rejects(words):
    with pytest.raises(ValueError):
        exact.verify(words)
    assert not exact.is_valid(words)


# -- graph tables ----------------------------------------------------------------


def test_tables_shapes_and_counts(tables):
    assert tables.dptr.shape == (65537,)
    assert tables.pptr.shape == (16385,)
    nd = np.diff(tables.dptr)
    npar = np.diff(tables.pptr)
    assert nd.min() == 1 and nd.max() <= 120
    # every length-14 word has exactly C(16,0)+C(16,1)+C(16,2) = 137 supersequences of length 16
    assert npar.min() == 137 and npar.max() == 137
    assert tables.didx.shape[0] == 16384 * 137
    assert tables.pidx.shape[0] == 16384 * 137
    assert tables.didx.min() == 0 and tables.didx.max() == 16383
    assert tables.pidx.min() == 0 and tables.pidx.max() == 65535


def test_tables_hand_checked_words(tables):
    def desc(s):
        return {graph.int_to_word(d, 14) for d in tables.desc(int(s, 2)).tolist()}

    assert desc(Z16) == {Z14}
    assert desc("1" * 16) == {"1" * 14}
    assert desc("0" * 15 + "1") == {Z14, "0" * 13 + "1"}
    assert desc("0" * 14 + "11") == {Z14, "0" * 13 + "1", "0" * 12 + "11"}
    assert desc("1" + "0" * 14 + "1") == {Z14, "0" * 13 + "1", "1" + "0" * 13, "1" + "0" * 12 + "1"}


def test_tables_match_string_definition(tables):
    rng = np.random.default_rng(1)
    sample = rng.integers(0, 1 << 16, size=300).tolist() + [0, 1, 0x8000, 0xFFFF, 0x5555, 0xAAAA, 0x00FF]
    for w in sample:
        expect = sorted(int(d, 2) for d in exact.descendants(graph.int_to_word(w)))
        got = tables.desc(w).tolist()
        assert got == expect  # sorted, distinct, identical to the string definition


def test_inverse_table_is_the_transpose(tables):
    nd = np.diff(tables.dptr)
    npar = np.diff(tables.pptr)
    fwd = np.repeat(np.arange(65536, dtype=np.int64), nd) * 16384 + tables.didx
    inv = tables.pidx.astype(np.int64) * 16384 + np.repeat(np.arange(16384, dtype=np.int64), npar)
    assert np.array_equal(np.sort(fwd), np.sort(inv))
    # parents of 0^14: words of weight <= 2
    wt = graph.weights(16)
    assert tables.parents(0).tolist() == np.flatnonzero(wt <= 2).tolist()


def test_weight_band_locality(tables):
    w16 = graph.weights(16)
    w14 = graph.weights(14)
    rep = np.repeat(np.arange(65536), np.diff(tables.dptr))
    diff = w16[rep] - w14[tables.didx]
    assert diff.min() >= 0 and diff.max() <= 2


def test_symmetries_preserve_validity(tables, public):
    assert int(graph.reverse(1)) == 0x8000
    assert int(graph.complement(0)) == 0xFFFF
    words = public[0]["data"]["words"]
    ints = graph.words_to_ints(words)
    for image in graph.symmetry_images(ints):
        assert exact.verify(graph.ints_to_words(image)) == len(words)


def test_public_solutions_valid_in_fast_tables(tables, public):
    for item in public:
        ints = graph.words_to_ints(item["data"]["words"])
        used = np.concatenate([tables.desc(int(w)) for w in ints])
        assert np.unique(used).shape[0] == used.shape[0]


# -- exchange checks ---------------------------------------------------------------


def test_exchange_finds_removed_word(tables, public):
    ints = graph.words_to_ints(public[0]["data"]["words"])
    report = exchange.analyse(ints[1:], tables)
    assert report["size"] == len(ints) - 1
    assert report["free_words"] >= 1


def test_window_milp_recovers_removed_word(tables, public):
    from ante.problems.deletion import window

    ints = graph.words_to_ints(public[0]["data"]["words"]).astype(np.int64)
    removed, code = int(ints[0]), ints[1:]
    release = window.cluster_window(tables, code, removed, 6)
    rep = window.solve_window(tables, code, release, time_limit=30.0)
    assert rep["gain"] >= 1
    assert rep["new_size"] >= len(ints)
    assert exact.verify(graph.ints_to_words(rep["new_code"])) == rep["new_size"]


# -- search --------------------------------------------------------------------------


def test_search_smoke_from_scratch(tables):
    out = search.search(seconds=2.0, seed=1, start="scratch", tables=tables)
    assert out["size"] == len(out["words"]) == exact.verify(out["words"])
    assert out["size"] >= 150
    assert out["meta"]["iterations"] > 0


def test_search_bookkeeping_stays_consistent(tables):
    s = search.Searcher(tables, seed=2, p_ruin=0.1, p_force=0.4, stall=500)
    s.start(None)
    s.check_state()
    res = s.run(1.5)
    s.check_state()
    words = s.best_words()
    assert exact.verify(words) == res["best"]


def test_search_ablation_modes_run(tables):
    for params in (
        dict(use_ls=False),
        dict(accept="strict", p_ruin=0.0, p_force=0.0),
        dict(accept="metropolis", temp=0.3, ruin_mode="band", p_ruin=0.2, ruin_greedy_k=3),
    ):
        s = search.Searcher(tables, seed=3, **params)
        s.start(None)
        res = s.run(0.5)
        s.check_state()
        assert exact.verify(s.best_words()) == res["best"]


def test_search_from_public_never_loses(tables):
    out = search.search(seconds=1.0, seed=0, start="226", tables=tables)
    info = out["meta"]["start_info"]
    assert info["loaded"] == 226 and info["skipped_conflicting"] == 0
    assert out["size"] >= 226
    assert exact.verify(out["words"]) == out["size"]


def test_search_cli_writes_verified_json(tmp_path):
    path = tmp_path / "run.json"
    search.main(["--seconds", "1", "--seed", "0", "--start", "scratch", "--out", str(path)])
    obj = json.loads(path.read_text())
    assert set(obj) == {"words", "size", "meta"}
    assert exact.verify(obj["words"]) == obj["size"]
