"""Reciprocal Rank Fusion: hand-computed expected values."""

from __future__ import annotations

import pytest

from retriever import rrf_fuse


def test_hand_computed_scores_k60() -> None:
    semantic = ["a", "b", "c"]
    bm25 = ["c", "a", "d"]
    fused = dict(rrf_fuse([semantic, bm25], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_order_rewards_agreement() -> None:
    # "c" is mediocre in both lists but beats "x", which tops only one list.
    semantic = ["x", "c", "y"]
    bm25 = ["z", "c", "w"]
    order = [d for d, _ in rrf_fuse([semantic, bm25], k=60)]
    assert order[0] == "c"


def test_k_controls_top_rank_dominance() -> None:
    # "a" is #1 in one list only; "b" is #3 and #4. With a tiny k a single
    # first place dominates; with a large k consensus wins.
    l1 = ["a", "q", "b"]
    l2 = ["r", "s", "t", "b"]
    assert rrf_fuse([l1, l2], k=1)[0][0] == "a"  # 1/2 = 0.50 > 1/4 + 1/5 = 0.45
    assert rrf_fuse([l1, l2], k=60)[0][0] == "b"  # 1/63 + 1/64 ≈ 0.0315 > 1/61 ≈ 0.0164


def test_disjoint_lists_interleave() -> None:
    order = [d for d, _ in rrf_fuse([["a", "b"], ["c", "d"]], k=60)]
    assert order == ["a", "c", "b", "d"]  # equal-rank ties keep first-seen order


def test_empty_inputs() -> None:
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []
    assert [d for d, _ in rrf_fuse([["a"], []])] == ["a"]


def test_scores_sorted_descending() -> None:
    fused = rrf_fuse([["a", "b", "c", "d"], ["d", "c", "b", "a"], ["b"]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0][0] == "b"
