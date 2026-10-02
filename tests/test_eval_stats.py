"""eval.py v2 helpers: McNemar, bootstrap CI, Recall@K, frozen-hash lock, unanswerable skip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval import (
    EvalItem,
    Expected,
    bootstrap_ci,
    bootstrap_indices,
    check_frozen_hash,
    load_eval_set,
    mcnemar_exact,
    recall_at,
)
from retriever import RetrievedChunk


def test_mcnemar_known_values() -> None:
    # 0 discordant pairs -> no evidence of a difference
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)
    # 6 discordant, all one way: p = 2 * 0.5^6 = 0.03125
    a, b = [False] * 6 + [True] * 4, [True] * 6 + [True] * 4
    assert mcnemar_exact(a, b) == (0, 6, pytest.approx(0.03125))
    # 3 vs 3 -> p capped at 1
    assert mcnemar_exact([True] * 3 + [False] * 3, [False] * 3 + [True] * 3)[2] == 1.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    v = np.array([1.0] * 70 + [0.0] * 30)
    idx = bootstrap_indices(len(v), b=2000)
    lo, hi = bootstrap_ci(v, idx)
    assert lo < 0.70 < hi and 0.55 < lo and hi < 0.85
    assert (lo, hi) == bootstrap_ci(v, bootstrap_indices(len(v), b=2000))  # fixed seed
    assert bootstrap_ci(np.ones(10), bootstrap_indices(10, b=100)) == (1.0, 1.0)


def test_recall_counts_each_expected_entry() -> None:
    item = EvalItem("q", "conceptual", [Expected("m.pdf", 3, "alpha beta"), Expected("m.pdf", 9, "gamma")])
    hit3 = RetrievedChunk("c1", "m.pdf", 3, "x Alpha  beta y")
    other = RetrievedChunk("c2", "m.pdf", 9, "no match here")
    assert recall_at([hit3, other], item) == 0.5
    assert recall_at([other], item) == 0.0


def test_frozen_hash_lock(tmp_path: Path) -> None:
    f = tmp_path / "set.json"
    f.write_text("[]", encoding="utf-8")
    digest = check_frozen_hash(f)  # no lock file: just returns the hash
    (tmp_path / "set.json.sha256").write_text(digest + "  set.json\n", encoding="utf-8")
    assert check_frozen_hash(f) == digest
    f.write_text("[ ]", encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen hash"):
        check_frozen_hash(f)


def test_load_eval_set_skips_unanswerable(tmp_path: Path) -> None:
    f = tmp_path / "set.json"
    f.write_text(json.dumps([
        {"question": "Code E-1?", "type": "auto", "expected": [{"source": "m.pdf", "page": 1, "phrase": "x"}]},
        {"question": "Weather?", "type": "unanswerable", "category": "irrelevant", "expected": []},
    ]), encoding="utf-8")
    items = load_eval_set(f)
    assert [(i.question, i.type) for i in items] == [("Code E-1?", "exact_term")]
