"""Retrieval-only evaluation: Hit Rate@K and MRR@K for all four retrieval modes.

No LLM calls are made, so running it costs nothing.

Labels live in a JSON file (default: data/eval_set.json):
    [{"question": "...", "type": "conceptual" | "exact_term" | "auto",
      "author": "project" | "independent",
      "expected": [{"source": "file.pdf", "page": 3, "phrase": "verbatim snippet"}]}]
"auto" applies a fixed rule instead of human judgement: a question containing a
digit (a code, part number, clause or figure) is exact_term, otherwise conceptual.
A retrieved chunk is relevant if its source and page match an expected entry AND
it contains the phrase (case/whitespace-insensitive). Labelling by phrase instead
of chunk_id keeps the eval valid if chunk size or overlap change.

Items with "type": "unanswerable" (no expected pages) are skipped here; they are
for the end-to-end eval (evals/eval_e2e.py).

--ci adds Hit@1 and Recall@10, bootstrap 95% confidence intervals (resampling
questions) and paired comparisons (paired-bootstrap CI of the difference plus
McNemar's exact test). If a file "<eval set>.sha256" exists, the eval set must
match that frozen hash or the script refuses to run.

Usage:
    python eval.py                          # sample docs in data/, writes eval_results.md
    python eval.py --data-dir tests/fixtures --eval-set tests/fixtures/aurora_eval.json --out -
    python eval.py --eval-set evals/eval_set_v2.json --out evals/retrieval_v2.md --ci
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np

import config
from ingest import Chunk, ingest_sample_dir
from retriever import HybridIndex, Mode, RetrievedChunk, load_embedder, load_reranker, make_client

MODES: list[Mode] = [Mode.SEMANTIC, Mode.BM25, Mode.HYBRID, Mode.HYBRID_RERANK]
TYPES: list[str] = ["conceptual", "exact_term"]


@dataclass
class Expected:
    source: str
    page: int
    phrase: str


@dataclass
class EvalItem:
    question: str
    type: str
    expected: list[Expected]
    author: str = "project"


def auto_type(question: str) -> str:
    """Deterministic type rule: any digit in the question => exact_term."""
    return "exact_term" if re.search(r"\d", question) else "conceptual"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def is_relevant(chunk: RetrievedChunk | Chunk, item: EvalItem) -> bool:
    text = _norm(chunk.text)
    return any(
        chunk.source.lower() == e.source.lower() and chunk.page == e.page and _norm(e.phrase) in text
        for e in item.expected
    )


def load_eval_set(path: Path) -> list[EvalItem]:
    """Answerable items only (unanswerable ones have no pages to retrieve)."""
    raw = [r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("type") != "unanswerable"]
    items = [
        EvalItem(
            r["question"],
            auto_type(r["question"]) if r["type"] == "auto" else r["type"],
            [Expected(**e) for e in r["expected"]],
            r.get("author", "project"),
        )
        for r in raw
    ]
    for it in items:
        if it.type not in TYPES:
            raise ValueError(f"Unknown question type {it.type!r} in: {it.question}")
    return items


def entry_matches(chunk: RetrievedChunk | Chunk, e: Expected) -> bool:
    return chunk.source.lower() == e.source.lower() and chunk.page == e.page         and _norm(e.phrase) in _norm(chunk.text)


def validate_labels(items: list[EvalItem], chunks: list[Chunk], every_entry: bool = False) -> None:
    """Fail loudly if a label matches no chunk at all (typo in phrase/page/source).

    every_entry=True is stricter: each expected (page, phrase) must match some chunk,
    not just one of them (used for multi-page questions in eval v2).
    """
    if every_entry:
        bad = [f"{it.question}  ->  p.{e.page}: {e.phrase!r}" for it in items for e in it.expected
               if not any(entry_matches(c, e) for c in chunks)]
    else:
        bad = [it.question for it in items if not any(is_relevant(c, it) for c in chunks)]
    if bad:
        raise SystemExit("Labels match no chunk (fix eval set):\n  - " + "\n  - ".join(bad))


def first_relevant_rank(results: list[RetrievedChunk], item: EvalItem) -> int | None:
    for rank, c in enumerate(results, start=1):
        if is_relevant(c, item):
            return rank
    return None


def recall_at(results: list[RetrievedChunk], item: EvalItem) -> float:
    """Share of the item's expected (page, phrase) entries found anywhere in `results`."""
    found = sum(any(entry_matches(c, e) for c in results) for e in item.expected)
    return found / len(item.expected)


def check_frozen_hash(path: Path) -> str:
    """Full SHA-256 of the eval set; must equal '<path>.sha256' when that file exists."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = path.with_name(path.name + ".sha256")
    if lock.exists():
        frozen = lock.read_text(encoding="utf-8").split()[0].strip().lower()
        if frozen != digest:
            raise SystemExit(f"{path} does not match its frozen hash in {lock.name} "
                             f"({digest[:16]} != {frozen[:16]}). Refusing to run.")
    return digest


# ---------------------------------------------------------------------------
# Statistics for --ci: bootstrap confidence intervals and paired tests
# ---------------------------------------------------------------------------
BOOT_N = 10_000
BOOT_SEED = 2026


def bootstrap_indices(n: int, b: int = BOOT_N, seed: int = BOOT_SEED) -> np.ndarray:
    """(b, n) question indices resampled with replacement; reused for every metric (paired)."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def bootstrap_ci(values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Percentile 95% CI of the mean of `values` over resampled question sets."""
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired hits. Returns (a-only, b-only, p-value)."""
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n
    return a_only, b_only, min(1.0, 2 * tail)


def summarise(ranks: list[int | None]) -> tuple[float, float]:
    """(Hit Rate@K, MRR@K) given the first relevant rank per question (None = miss)."""
    if not ranks:
        return 0.0, 0.0
    hit = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    return hit, mrr


def _metric_table(items: list[EvalItem], ranks: dict[Mode, list[int | None]], k: int) -> list[str]:
    lines = [
        f"| Mode | Hit@{k} | MRR@{k} | Hit@{k} conceptual | MRR@{k} conceptual "
        f"| Hit@{k} exact-term | MRR@{k} exact-term |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        row = [mode.value]
        row += [f"{v:.2f}" for v in summarise(ranks[mode])]
        for t in TYPES:
            sub = [r for r, it in zip(ranks[mode], items) if it.type == t]
            row += [f"{v:.2f}" for v in summarise(sub)] if sub else ["n/a", "n/a"]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(
    items: list[EvalItem],
    ranks: dict[Mode, list[int | None]],
    n_chunks: int,
    sources: list[str],
    k: int,
    eval_hash: str,
) -> str:
    n_by_type = {t: sum(it.type == t for it in items) for t in TYPES}
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Corpus: {', '.join(sources)} ({n_chunks} chunks of ~{config.CHUNK_TOKENS} tokens)",
        f"- Questions: {len(items)} ({n_by_type['conceptual']} conceptual / "
        f"{n_by_type['exact_term']} exact-term), hand-labeled",
        f"- Metrics at K={k}. Semantic/BM25 each fetch top {config.SEM_TOP_K}/{config.BM25_TOP_K}; "
        f"RRF k={config.RRF_K}; reranker `{config.RERANK_MODEL}`",
        f"- Eval set SHA-256 (first 16 hex): `{eval_hash}`",
        "",
        "## All questions",
        "",
        *_metric_table(items, ranks, k),
    ]
    authors = sorted({it.author for it in items})
    if len(authors) > 1:
        for author in authors:
            idx = [i for i, it in enumerate(items) if it.author == author]
            sub_items = [items[i] for i in idx]
            sub_ranks = {m: [ranks[m][i] for i in idx] for m in MODES}
            lines += ["", f"## {author.capitalize()} questions ({len(idx)})", "",
                      *_metric_table(sub_items, sub_ranks, k)]

    lines += [
        "",
        "## Per-question rank of first relevant chunk (– = not in top K)",
        "",
        "| # | Author | Type | Question | " + " | ".join(m.value for m in MODES) + " |",
        "|---|---|---|---|" + "---|" * len(MODES),
    ]
    for i, it in enumerate(items):
        cells = [str(ranks[m][i]) if ranks[m][i] is not None else "–" for m in MODES]
        q = it.question.replace("|", "\\|")
        lines.append(f"| {i + 1} | {it.author} | {it.type} | {q} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "_Small eval set: \"project\" questions were written and labeled by the project author; "
        "\"independent\" questions were written by a separate agent from randomly drawn pages "
        "(seed 2026). Treat differences of one or two questions as noise._",
        "",
    ]
    return "\n".join(lines)


# Pre-specified (before any v2 run) paired comparisons: (baseline, challenger).
COMPARISONS: list[tuple[Mode, Mode]] = [
    (Mode.SEMANTIC, Mode.HYBRID),
    (Mode.HYBRID, Mode.HYBRID_RERANK),
    (Mode.SEMANTIC, Mode.HYBRID_RERANK),
]
RECALL_K = 10


def per_question_metrics(rank10: list[int | None], recall10: list[float], k: int) -> dict[str, np.ndarray]:
    return {
        "Hit@1": np.array([r == 1 for r in rank10], dtype=float),
        f"Hit@{k}": np.array([r is not None and r <= k for r in rank10], dtype=float),
        f"MRR@{k}": np.array([1.0 / r if r is not None and r <= k else 0.0 for r in rank10]),
        f"Recall@{RECALL_K}": np.array(recall10, dtype=float),
    }


def _ci_table(metrics: dict[Mode, dict[str, np.ndarray]], sel: np.ndarray) -> list[str]:
    names = list(next(iter(metrics.values())).keys())
    idx = bootstrap_indices(len(sel))
    lines = ["| Mode | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for mode in MODES:
        cells = []
        for name in names:
            v = metrics[mode][name][sel]
            lo, hi = bootstrap_ci(v, idx)
            cells.append(f"{v.mean():.2f} [{lo:.2f}–{hi:.2f}]")
        lines.append(f"| {mode.value} | " + " | ".join(cells) + " |")
    return lines


def render_markdown_ci(
    items: list[EvalItem],
    rank10: dict[Mode, list[int | None]],
    recall10: dict[Mode, list[float]],
    n_chunks: int,
    sources: list[str],
    k: int,
    eval_hash: str,
) -> str:
    n = len(items)
    metrics = {m: per_question_metrics(rank10[m], recall10[m], k) for m in MODES}
    n_by_type = {t: sum(it.type == t for it in items) for t in TYPES}
    lines = [
        "# Retrieval evaluation v2 (with confidence intervals)",
        "",
        f"- Corpus: {', '.join(sources)} ({n_chunks} chunks of ~{config.CHUNK_TOKENS} tokens)",
        f"- Answerable questions: {n} ({n_by_type['conceptual']} conceptual / "
        f"{n_by_type['exact_term']} exact-term; type set by rule: a digit means exact-term)",
        f"- Semantic/BM25 each fetch top {config.SEM_TOP_K}/{config.BM25_TOP_K}; RRF k={config.RRF_K}; "
        f"reranker `{config.RERANK_MODEL}`. Hit/MRR at K={k} (what the LLM sees); Recall at {RECALL_K}.",
        f"- Eval set SHA-256: `{eval_hash}`",
        f"- Brackets are bootstrap 95% confidence intervals ({BOOT_N:,} resamples of the questions, "
        f"seed {BOOT_SEED}).",
        "",
        "## All answerable questions",
        "",
        *_ci_table(metrics, np.arange(n)),
    ]
    for t in TYPES:
        sel = np.array([i for i, it in enumerate(items) if it.type == t])
        if len(sel):
            lines += ["", f"## {t.replace('_', '-').capitalize()} questions ({len(sel)})", "",
                      *_ci_table(metrics, sel)]

    idx = bootstrap_indices(n)
    lines += [
        "",
        "## Paired comparisons (same questions, both modes)",
        "",
        "Difference = challenger − baseline, with a paired-bootstrap 95% CI. McNemar's exact test "
        "uses only the questions where exactly one mode hits; p < 0.05 means the difference is "
        "unlikely to be chance. Six McNemar tests are run, so a cautious (Bonferroni) threshold "
        "is p < 0.008.",
        "",
        "| Baseline → challenger | Metric | Baseline | Challenger | Difference [95% CI] "
        "| Only baseline hit / only challenger hit | McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, b in COMPARISONS:
        for name in ("Hit@1", f"Hit@{k}", f"MRR@{k}"):
            va, vb = metrics[a][name], metrics[b][name]
            lo, hi = bootstrap_ci(vb - va, idx)
            if name.startswith("Hit"):
                ao, bo, pval = mcnemar_exact(list(va > 0), list(vb > 0))
                test = f"{ao} / {bo} | {pval:.3f}"
            else:
                test = "– | –"
            lines.append(f"| {a.value} → {b.value} | {name} | {va.mean():.2f} | {vb.mean():.2f} "
                         f"| {vb.mean() - va.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {test} |")

    lines += [
        "",
        f"## Per-question rank of first relevant chunk in the top {RECALL_K} (– = not found)",
        "",
        "| # | Author | Type | Question | Pages | " + " | ".join(m.value for m in MODES) + " |",
        "|---|---|---|---|---|" + "---|" * len(MODES),
    ]
    for i, it in enumerate(items):
        cells = [str(rank10[m][i]) if rank10[m][i] is not None else "–" for m in MODES]
        q = it.question.replace("|", "\\|")
        pages = ", ".join(str(e.page) for e in it.expected)
        lines.append(f"| {i + 1} | {it.author} | {it.type} | {q} | {pages} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--eval-set", type=Path, default=config.DATA_DIR / "eval_set.json")
    ap.add_argument("--out", default=str(config.ROOT_DIR / "eval_results.md"),
                    help="markdown output path, or '-' for stdout only")
    ap.add_argument("-k", type=int, default=config.FINAL_TOP_K)
    ap.add_argument("--ci", action="store_true",
                    help="add Hit@1, Recall@10, bootstrap CIs and paired tests")
    args = ap.parse_args()

    t0 = time.perf_counter()
    items = load_eval_set(args.eval_set)
    full_hash = check_frozen_hash(args.eval_set)
    eval_hash = full_hash[:16]
    chunks = ingest_sample_dir(args.data_dir)
    if not chunks:
        raise SystemExit(f"No supported documents found in {args.data_dir}")
    validate_labels(items, chunks, every_entry=args.ci)

    index = HybridIndex(make_client(), load_embedder(), load_reranker(), name_prefix="eval")
    index.add_chunks(chunks)
    print(f"Indexed {len(index)} chunks from {len(index.sources)} file(s) "
          f"in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    if args.ci:
        rank10: dict[Mode, list[int | None]] = {m: [] for m in MODES}
        recall10: dict[Mode, list[float]] = {m: [] for m in MODES}
        for it in items:
            for mode in MODES:
                res = index.search(it.question, mode, final_k=max(RECALL_K, args.k))
                rank10[mode].append(first_relevant_rank(res, it))
                recall10[mode].append(recall_at(res[:RECALL_K], it))
        md = render_markdown_ci(items, rank10, recall10, len(index), index.sources, args.k, full_hash)
    else:
        ranks: dict[Mode, list[int | None]] = {m: [] for m in MODES}
        for it in items:
            for mode in MODES:
                ranks[mode].append(first_relevant_rank(index.search(it.question, mode, final_k=args.k), it))
        md = render_markdown(items, ranks, len(index), index.sources, args.k, eval_hash)
    print(md)
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Saved to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
