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

Usage:
    python eval.py                          # sample docs in data/, writes eval_results.md
    python eval.py --data-dir tests/fixtures --eval-set tests/fixtures/aurora_eval.json --out -
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    raw = json.loads(path.read_text(encoding="utf-8"))
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


def validate_labels(items: list[EvalItem], chunks: list[Chunk]) -> None:
    """Fail loudly if a label matches no chunk at all (typo in phrase/page/source)."""
    bad = [it.question for it in items if not any(is_relevant(c, it) for c in chunks)]
    if bad:
        raise SystemExit("Labels match no chunk (fix eval set):\n  - " + "\n  - ".join(bad))


def first_relevant_rank(results: list[RetrievedChunk], item: EvalItem) -> int | None:
    for rank, c in enumerate(results, start=1):
        if is_relevant(c, item):
            return rank
    return None


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--eval-set", type=Path, default=config.DATA_DIR / "eval_set.json")
    ap.add_argument("--out", default=str(config.ROOT_DIR / "eval_results.md"),
                    help="markdown output path, or '-' for stdout only")
    ap.add_argument("-k", type=int, default=config.FINAL_TOP_K)
    args = ap.parse_args()

    t0 = time.perf_counter()
    items = load_eval_set(args.eval_set)
    eval_hash = hashlib.sha256(args.eval_set.read_bytes()).hexdigest()[:16]
    chunks = ingest_sample_dir(args.data_dir)
    if not chunks:
        raise SystemExit(f"No supported documents found in {args.data_dir}")
    validate_labels(items, chunks)

    index = HybridIndex(make_client(), load_embedder(), load_reranker(), name_prefix="eval")
    index.add_chunks(chunks)
    print(f"Indexed {len(index)} chunks from {len(index.sources)} file(s) "
          f"in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

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
