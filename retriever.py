"""Retrieval: semantic (Chroma), keyword (BM25), Reciprocal Rank Fusion, reranking.

ChromaDB is the single source of truth. The BM25 index is rebuilt in memory
from the chunks stored in Chroma every time the collection changes, so the two
can never drift apart and nothing is pickled to disk.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

import config
from ingest import Chunk

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import CrossEncoder, SentenceTransformer


class EmptyIndexError(Exception):
    """Raised when searching an index that has no chunks."""


class Mode(str, Enum):
    SEMANTIC = "Semantic only"
    BM25 = "BM25 only"
    HYBRID = "Hybrid (no rerank)"
    HYBRID_RERANK = "Hybrid + Rerank"


# ---------------------------------------------------------------------------
# BM25 tokenizer
# ---------------------------------------------------------------------------
# A token is a run of letters/digits, optionally joined to further runs by a
# single '-', '.', '_' or '/'. So "E-4021", "ISO-9001", "v2.3", "R-513A" and
# "snake_case" stay whole, while sentence punctuation ("end.", "(E-4021),") is
# dropped because a joiner must be followed by another alphanumeric run.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-._/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into BM25 tokens, keeping codes/identifiers intact."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def rrf_fuse(ranked_lists: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank_d), rank 1-based.

    Only ranks matter, not raw scores, which is why RRF can combine cosine
    similarities and BM25 scores that live on completely different scales.
    Ties are broken by first appearance to keep the output deterministic.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    order = {d: i for i, d in enumerate(scores)}  # insertion order = first appearance
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    page: int
    text: str
    scores: dict[str, float | None] = field(
        default_factory=lambda: {"semantic": None, "bm25": None, "rrf": None, "rerank": None}
    )
    # 1-based rank in each stage's list (None = not in that list)
    ranks: dict[str, int | None] = field(
        default_factory=lambda: {"semantic": None, "bm25": None, "rrf": None, "rerank": None}
    )


class HybridIndex:
    """One Chroma collection + an in-memory BM25 index derived from it.

    Isolation note: all ephemeral Chroma clients in a process share one backend,
    so each index gets a random collection name and is only reachable through
    the Python object holding it. Nothing ever lists or looks up collections.
    """

    def __init__(
        self,
        client: "ClientAPI",
        embedder: "SentenceTransformer",
        reranker: "CrossEncoder | None" = None,
        name_prefix: str = "idx",
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._reranker = reranker
        self.name = f"{name_prefix}_{uuid.uuid4().hex}"
        self._col: "Collection" = client.create_collection(
            name=self.name,
            embedding_function=None,  # we embed ourselves, with the query prefix
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._docs: dict[str, tuple[str, dict]] = {}  # id -> (text, metadata)

    # -- writes -------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed and store chunks, then rebuild BM25 from Chroma. Returns count added."""
        if not chunks:
            return 0
        existing = set(self._col.get(ids=[c.chunk_id for c in chunks], include=[])["ids"])
        new = [c for c in chunks if c.chunk_id not in existing]
        for i in range(0, len(new), batch_size):
            batch = new[i : i + batch_size]
            vecs = self._embedder.encode(
                [c.text for c in batch], normalize_embeddings=True, show_progress_bar=False
            )
            self._col.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=vecs.tolist(),
            )
        self._rebuild_bm25()
        return len(new)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 from whatever Chroma holds right now (single source of truth)."""
        data = self._col.get(include=["documents", "metadatas"])
        self._bm25_ids = list(data["ids"])
        self._docs = {
            i: (doc, meta)
            for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        corpus = [tokenize(doc) for doc in data["documents"]]
        # BM25Okapi divides by corpus size, so only build it when non-empty.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def delete(self) -> None:
        """Drop the underlying collection (frees memory for a session's uploads)."""
        try:
            self._client.delete_collection(self.name)
        finally:
            self._bm25, self._bm25_ids, self._docs = None, [], {}

    # -- reads --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._bm25_ids)

    @property
    def sources(self) -> list[str]:
        return sorted({str(meta["source"]) for _, meta in self._docs.values()})

    def semantic_search(self, query: str, k: int = config.SEM_TOP_K) -> list[tuple[str, float]]:
        """Top-k by cosine similarity. Returns (chunk_id, similarity)."""
        self._require_nonempty()
        qvec = self._embedder.encode(
            config.EMBED_QUERY_PREFIX + query, normalize_embeddings=True, show_progress_bar=False
        )
        res = self._col.query(
            query_embeddings=[qvec.tolist()], n_results=min(k, len(self)), include=["distances"]
        )
        # Chroma's cosine "distance" is 1 - cosine similarity.
        return [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def bm25_search(self, query: str, k: int = config.BM25_TOP_K) -> list[tuple[str, float]]:
        """Top-k by BM25. Chunks with score <= 0 (no shared terms) are dropped."""
        self._require_nonempty()
        assert self._bm25 is not None
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def search(
        self,
        query: str,
        mode: Mode = Mode.HYBRID_RERANK,
        final_k: int = config.FINAL_TOP_K,
        rrf_k: int = config.RRF_K,
    ) -> list[RetrievedChunk]:
        """Run the retrieval pipeline for the chosen mode and return the top chunks."""
        self._require_nonempty()
        results: dict[str, RetrievedChunk] = {}

        def get(cid: str) -> RetrievedChunk:
            if cid not in results:
                text, meta = self._docs[cid]
                results[cid] = RetrievedChunk(cid, str(meta["source"]), int(meta["page"]), text)
            return results[cid]

        sem: list[tuple[str, float]] = []
        kw: list[tuple[str, float]] = []
        if mode in (Mode.SEMANTIC, Mode.HYBRID, Mode.HYBRID_RERANK):
            sem = self.semantic_search(query)
            for r, (cid, s) in enumerate(sem, 1):
                get(cid).scores["semantic"], get(cid).ranks["semantic"] = s, r
        if mode in (Mode.BM25, Mode.HYBRID, Mode.HYBRID_RERANK):
            kw = self.bm25_search(query)
            for r, (cid, s) in enumerate(kw, 1):
                get(cid).scores["bm25"], get(cid).ranks["bm25"] = s, r

        if mode is Mode.SEMANTIC:
            ordered = [cid for cid, _ in sem]
        elif mode is Mode.BM25:
            ordered = [cid for cid, _ in kw]
        else:
            fused = rrf_fuse([[c for c, _ in sem], [c for c, _ in kw]], k=rrf_k)
            for r, (cid, s) in enumerate(fused, 1):
                get(cid).scores["rrf"], get(cid).ranks["rrf"] = s, r
            ordered = [cid for cid, _ in fused]
            if mode is Mode.HYBRID_RERANK and ordered:
                ordered = self._rerank(query, [get(c) for c in ordered])

        return [results[cid] for cid in ordered[:final_k]]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[str]:
        """Score every fused candidate with the cross-encoder and reorder."""
        if self._reranker is None:
            raise RuntimeError("Hybrid + Rerank mode needs a reranker model.")
        scores = self._reranker.predict(
            [(query, c.text) for c in candidates], show_progress_bar=False
        )
        ranked = sorted(zip(candidates, scores), key=lambda cs: -float(cs[1]))
        for r, (c, s) in enumerate(ranked, 1):
            c.scores["rerank"], c.ranks["rerank"] = float(s), r
        return [c.chunk_id for c, _ in ranked]

    def _require_nonempty(self) -> None:
        if len(self) == 0:
            raise EmptyIndexError("The index is empty. Index some documents first.")


# ---------------------------------------------------------------------------
# Model / client loaders (wrapped in st.cache_resource by the app)
# ---------------------------------------------------------------------------
def load_embedder() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


def load_reranker() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL, device="cpu")


def make_client() -> "ClientAPI":
    """In-memory Chroma client (no disk persistence, telemetry off)."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
