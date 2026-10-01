"""Document loading and token-aware chunking.

Pipeline: bytes -> pages (text + 1-based page number) -> ~350-token chunks.
Chunk length is measured with the embedding model's own tokenizer, so a chunk
can never be silently truncated by the embedder.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

import config


class IngestError(Exception):
    """User-facing ingestion failure (unsupported file, no text, corrupt PDF...)."""


@dataclass(frozen=True)
class Page:
    source: str
    page: int  # 1-based physical page index (TXT/MD files are a single page 1)
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str

    @property
    def metadata(self) -> dict[str, str | int]:
        return {"source": self.source, "page": self.page, "chunk_id": self.chunk_id}


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    """The embedding model's tokenizer (cached; downloaded once from HF Hub)."""
    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def _clean(text: str) -> str:
    """Normalise whitespace that PDF extraction tends to mangle."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_running_lines(page_texts: list[str], min_share: float = 0.5) -> list[str]:
    """Remove running headers/footers: lines that repeat on most pages.

    Two rules, so body text is never touched by accident:
    - a line that repeats *exactly* on most pages is removed anywhere;
    - a line that repeats only after masking digits ("Page 3 of 37" vs
      "Page 4 of 37") is removed only from the first/last line of a page.
      Masking everywhere would wrongly delete body lines such as
      "E-1011 Fault / Reset: Auto", which differ only in their digits.
    Only applied to documents with at least 3 pages.
    """
    if len(page_texts) < 3:
        return page_texts
    edge = 1  # page numbers sit on the very first or last line

    def exact(line: str) -> str:
        return line.strip().lower()

    def masked(line: str) -> str:
        return re.sub(r"\d+", "#", exact(line))

    pages = [[ln for ln in text.splitlines() if ln.strip()] for text in page_texts]
    exact_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for lines in pages:
        for k in {exact(ln) for ln in lines}:
            exact_counts[k] = exact_counts.get(k, 0) + 1
        for k in {masked(ln) for ln in lines[:edge] + lines[-edge:]}:
            edge_counts[k] = edge_counts.get(k, 0) + 1
    threshold = max(3, int(len(page_texts) * min_share))
    running_exact = {k for k, n in exact_counts.items() if n >= threshold}
    running_edge = {k for k, n in edge_counts.items() if n >= threshold}

    out: list[str] = []
    for lines in pages:
        kept = []
        for i, ln in enumerate(lines):
            at_edge = i < edge or i >= len(lines) - edge
            if exact(ln) in running_exact or (at_edge and masked(ln) in running_edge):
                continue
            kept.append(ln)
        out.append("\n".join(kept))
    return out


def load_document(name: str, data: bytes) -> list[Page]:
    """Extract text per page from a PDF, TXT or MD file given as raw bytes."""
    source = Path(name).name  # never trust a client-supplied path
    ext = Path(source).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"'{source}': unsupported file type '{ext or '(none)'}'. "
            f"Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}"
        )

    pages: list[Page] = []
    if ext == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise IngestError(f"'{source}': encrypted PDFs are not supported.")
            raw = [_clean(p.extract_text() or "") for p in reader.pages]
            for i, text in enumerate(strip_running_lines(raw), start=1):
                text = text.strip()
                if text:
                    pages.append(Page(source, i, text))
        except IngestError:
            raise
        except (PdfReadError, ValueError, KeyError, TypeError) as exc:
            raise IngestError(f"'{source}': could not read PDF ({type(exc).__name__}).") from exc
        if not pages:
            raise IngestError(
                f"'{source}': no extractable text. It may be a scanned/image-only PDF "
                "(OCR is not supported)."
            )
    else:
        text = _clean(data.decode("utf-8", errors="replace"))
        if not text:
            raise IngestError(f"'{source}': file is empty.")
        pages.append(Page(source, 1, text))
    return pages


def chunk_pages(
    pages: list[Page],
    chunk_tokens: int = config.CHUNK_TOKENS,
    overlap: int = config.CHUNK_OVERLAP,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[Chunk]:
    """Split each page into overlapping token windows.

    Windows are cut in token space, then mapped back to the ORIGINAL text via
    character offsets, so chunk text is an exact substring of the page (no
    decode artefacts). Chunks never span pages, which keeps page citations exact.
    """
    if not 0 <= overlap < chunk_tokens:
        raise ValueError("overlap must be >= 0 and smaller than chunk_tokens")
    tok = tokenizer or get_tokenizer()
    stride = chunk_tokens - overlap
    chunks: list[Chunk] = []

    for page in pages:
        enc = tok(
            page.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,  # silence "sequence longer than max length" warning
        )
        offsets: list[tuple[int, int]] = enc["offset_mapping"]
        n = len(offsets)
        start, idx = 0, 0
        while start < n:
            end = min(start + chunk_tokens, n)
            text = page.text[offsets[start][0] : offsets[end - 1][1]].strip()
            if text:
                cid = f"{page.source}::p{page.page}::c{idx}"
                chunks.append(Chunk(cid, page.source, page.page, text))
                idx += 1
            if end == n:
                break
            start += stride
    return chunks


def assert_chunk_lengths(
    chunks: list[Chunk], tokenizer: PreTrainedTokenizerBase | None = None
) -> int:
    """Assert no chunk (incl. [CLS]/[SEP]) exceeds the model's max sequence length.

    Returns the longest chunk length in tokens.
    """
    tok = tokenizer or get_tokenizer()
    max_len = tok.model_max_length  # 512 for bge-small-en-v1.5
    longest = 0
    for c in chunks:
        n = len(tok(c.text, add_special_tokens=True, verbose=False)["input_ids"])
        assert n <= max_len, f"{c.chunk_id} has {n} tokens > model max {max_len}"
        longest = max(longest, n)
    return longest


def ingest(name: str, data: bytes) -> list[Chunk]:
    """Load + chunk + validate a single file."""
    chunks = chunk_pages(load_document(name, data))
    assert_chunk_lengths(chunks)
    return chunks


def sample_files(data_dir: Path = config.DATA_DIR) -> list[Path]:
    """Supported documents in /data (the shared, read-only sample corpus)."""
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


def ingest_sample_dir(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    """Chunk every sample document in /data."""
    chunks: list[Chunk] = []
    for path in sample_files(data_dir):
        chunks.extend(ingest(path.name, path.read_bytes()))
    return chunks
