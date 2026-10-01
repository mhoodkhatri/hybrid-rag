"""Ingestion tests: chunk length vs. embedding model limit, overlap, metadata, errors."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

import config
from ingest import (
    IngestError,
    Page,
    assert_chunk_lengths,
    chunk_pages,
    get_tokenizer,
    ingest,
    load_document,
    strip_running_lines,
)


def test_fixture_chunks_fit_model_limit(fixture_bytes: bytes) -> None:
    chunks = ingest("aurora_manual.md", fixture_bytes)
    assert len(chunks) >= 2, "fixture should produce several chunks"
    longest = assert_chunk_lengths(chunks)
    assert longest <= get_tokenizer().model_max_length


def test_chunks_never_exceed_limit_even_for_huge_page() -> None:
    # Pathological input: one page of ~6k tokens, including long codes.
    text = " ".join(f"word{i} E-{i:04d} ISO-9001 v2.3" for i in range(1500))
    chunks = chunk_pages([Page("big.txt", 1, text)])
    assert len(chunks) > 10
    tok = get_tokenizer()
    for c in chunks:
        n_content = len(tok(c.text, add_special_tokens=False, verbose=False)["input_ids"])
        assert n_content <= config.CHUNK_TOKENS + 2  # small slack for re-tokenisation at edges
    assert_chunk_lengths(chunks)  # with [CLS]/[SEP] still <= 512


def test_assert_catches_oversized_chunk() -> None:
    long_text = "token " * 2000
    chunks = chunk_pages([Page("x.txt", 1, long_text)], chunk_tokens=1000, overlap=0)
    with pytest.raises(AssertionError):
        assert_chunk_lengths(chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"alpha{i}" for i in range(2000))
    chunks = chunk_pages([Page("o.txt", 1, text)])
    tail = chunks[0].text.split()[-5:]
    assert " ".join(tail) in chunks[1].text


def test_metadata_and_ids(fixture_bytes: bytes) -> None:
    chunks = ingest("../../evil/aurora_manual.md", fixture_bytes)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for c in chunks:
        assert c.source == "aurora_manual.md"  # path components stripped
        assert c.page == 1
        assert c.metadata == {"source": c.source, "page": c.page, "chunk_id": c.chunk_id}


def test_chunk_text_is_exact_substring(fixture_bytes: bytes) -> None:
    page = load_document("a.md", fixture_bytes)[0]
    for c in chunk_pages([page]):
        assert c.text in page.text


def test_running_headers_and_footers_are_stripped() -> None:
    pages = [
        f"ACME Manual | Restricted\nUnique body text {i}\nACME Manual Page {i} of 4" for i in range(1, 5)
    ]
    stripped = strip_running_lines(pages)
    for i, text in enumerate(stripped, start=1):
        assert text == f"Unique body text {i}"


def test_body_lines_differing_only_in_digits_are_kept() -> None:
    # Error-code headings repeat after digit masking but are real content.
    pages = [
        f"Header\nE-{1000 + i} Fault / Reset: Auto\nDescription {i}\nE-{2000 + i} Fault / Reset: Auto\n"
        f"More text\nFooter Page {i} of 6"
        for i in range(1, 7)
    ]
    for i, text in enumerate(strip_running_lines(pages), start=1):
        assert f"E-{1000 + i} Fault / Reset: Auto" in text
        assert f"E-{2000 + i} Fault / Reset: Auto" in text
        assert "Header" not in text and "Footer" not in text


def test_short_documents_keep_all_lines() -> None:
    pages = ["Header\nA", "Header\nB"]
    assert strip_running_lines(pages) == pages


def test_unsupported_extension() -> None:
    with pytest.raises(IngestError, match="unsupported"):
        load_document("malware.exe", b"MZ...")


def test_empty_text_file() -> None:
    with pytest.raises(IngestError, match="empty"):
        load_document("empty.txt", b"   \n\n ")


def test_pdf_without_text_is_rejected() -> None:
    # A PDF with only blank pages behaves like a scanned PDF: no extractable text.
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(IngestError, match="no extractable text"):
        load_document("scan.pdf", buf.getvalue())


def test_corrupt_pdf_is_rejected() -> None:
    with pytest.raises(IngestError):
        load_document("broken.pdf", b"%PDF-1.4 this is not really a pdf")
