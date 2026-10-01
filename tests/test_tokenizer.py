"""BM25 tokenizer: codes, identifiers and hyphenated terms must stay intact."""

from __future__ import annotations

import pytest

from retriever import tokenize


@pytest.mark.parametrize(
    "text, expected_token",
    [
        ("Fault E-4021 occurred", "e-4021"),
        ("Certified to ISO-9001.", "iso-9001"),
        ("Firmware v2.3 is required", "v2.3"),
        ("Recharge with R-513A only", "r-513a"),
        ("Plan AX-EXT-5 covers parts", "ax-ext-5"),
        ("set max_retries in config", "max_retries"),
        ("see docs/api for details", "docs/api"),
        ("state-of-the-art design", "state-of-the-art"),
        ("IP 192.168.0.1 is reserved", "192.168.0.1"),
    ],
)
def test_identifiers_stay_whole(text: str, expected_token: str) -> None:
    assert expected_token in tokenize(text)


def test_lowercases() -> None:
    assert tokenize("The QUICK Brown") == ["the", "quick", "brown"]


def test_trailing_and_surrounding_punctuation_dropped() -> None:
    assert tokenize("(E-4021), then v2.3. Done!") == ["e-4021", "then", "v2.3", "done"]


def test_dangling_joiners_are_not_tokens() -> None:
    assert tokenize("pre- and post-war -- a - b") == ["pre", "and", "post-war", "a", "b"]


def test_query_and_document_tokens_match() -> None:
    doc = tokenize("Repeated E-4021 faults indicate a failing EXV-88 valve.")
    query = tokenize("what does e-4021 mean?")
    assert "e-4021" in set(doc) & set(query)


def test_empty_and_symbol_only() -> None:
    assert tokenize("") == []
    assert tokenize("?!... --- ///") == []
