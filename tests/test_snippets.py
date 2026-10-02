"""Snippet and highlighting helpers used by the chunk cards and citation dialog (pure functions)."""

from __future__ import annotations

from app import highlight, make_snippet, md_escape, query_terms, term_spans


def test_query_terms_drop_stopwords_keep_codes() -> None:
    assert query_terms("What does P-4021 indicate on the display?") == {"p-4021", "indicate", "display"}


def test_term_spans_match_whole_codes_and_parts() -> None:
    text = "Fault P-4021: check the flow-switch wiring."
    hits = {t for _, _, t in term_spans(text, {"p-4021", "switch"})}
    assert hits == {"p-4021", "switch"}
    # A different code that merely shares digits is not a whole-token match
    assert term_spans("E-4021 is different", {"p-4021"}) == []


def test_md_escape_neutralises_markdown_and_streamlit_syntax() -> None:
    raw = "*bold* _x_ [link](u) $5 :red[hi] # h <b>"
    escaped = md_escape(raw)
    for ch in "*_[]()$:#<>":
        assert f"\\{ch}" in escaped
    assert "**" not in escaped


def test_snippet_short_text_is_whole_and_bolded() -> None:
    out = make_snippet("Reset the pump after E-1011.", "How do I reset E-1011?")
    assert out.startswith("**Reset** the pump") and "**E\\-1011**" in out
    assert "…" not in out


def test_snippet_centres_on_best_keyword_cluster() -> None:
    filler = "lorem ipsum dolor sit amet " * 40  # ~1080 chars of noise
    text = filler + "The condenser fan relay K3 controls the condenser fan speed. " + filler
    out = make_snippet(text, "condenser fan relay", width=200)
    assert out.startswith("…") and out.endswith("…")
    assert "**condenser** **fan** **relay**" in out
    plain = out.replace("**", "").replace("\\", "")
    assert len(plain) <= 200 + 2  # width plus the two ellipses


def test_snippet_without_hits_starts_at_beginning() -> None:
    text = "alpha beta gamma " * 50
    out = make_snippet(text, "zeta", width=100)
    assert out.startswith("alpha") and out.endswith("…") and "**" not in out


def test_highlight_keeps_lines_and_escapes() -> None:
    out = highlight("Line one: E-4021\n*Line* two", "E-4021")
    assert out == "Line one\\: **E\\-4021**  \n\\*Line\\* two"
