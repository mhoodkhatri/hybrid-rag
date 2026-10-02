"""Streamlit UI for the Hybrid RAG demo.

Multi-user model:
- Models, the Chroma client and the sample-document index are process-wide
  singletons (st.cache_resource), shared read-only by every visitor.
- Uploaded documents go into a per-session collection held in the
  SessionRegistry, keyed by this session's id, so one visitor can never query
  another visitor's files. The registry expires idle sessions and caps how many
  are kept, because Streamlit never tells us when a session ends.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import config
from generator import (
    NOT_SUPPORTED,
    PARTLY,
    SUPPORTED,
    GenerationError,
    check_citations,
    make_client,
    rewrite_query,
    stream_answer,
    verify_claims,
)
from ingest import IngestError, ingest, ingest_sample_dir, sample_files
from retriever import (
    _TOKEN_RE,
    EmptyIndexError,
    HybridIndex,
    Mode,
    RetrievedChunk,
    load_embedder,
    load_reranker,
    tokenize,
)
from retriever import make_client as make_chroma_client

SAMPLE = "Sample document"
UPLOADS = "My uploads"

st.set_page_config(page_title="Hybrid RAG", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Per-session uploads, with expiry
# ---------------------------------------------------------------------------
@dataclass
class SessionEntry:
    index: HybridIndex
    files: dict[str, bytes] = field(default_factory=dict)  # name -> raw bytes (page previews)
    last_seen: float = 0.0


class SessionRegistry:
    """Maps session id -> that session's upload index and file bytes.

    Thread-safe (Streamlit runs each session's script in its own thread).
    `sweep()` deletes entries idle longer than the TTL, then evicts the least
    recently used ones beyond `max_sessions`, calling HybridIndex.delete() so
    the Chroma collection's memory is actually released.
    """

    def __init__(self, ttl_s: float, max_sessions: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, SessionEntry] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, sid: str) -> SessionEntry | None:
        with self._lock:
            return self._entries.get(sid)

    def put(self, sid: str, entry: SessionEntry) -> None:
        with self._lock:
            entry.last_seen = self._clock()
            self._entries[sid] = entry
        self.sweep()

    def touch(self, sid: str) -> None:
        with self._lock:
            if sid in self._entries:
                self._entries[sid].last_seen = self._clock()

    def remove(self, sid: str) -> None:
        with self._lock:
            entry = self._entries.pop(sid, None)
        if entry is not None:
            self._drop(entry)

    def sweep(self) -> list[str]:
        """Expire idle entries, then enforce the size cap. Returns removed ids."""
        now = self._clock()
        with self._lock:
            removed = [s for s, e in self._entries.items() if now - e.last_seen > self.ttl_s]
            keep = sorted((s for s in self._entries if s not in removed),
                          key=lambda s: self._entries[s].last_seen, reverse=True)
            removed += keep[self.max_sessions:]
            dropped = [self._entries.pop(s) for s in removed]
        for entry in dropped:  # outside the lock: deleting a collection can be slow
            self._drop(entry)
        return removed

    @staticmethod
    def _drop(entry: SessionEntry) -> None:
        try:
            entry.index.delete()
        except Exception:
            pass
        entry.files.clear()


# ---------------------------------------------------------------------------
# Shared, cached resources (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return load_embedder()


@st.cache_resource(show_spinner=False)
def get_reranker():
    return load_reranker()


@st.cache_resource
def get_chroma():
    return make_chroma_client()


@st.cache_resource
def get_registry() -> SessionRegistry:
    return SessionRegistry(config.SESSION_TTL_MIN * 60, config.MAX_UPLOAD_SESSIONS)


@st.cache_resource
def _startup_state() -> dict:
    """Process-wide flag: have the models and sample index been loaded yet?"""
    return {"ready": False}


@st.cache_resource(show_spinner=False)
def get_sample_index() -> tuple[HybridIndex | None, str | None]:
    """Build the shared read-only index. Returns (index, error message)."""
    try:
        chunks = ingest_sample_dir()
        if not chunks:
            return None, "No sample document found in data/."
        index = HybridIndex(get_chroma(), get_embedder(), get_reranker(), name_prefix="sample")
        index.add_chunks(chunks)
        return index, None
    except (IngestError, OSError, AssertionError) as exc:
        return None, f"Sample document could not be indexed: {exc}"


@st.cache_data
def example_questions() -> list[str]:
    try:
        items = json.loads((config.DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
        by_type = {t: [i["question"] for i in items if i["type"] == t] for t in ("conceptual", "exact_term")}
        return by_type["conceptual"][:2] + by_type["exact_term"][:2]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])  # {"role", "content", "meta"?, "error"?}
    ss.setdefault("query_count", 0)
    ss.setdefault("uploaded_names", [])  # this session's files (index lives in the registry)
    ss.setdefault("pending_question", None)
    ss.setdefault("notices", [])  # (kind, message) shown once in the sidebar after a rerun


def session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is not None:
        return ctx.session_id
    return st.session_state.setdefault("_fallback_sid", uuid.uuid4().hex)


def user_index() -> HybridIndex | None:
    entry = get_registry().get(session_id())
    return entry.index if entry is not None else None


def refresh_uploads() -> None:
    """Mark this session active, sweep idle sessions, and notice if ours expired."""
    ss, registry, sid = st.session_state, get_registry(), session_id()
    registry.touch(sid)
    registry.sweep()
    if ss.uploaded_names and registry.get(sid) is None:
        ss.uploaded_names = []
        ss.notices.append(("warning", f"Your uploaded documents expired after "
                                      f"{config.SESSION_TTL_MIN} min of inactivity — please re-upload. "
                                      "Searching the sample document for now."))


def queries_left() -> int:
    return max(0, config.MAX_QUERIES_PER_SESSION - st.session_state.query_count)


def clear_uploads() -> None:
    get_registry().remove(session_id())
    st.session_state.uploaded_names = []


def load_resources() -> tuple[HybridIndex | None, str | None]:
    """Load models + sample index, showing each step the first time the server starts."""
    state = _startup_state()
    if state["ready"]:
        return get_sample_index()
    with st.status("Starting up (first visit after a restart takes about a minute)…",
                   expanded=True) as status:
        st.write("Loading search model…")
        get_embedder()
        st.write("Loading reranker…")
        get_reranker()
        st.write("Indexing sample manual…")
        result = get_sample_index()
        status.update(label="Ready", state="complete", expanded=False)
    state["ready"] = True
    return result


def index_uploads(files: list) -> None:
    """Validate and index uploaded files into this session's private collection."""
    ss = st.session_state
    registry, sid = get_registry(), session_id()
    new_files = [f for f in files if f.name not in ss.uploaded_names]
    if not new_files:
        ss.notices.append(("info", "These files are already indexed."))
        return
    if len(ss.uploaded_names) + len(new_files) > config.MAX_FILES:
        ss.notices.append(("error", f"Limit is {config.MAX_FILES} files per session. "
                                    "Clear your documents to start over."))
        return

    ok, errors = 0, []
    for f in new_files:
        if f.size > config.MAX_FILE_MB * 1024 * 1024:
            errors.append(f"'{f.name}' exceeds {config.MAX_FILE_MB} MB.")
            continue
        bar = st.sidebar.progress(0.0, text=f"Reading {f.name}…")
        try:
            data = f.getvalue()
            chunks = ingest(f.name, data)
            entry = registry.get(sid)
            current = len(entry.index) if entry is not None else 0
            if current + len(chunks) > config.MAX_CHUNKS_PER_SESSION:
                raise IngestError(f"'{f.name}' is too large for the demo "
                                  f"(>{config.MAX_CHUNKS_PER_SESSION} chunks per session).")
            if entry is None:
                entry = SessionEntry(HybridIndex(get_chroma(), get_embedder(), get_reranker(),
                                                 name_prefix="sess"))
                registry.put(sid, entry)

            def report(done: int, total: int, name: str = f.name) -> None:
                bar.progress(done / total, text=f"Indexing {name}: {done}/{total} chunks")

            entry.index.add_chunks(chunks, progress=report)
            entry.files[f.name] = data
            ss.uploaded_names.append(f.name)
            ok += 1
        except IngestError as exc:
            errors.append(str(exc))
        except Exception as exc:  # never crash the app on a bad file
            errors.append(f"'{f.name}': unexpected error while indexing ({type(exc).__name__}).")
        finally:
            bar.empty()
    entry = registry.get(sid)
    if entry is not None and len(entry.index) == 0:  # every file failed: don't keep an empty slot
        registry.remove(sid)
    ss.notices.extend(("error", e) for e in errors)
    if ok:
        ss.notices.append(("success", f"Indexed {ok} file(s). Now searching “{UPLOADS}”."))
        ss.switch_corpus = UPLOADS  # applied before the radio widget is drawn


# ---------------------------------------------------------------------------
# Text helpers: query-term highlighting and snippets (pure, unit-tested)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a an and any are as at be been by can could did do does for from has have how i if in "
    "is it its me my no not of on or our should so than that the their there these they this "
    "to was we were what when where which who why will with would you your".split()
)
_TOKEN_RE_ANYCASE = re.compile(_TOKEN_RE.pattern, re.IGNORECASE)
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:=])")


def md_escape(text: str) -> str:
    """Backslash-escape markdown/Streamlit syntax so document text renders literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def query_terms(query: str) -> set[str]:
    """The query's content words, using the same tokenizer as BM25."""
    return {t for t in tokenize(query) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())}


def term_spans(text: str, terms: set[str]) -> list[tuple[int, int, str]]:
    """(start, end, term) for every token in `text` matching a query term.

    A token also matches if one of its parts does ("flow-switch" for "switch").
    """
    spans = []
    for m in _TOKEN_RE_ANYCASE.finditer(text):
        tok = m.group().lower()
        hit = tok if tok in terms else next((p for p in re.split(r"[-._/]", tok) if p in terms), None)
        if hit:
            spans.append((m.start(), m.end(), hit))
    return spans


def _render(text: str, spans: list[tuple[int, int, str]], start: int, end: int) -> str:
    out, pos = [], start
    for s, e, _ in spans:
        if s < start or e > end:
            continue
        out += [md_escape(text[pos:s]), "**", md_escape(text[s:e]), "**"]
        pos = e
    out.append(md_escape(text[pos:end]))
    return "".join(out)


def highlight(text: str, query: str) -> str:
    """Whole text as markdown with query terms in bold (line breaks kept)."""
    lines = text.splitlines()
    return "  \n".join(_render(ln, term_spans(ln, query_terms(query)), 0, len(ln)) for ln in lines)


def make_snippet(text: str, query: str, width: int = 350) -> str:
    """~`width` chars of `text`, centred on the densest cluster of query terms, as markdown."""
    flat = re.sub(r"\s+", " ", text).strip()
    spans = term_spans(flat, query_terms(query))
    if len(flat) <= width:
        return _render(flat, spans, 0, len(flat))
    start = 0
    if spans:
        def window(span: tuple[int, int, str]) -> tuple[int, int]:
            s = min(max(0, (span[0] + span[1]) // 2 - width // 2), len(flat) - width)
            return s, s + width

        def score(span: tuple[int, int, str]) -> tuple[int, int]:
            s, e = window(span)
            inside = [t for a, b, t in spans if a >= s and b <= e]
            return len(set(inside)), len(inside)

        start = window(max(spans, key=score))[0]  # max() keeps the first of equal scores
    end = start + width
    if start > 0:  # snap to word boundaries
        start = flat.find(" ", start) + 1 or start
    if end < len(flat) and (cut := flat.rfind(" ", start, end)) > start:
        end = cut
    return ("…" if start > 0 else "") + _render(flat, spans, start, end) + \
        ("…" if end < len(flat) else "")


# ---------------------------------------------------------------------------
# Source pages for the citation dialog
# ---------------------------------------------------------------------------
def source_bytes(source: str, corpus: str) -> bytes | None:
    """Raw bytes of a cited file: sample files from data/, uploads from the registry."""
    name = Path(source).name
    if corpus == SAMPLE:
        path = config.DATA_DIR / name
        return path.read_bytes() if path.is_file() else None
    entry = get_registry().get(session_id())
    return entry.files.get(name) if entry is not None else None


@st.cache_data(max_entries=64, show_spinner=False)
def render_pdf_page(file_hash: str, page: int, _data: bytes) -> bytes | None:
    """PNG of one PDF page (1-based). Cached per (file hash, page); `_data` isn't hashed."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(_data)
    try:
        if not 1 <= page <= len(doc):
            return None
        img = doc[page - 1].render(scale=1.5).to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


@st.dialog("Cited source", width="large")
def show_source(source: str, page: int, corpus: str, passages: list[str], query: str) -> None:
    st.markdown(f"**{md_escape(source)} · page {page}**")
    data = source_bytes(source, corpus)
    if data is None:
        st.warning("This file is no longer available (uploads expire after "
                   f"{config.SESSION_TTL_MIN} min of inactivity). Please re-upload it.")
    elif source.lower().endswith(".pdf"):
        try:
            png = render_pdf_page(hashlib.sha256(data).hexdigest(), page, data)
        except Exception as exc:
            png = None
            st.warning(f"Could not render this page ({type(exc).__name__}).")
        if png:
            st.image(png, use_container_width=True)
    else:
        with st.container(height=380):
            st.text(data.decode("utf-8", errors="replace"))
    st.markdown("**Retrieved passage(s) from this page** (search terms in bold)")
    for text in passages:
        with st.container(border=True):
            st.markdown(highlight(text, query))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
STAGES: list[tuple[str, str]] = [("bm25", "Keyword match"), ("semantic", "Meaning match"),
                                 ("rrf", "Combined"), ("rerank", "Reranker")]
MODE_STAGES: dict[Mode, set[str]] = {
    Mode.SEMANTIC: {"semantic"},
    Mode.BM25: {"bm25"},
    Mode.HYBRID: {"bm25", "semantic", "rrf"},
    Mode.HYBRID_RERANK: {"bm25", "semantic", "rrf", "rerank"},
}
CHIP_COLOURS = {"bm25": "orange", "semantic": "violet", "rrf": "blue", "rerank": "green"}


def _fmt(score: float | None, rank: int | None, digits: int) -> str:
    return "—" if score is None else f"{score:.{digits}f} (#{rank})"


def rank_chips(chunk: dict, mode: Mode) -> str:
    """Plain-language rank badges for the stages that ran in this mode."""
    chips = []
    for key, label in STAGES:
        if key in MODE_STAGES[mode]:
            rank = chunk["ranks"][key]
            chips.append(f":{CHIP_COLOURS[key]}-badge[{label} {'#' + str(rank) if rank else '–'}]")
    return " ".join(chips)


def render_citation_buttons(meta: dict, msg_idx: int) -> None:
    report = meta.get("citations")
    if not report or not report["valid"]:
        return
    query = meta.get("rewritten") or meta["question"]
    many_files = len({s.lower() for s, _ in report["valid"]}) > 1
    with st.container(horizontal=True):
        for j, (src, page) in enumerate(report["valid"]):
            label = f"📄 {src} p.{page}" if many_files else f"📄 p.{page}"
            if st.button(label, key=f"cite_{msg_idx}_{j}", help="Show this page"):
                passages = [c["text"] for c in meta["chunks"]
                            if c["source"].lower() == src.lower() and c["page"] == page]
                show_source(src, page, meta["corpus"], passages, query)


CLAIM_BADGES = {SUPPORTED: ":green-badge[✓ supported]", PARTLY: ":orange-badge[~ partly supported]",
                NOT_SUPPORTED: ":red-badge[✗ not supported]"}


def render_claims(report: dict | None) -> None:
    """One badge per cited sentence, graded by the AI judge, plus an overall line."""
    if not report:
        return
    if report["error"]:
        st.caption(f"🔎 {report['error']} The answer above is unaffected.")
        return
    claims = report["claims"]
    if not claims:
        return
    ok = sum(c["verdict"] == SUPPORTED for c in claims)
    icon = "✅" if ok == len(claims) else "⚠️"
    st.caption(f"{icon} Claim check (AI judge): **{ok}/{len(claims)} claims supported** by the "
               "passages they cite.")
    for c in claims:
        badge = CLAIM_BADGES.get(c["verdict"], ":gray-badge[? unchecked]")
        text = c["text"] if len(c["text"]) <= 140 else c["text"][:137] + "…"
        reason = f" — _{md_escape(c['reason'])}_" if c["reason"] else ""
        st.caption(f"{badge} {md_escape(text)}{reason}")


def render_details(meta: dict, msg_idx: int) -> None:
    render_citation_buttons(meta, msg_idx)
    report = meta.get("citations")
    if report:
        if report["invalid"]:
            bad = ", ".join(f"[{s}, p.{p}]" for s, p in report["invalid"])
            st.warning(f"⚠️ Citation check: {bad} does not match any retrieved chunk.")
        elif report["uncited"]:
            st.warning("⚠️ Citation check: the answer contains no citations.")
        elif report["valid"]:
            st.caption(f"✅ All {len(report['valid'])} citation(s) match retrieved chunks.")
    render_claims(meta.get("claims"))

    mode = Mode(meta["mode"])
    query = meta.get("rewritten") or meta["question"]
    with st.expander(f"Retrieved chunks · {meta['mode']} · {meta['corpus']}"):
        st.caption("Passages the answer could draw on, best first. Badges show where each search "
                   "method ranked a passage (#1 = its top pick, – = it didn't find it). "
                   "Search terms are in bold; **Details** has the full text and raw scores.")
        if meta.get("rewritten") and meta["rewritten"] != meta["question"]:
            st.markdown(f"**Rewritten query:** {md_escape(meta['rewritten'])}")
        if not meta["chunks"]:
            st.info("No chunks retrieved (BM25 found no matching terms?).")
        for i, c in enumerate(meta["chunks"], start=1):
            with st.container(border=True):
                head, pop = st.columns([5, 1], vertical_alignment="center")
                head.markdown(f"**#{i} · p.{c['page']}** · {md_escape(c['source'])}  \n"
                              + rank_chips(c, mode))
                with pop.popover("Details", key=f"pop_{msg_idx}_{i}"):
                    st.markdown("**Full text**")
                    st.text(c["text"])
                    s, r = c["scores"], c["ranks"]
                    st.markdown("**Technical scores** (raw value, rank)")
                    st.markdown(
                        f"- Semantic cosine similarity: {_fmt(s['semantic'], r['semantic'], 3)}\n"
                        f"- BM25: {_fmt(s['bm25'], r['bm25'], 2)}\n"
                        f"- RRF (k={config.RRF_K}): {_fmt(s['rrf'], r['rrf'], 4)}\n"
                        f"- Cross-encoder rerank logit: {_fmt(s['rerank'], r['rerank'], 2)}"
                    )
                st.markdown(make_snippet(c["text"], query))


def render_history() -> None:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("meta"):
                render_details(msg["meta"], idx)


# ---------------------------------------------------------------------------
# Compare modes (retrieval only: no LLM call, doesn't use the question quota)
# ---------------------------------------------------------------------------
TAB_ASK, TAB_COMPARE = "💬 Ask", "⚖️ Compare modes"
COMPARE_TOP = 4
# Mid-tone colours that stay visible on both light and dark backgrounds.
DOT_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
               "#9acd32", "#9a6324", "#469990", "#d4a017", "#ff69b4", "#808000", "#708090",
               "#a0522d", "#6a5acd"]


@st.cache_data
def expected_pages() -> dict[str, set[tuple[str, int]]]:
    """Normalised question -> labelled (source, page) pairs, from the eval sets."""
    out: dict[str, set[tuple[str, int]]] = {}
    for path in (config.DATA_DIR / "eval_set.json", config.ROOT_DIR / "evals" / "eval_set_v2.json"):
        try:
            for item in json.loads(path.read_text(encoding="utf-8")):
                pages = {(e["source"].lower(), int(e["page"])) for e in item.get("expected", [])}
                if pages:
                    out[_norm_q(item["question"])] = pages
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().lower()


def page_colours(results: dict[str, list[dict]]) -> dict[tuple[str, int], str]:
    """Same (source, page) -> same colour in every column; top ranks get colours first."""
    colours: dict[tuple[str, int], str] = {}
    for rank in range(COMPARE_TOP):
        for chunks in results.values():
            if rank < len(chunks):
                key = (chunks[rank]["source"].lower(), chunks[rank]["page"])
                if key not in colours:
                    colours[key] = DOT_COLOURS[len(colours) % len(DOT_COLOURS)]
    return colours


def request_compare(question: str | None = None) -> None:
    ss = st.session_state
    if question is not None:
        ss.cmp_q = question  # allowed: callbacks run before the text box is drawn
    ss.cmp_request = ss.cmp_q


def answer_from_compare() -> None:
    ss = st.session_state
    ss.pending_question = ss.compare["question"]
    ss.mode = Mode.HYBRID_RERANK
    ss.main_tab = TAB_ASK


def render_compare(index: HybridIndex | None, corpus: str) -> None:
    ss = st.session_state
    ss.setdefault("cmp_q", "")
    ss.setdefault("compare_count", 0)
    # Run a requested comparison first, so the counter below is already up to date.
    request = (ss.pop("cmp_request", None) or "").strip()
    problem = None
    if request and ss.compare_count < config.MAX_COMPARES_PER_SESSION:
        if index is None or len(index) == 0:
            problem = "There's nothing to search yet. Upload and index a document first."
        else:
            try:
                with st.spinner("Running all four modes…"):
                    results = {m.value: [asdict(c) for c in index.search(request, m, final_k=COMPARE_TOP)]
                               for m in Mode}
                ss.compare_count += 1
                ss.compare = {"question": request, "corpus": corpus, "results": results}
            except Exception as exc:
                problem = f"Retrieval failed ({type(exc).__name__}). Please try again."
    left = max(0, config.MAX_COMPARES_PER_SESSION - ss.compare_count)

    st.caption("See what each search method finds for the same question. Each column is one "
               f"method's top {COMPARE_TOP}; the same page gets the same coloured dot in every "
               "column, so you can spot where methods agree. ✅ marks the page with the answer "
               "(known for the example questions). This uses no AI calls and doesn't count "
               "toward your question limit.")
    st.text_input("Question to compare", key="cmp_q", max_chars=config.MAX_QUESTION_CHARS,
                  placeholder="e.g. What does P-4021 indicate?")
    with st.container(horizontal=True):
        st.button("Compare", type="primary", on_click=request_compare, disabled=left == 0)
        if corpus == SAMPLE:
            for i, q in enumerate(example_questions()):
                st.button(q, key=f"cmp_example_{i}", on_click=request_compare, args=(q,),
                          disabled=left == 0)
    st.caption(f"Comparisons left this session: {left} / {config.MAX_COMPARES_PER_SESSION}")
    if problem:
        st.warning(problem)

    cmp = ss.get("compare")
    if not cmp:
        return
    st.markdown(f"**Results for:** {md_escape(cmp['question'])} · _{cmp['corpus']}_")
    gold = expected_pages().get(_norm_q(cmp["question"]), set()) if cmp["corpus"] == SAMPLE else set()
    colours = page_colours(cmp["results"])
    for col, (mode_name, chunks) in zip(st.columns(len(cmp["results"])), cmp["results"].items()):
        with col:
            st.markdown(f"**{mode_name}**")
            if not chunks:
                st.caption("No matches (no shared keywords).")
            for rank, c in enumerate(chunks, start=1):
                key = (c["source"].lower(), c["page"])
                dot = f'<span style="color:{colours[key]};font-size:1.25em">●</span>'
                with st.container(border=True):
                    st.markdown(f"{dot} **#{rank} · p.{c['page']}**" + (" ✅" if key in gold else ""),
                                unsafe_allow_html=True)
                    st.caption(make_snippet(c["text"], cmp["question"], width=110))
    st.button("💬 Answer this with Hybrid + Rerank", on_click=answer_from_compare,
              disabled=queries_left() == 0,
              help="Sends the question to the Ask tab (uses one of your questions).")


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------
def answer_question(question: str, index: HybridIndex | None, mode: Mode, corpus: str) -> None:
    ss = st.session_state
    question = question.strip()
    if not question:
        return
    if len(question) > config.MAX_QUESTION_CHARS:
        st.warning(f"Please keep questions under {config.MAX_QUESTION_CHARS} characters.")
        return
    if queries_left() == 0:
        st.info(f"You've used all {config.MAX_QUERIES_PER_SESSION} questions for this demo session. "
                "Thanks for trying it! Clone the repo to run it with your own key.")
        return
    if index is None or len(index) == 0:
        st.warning("There's nothing to search yet. Upload and index a document first.")
        return

    # History for the rewrite: successful turns only, before this question.
    history = [{"role": m["role"], "content": m["content"]}
               for m in ss.messages if not m.get("error")]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 1. LLM client. A missing key is reported, but retrieval still runs (it's free).
        client, client_err = None, None
        try:
            client = make_client()
        except GenerationError as exc:
            client_err = str(exc)

        # 2. Query rewriting (only with history) + retrieval.
        try:
            with st.spinner("Searching…"):
                rewritten = rewrite_query(client, history, question) if (client and history) else question
                chunks: list[RetrievedChunk] = index.search(rewritten, mode)
        except EmptyIndexError as exc:
            st.warning(str(exc))
            ss.messages.pop()
            return
        except Exception as exc:
            msg = f"Retrieval failed ({type(exc).__name__}). Please try again."
            st.error(msg)
            ss.messages.append({"role": "assistant", "content": msg, "error": True})
            return

        msg_idx = len(ss.messages)  # where this answer will sit in the history (widget keys)
        meta = {"question": question, "rewritten": rewritten, "mode": mode.value, "corpus": corpus,
                "chunks": [asdict(c) for c in chunks], "citations": None}

        if client_err:
            st.error(client_err)
            ss.messages.append({"role": "assistant", "content": client_err, "error": True, "meta": meta})
            render_details(meta, msg_idx)
            return

        # 3. Generation (counts toward the per-session limit).
        ss.query_count += 1
        if not chunks:
            answer = "I don't know based on the documents."
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(stream_answer(client, rewritten, chunks))
            except GenerationError as exc:
                st.error(str(exc))
                ss.messages.append({"role": "assistant", "content": str(exc), "error": True, "meta": meta})
                render_details(meta, msg_idx)
                return
        answer = answer if isinstance(answer, str) else "".join(map(str, answer))

        # 4. Citation check against what was actually retrieved.
        rep = check_citations(answer, chunks)
        meta["citations"] = {"valid": rep.valid, "invalid": rep.invalid, "uncited": rep.uncited}

        # 5. Claim check (AI judge). Not counted toward the limit; never blocks the answer.
        if config.CLAIM_CHECK_ENABLED and chunks:
            with st.spinner("Checking each claim against its cited passage…"):
                meta["claims"] = asdict(verify_claims(client, answer, chunks))
        render_details(meta, msg_idx)
        ss.messages.append({"role": "assistant", "content": answer, "meta": meta})


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def ask_example(question: str) -> None:
    st.session_state.pending_question = question


def new_chat() -> None:
    st.session_state.messages = []


def main() -> None:
    init_state()
    ss = st.session_state

    st.title("🔎 Hybrid RAG")
    st.caption("Ask questions about the documents. Answers are grounded in retrieved passages, "
               "with citations. Pick a retrieval mode in the sidebar, or see all four side by "
               "side in Compare modes.")
    sample_index, sample_err = load_resources()
    refresh_uploads()
    uploads_index = user_index()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📄 Documents")
        has_uploads = uploads_index is not None and len(uploads_index) > 0
        corpora = [SAMPLE, UPLOADS] if has_uploads else [SAMPLE]
        if ss.pop("switch_corpus", None) in corpora:
            ss.corpus = UPLOADS
        if ss.get("corpus") not in corpora:
            ss.corpus = SAMPLE
        corpus = st.radio("Search in", corpora, key="corpus")
        if sample_index is not None:
            st.caption("Sample: " + ", ".join(sample_index.sources))
        elif sample_err:
            st.warning(sample_err)

        files = st.file_uploader(
            f"Upload PDF / TXT / MD (max {config.MAX_FILES} files, {config.MAX_FILE_MB} MB each)",
            type=[e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        for kind, msg in ss.notices:
            getattr(st, kind)(msg)
        ss.notices = []
        if st.button("Index documents", disabled=not files, use_container_width=True):
            index_uploads(files or [])
            st.rerun()
        if ss.uploaded_names:
            st.caption("Your files: " + ", ".join(ss.uploaded_names))
            if st.button("Clear my documents", use_container_width=True):
                clear_uploads()
                st.rerun()

        st.header("⚙️ Retrieval")
        ss.setdefault("mode", Mode.HYBRID_RERANK)
        mode = st.selectbox("Mode", list(Mode), key="mode", format_func=lambda m: m.value)
        st.caption("Semantic top-10 + BM25 top-10 → Reciprocal Rank Fusion → cross-encoder rerank → top-4.")
        counter = st.empty()
        # Filled at the end of the run, so it also appears right after the first answer.
        new_chat_slot = st.empty()

    # ---- Main ----
    index = uploads_index if corpus == UPLOADS else sample_index
    ask_tab, compare_tab = st.tabs([TAB_ASK, TAB_COMPARE], key="main_tab", on_change="rerun")
    with compare_tab:
        render_compare(index, corpus)
    with ask_tab:
        render_ask(index, mode, corpus, counter, new_chat_slot)


def render_ask(index: HybridIndex | None, mode: Mode, corpus: str, counter, new_chat_slot) -> None:
    ss = st.session_state
    render_history()

    # Buttons use on_click callbacks (no st.rerun()): the extra rerun used to
    # swallow a question typed straight after clicking.
    examples_slot = st.empty()  # cleared as soon as a question is asked in this run
    if not ss.messages and not ss.pending_question and corpus == SAMPLE \
            and (examples := example_questions()):
        with examples_slot.container():
            st.markdown("**Try one:**")
            cols = st.columns(len(examples))
            for i, (col, q) in enumerate(zip(cols, examples)):
                col.button(q, key=f"example_{i}", use_container_width=True,
                           on_click=ask_example, args=(q,))

    answer_area = st.container()  # keeps the new answer above the (inline) chat box
    typed = st.chat_input("Ask a question about the documents…", max_chars=config.MAX_QUESTION_CHARS)
    question = typed or ss.pending_question
    ss.pending_question = None
    if question:
        examples_slot.empty()
        with answer_area:
            answer_question(question, index, mode, corpus)

    counter.caption(f"Questions left this session: **{queries_left()} / {config.MAX_QUERIES_PER_SESSION}**")
    if ss.messages:
        new_chat_slot.button("New chat", key="new_chat", use_container_width=True, on_click=new_chat)


if __name__ == "__main__":  # Streamlit runs the script as __main__; tests can import it
    main()
