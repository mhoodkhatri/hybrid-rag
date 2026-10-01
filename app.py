"""Streamlit UI for the Hybrid RAG demo.

Multi-user model:
- Models, the Chroma client and the sample-document index are process-wide
  singletons (st.cache_resource), shared read-only by every visitor.
- Uploaded documents go into a per-session collection whose object lives only
  in st.session_state, so one visitor can never query another visitor's files.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import streamlit as st

import config
from generator import GenerationError, check_citations, make_client, rewrite_query, stream_answer
from ingest import IngestError, ingest, ingest_sample_dir, sample_files
from retriever import (
    EmptyIndexError,
    HybridIndex,
    Mode,
    RetrievedChunk,
    load_embedder,
    load_reranker,
)
from retriever import make_client as make_chroma_client

SAMPLE = "Sample document"
UPLOADS = "My uploads"

st.set_page_config(page_title="Hybrid RAG", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Shared, cached resources (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return load_embedder()


@st.cache_resource(show_spinner="Loading reranker…")
def get_reranker():
    return load_reranker()


@st.cache_resource
def get_chroma():
    return make_chroma_client()


@st.cache_resource(show_spinner="Indexing the sample document (first start only)…")
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
    ss.setdefault("user_index", None)  # HybridIndex | None, private to this session
    ss.setdefault("uploaded_names", [])
    ss.setdefault("pending_question", None)
    ss.setdefault("notices", [])  # (kind, message) shown once in the sidebar after a rerun


def queries_left() -> int:
    return max(0, config.MAX_QUERIES_PER_SESSION - st.session_state.query_count)


def clear_uploads() -> None:
    idx: HybridIndex | None = st.session_state.user_index
    if idx is not None:
        try:
            idx.delete()
        except Exception:
            pass
    st.session_state.user_index = None
    st.session_state.uploaded_names = []


def index_uploads(files: list) -> None:
    """Validate and index uploaded files into this session's private collection."""
    ss = st.session_state
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
        try:
            with st.sidebar.status(f"Indexing {f.name}…"):
                chunks = ingest(f.name, f.getvalue())
                current = len(ss.user_index) if ss.user_index is not None else 0
                if current + len(chunks) > config.MAX_CHUNKS_PER_SESSION:
                    raise IngestError(f"'{f.name}' is too large for the demo "
                                      f"(>{config.MAX_CHUNKS_PER_SESSION} chunks per session).")
                if ss.user_index is None:
                    ss.user_index = HybridIndex(get_chroma(), get_embedder(), get_reranker(),
                                                name_prefix="sess")
                ss.user_index.add_chunks(chunks)
            ss.uploaded_names.append(f.name)
            ok += 1
        except IngestError as exc:
            errors.append(str(exc))
        except Exception as exc:  # never crash the app on a bad file
            errors.append(f"'{f.name}': unexpected error while indexing ({type(exc).__name__}).")
    ss.notices.extend(("error", e) for e in errors)
    if ok:
        ss.notices.append(("success", f"Indexed {ok} file(s). Now searching “{UPLOADS}”."))
        ss.switch_corpus = UPLOADS  # applied before the radio widget is drawn


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _fmt(score: float | None, rank: int | None, digits: int) -> str:
    return "—" if score is None else f"{score:.{digits}f} (#{rank})"


def render_details(meta: dict) -> None:
    report = meta.get("citations")
    if report:
        if report["invalid"]:
            bad = ", ".join(f"[{s}, p.{p}]" for s, p in report["invalid"])
            st.warning(f"⚠️ Citation check: {bad} does not match any retrieved chunk.")
        elif report["uncited"]:
            st.warning("⚠️ Citation check: the answer contains no citations.")
        elif report["valid"]:
            st.caption(f"✅ All {len(report['valid'])} citation(s) match retrieved chunks.")

    with st.expander(f"Retrieved chunks · {meta['mode']} · {meta['corpus']}"):
        if meta.get("rewritten") and meta["rewritten"] != meta["question"]:
            st.markdown(f"**Rewritten query:** {meta['rewritten']}")
        else:
            st.markdown(f"**Query:** {meta['question']}")
        if not meta["chunks"]:
            st.info("No chunks retrieved (BM25 found no matching terms?).")
        for i, c in enumerate(meta["chunks"], start=1):
            s, r = c["scores"], c["ranks"]
            st.markdown(
                f"**#{i} · {c['source']}, p.{c['page']}**  \n"
                f"semantic {_fmt(s['semantic'], r['semantic'], 3)} · "
                f"BM25 {_fmt(s['bm25'], r['bm25'], 2)} · "
                f"RRF {_fmt(s['rrf'], r['rrf'], 4)} · "
                f"rerank {_fmt(s['rerank'], r['rerank'], 2)}"
            )
            st.code(c["text"], language=None, wrap_lines=True)


def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("meta"):
                render_details(msg["meta"])


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

        meta = {"question": question, "rewritten": rewritten, "mode": mode.value, "corpus": corpus,
                "chunks": [asdict(c) for c in chunks], "citations": None}

        if client_err:
            st.error(client_err)
            ss.messages.append({"role": "assistant", "content": client_err, "error": True, "meta": meta})
            render_details(meta)
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
                render_details(meta)
                return
        answer = answer if isinstance(answer, str) else "".join(map(str, answer))

        # 4. Citation check against what was actually retrieved.
        rep = check_citations(answer, chunks)
        meta["citations"] = {"valid": rep.valid, "invalid": rep.invalid, "uncited": rep.uncited}
        render_details(meta)
        ss.messages.append({"role": "assistant", "content": answer, "meta": meta})


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    ss = st.session_state
    sample_index, sample_err = get_sample_index()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📄 Documents")
        has_uploads = ss.user_index is not None and len(ss.user_index) > 0
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
        st.caption("🔒 Uploaded files are processed in memory for this session only. "
                   "They are never stored and are gone when you close the tab.")

        st.header("⚙️ Retrieval")
        mode = st.selectbox("Mode", list(Mode), index=3, format_func=lambda m: m.value)
        st.caption("Semantic top-10 + BM25 top-10 → Reciprocal Rank Fusion → cross-encoder rerank → top-4.")
        counter = st.empty()
        # Filled at the end of the run, so it also appears right after the first answer.
        new_chat_slot = st.empty()

    # ---- Main ----
    st.title("🔎 Hybrid RAG")
    st.caption("Ask questions about the documents. Answers are grounded in retrieved passages, "
               "with citations. Compare retrieval modes in the sidebar.")

    index = ss.user_index if corpus == UPLOADS else sample_index
    render_history()

    examples_slot = st.empty()  # cleared as soon as a question is asked in this run
    if not ss.messages and corpus == SAMPLE and (examples := example_questions()):
        with examples_slot.container():
            st.markdown("**Try one:**")
            cols = st.columns(len(examples))
            for col, q in zip(cols, examples):
                if col.button(q, use_container_width=True):
                    ss.pending_question = q
                    st.rerun()

    typed = st.chat_input("Ask a question about the documents…", max_chars=config.MAX_QUESTION_CHARS)
    question = typed or ss.pending_question
    ss.pending_question = None
    if question:
        examples_slot.empty()
        answer_question(question, index, mode, corpus)

    counter.caption(f"Questions left this session: **{queries_left()} / {config.MAX_QUERIES_PER_SESSION}**")
    if ss.messages and new_chat_slot.button("New chat", key="new_chat", use_container_width=True):
        ss.messages = []
        st.rerun()


if __name__ == "__main__":  # Streamlit runs the script as __main__; tests can import it
    main()
