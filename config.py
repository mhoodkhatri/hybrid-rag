"""Central configuration. Swap the LLM provider by editing only this file.

Any OpenAI-compatible endpoint works, e.g.:
    Groq:   LLM_BASE_URL = "https://api.groq.com/openai/v1"
    Gemini: LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    Ollama: LLM_BASE_URL = "http://localhost:11434/v1"
(set LLM_EXTRA_BODY = {} for providers that don't understand DeepSeek's flags)
"""

from __future__ import annotations

from pathlib import Path

# --- LLM (the only paid component) ---------------------------------------
LLM_BASE_URL: str = "https://api.deepseek.com"
LLM_MODEL: str = "deepseek-flash"
LLM_API_KEY_NAME: str = "DEEPSEEK_API_KEY"  # looked up in st.secrets, then env/.env
# Provider-specific request fields. deepseek-flash reasons ("thinks") by default,
# which costs hidden tokens and latency; this turns it off. Verified 2026-10-01:
# default call used reasoning_tokens=10 for a one-word reply, disabled used 0.
LLM_EXTRA_BODY: dict = {"thinking": {"type": "disabled"}}
TEMPERATURE: float = 0.1
REWRITE_MAX_TOKENS: int = 80
ANSWER_MAX_TOKENS: int = 800
LLM_TIMEOUT_S: float = 30.0
LLM_MAX_RETRIES: int = 2
REWRITE_HISTORY_TURNS: int = 6
# Claim check: one extra JSON-mode call after each answer grades every cited
# sentence against the chunk(s) it cites. Failure never blocks the answer.
CLAIM_CHECK_ENABLED: bool = True
CLAIM_CHECK_MAX_CLAIMS: int = 12
CLAIM_CHECK_MAX_TOKENS: int = 700
CLAIM_CHECK_TIMEOUT_S: float = 30.0

# --- Local models (free, CPU) ---------------------------------------------
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
# BGE v1.5 recommended instruction for short retrieval queries (not for passages).
EMBED_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking ---------------------------------------------------------------
CHUNK_TOKENS: int = 350
CHUNK_OVERLAP: int = 50

# --- Retrieval --------------------------------------------------------------
SEM_TOP_K: int = 10
BM25_TOP_K: int = 10
RRF_K: int = 60
FINAL_TOP_K: int = 4

# --- Cost / abuse limits (public demo) ------------------------------------
MAX_QUERIES_PER_SESSION: int = 20
MAX_FILE_MB: int = 10
MAX_FILES: int = 3
MAX_QUESTION_CHARS: int = 500
# "Compare modes" runs retrieval only (no LLM, free), so it has its own CPU-protecting cap.
MAX_COMPARES_PER_SESSION: int = 50
# Protects the shared CPU: a 10 MB text-heavy PDF can yield thousands of chunks.
MAX_CHUNKS_PER_SESSION: int = 1500
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md")
# Streamlit has no session-end hook, so uploads are swept on every script run:
# idle sessions expire, and only the most recently active ones are kept.
SESSION_TTL_MIN: int = 30
MAX_UPLOAD_SESSIONS: int = 25

# --- Paths ----------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
