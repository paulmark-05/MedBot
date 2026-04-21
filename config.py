# =============================================================
# config.py — Medical Health FAQ Bot
# Central configuration. Edit here; nothing else needs to change.
# =============================================================

# ── LLM ──────────────────────────────────────────────────────
LLM_MODEL         = "llama-3.3-70b-versatile"
LLM_TEMPERATURE   = 0

# ── Embedding model (local, no API key needed) ───────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_COLLECTION = "medical_faq_kb"
RAG_TOP_K         = 3          # chunks returned per query

# ── Conversation memory ──────────────────────────────────────
MEMORY_WINDOW     = 6          # messages kept (= 3 turns)

# ── Self-reflection / eval ───────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.70  # below this → retry answer
MAX_EVAL_RETRIES       = 2     # safety valve

# ── Web search ───────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 3     # DuckDuckGo results to surface

# ── Domain metadata ──────────────────────────────────────────
DOMAIN_NAME        = "MediBot — Health FAQ Assistant"
DOMAIN_DESCRIPTION = (
    "An AI-powered health information assistant that answers "
    "medical and wellness questions using a curated knowledge base "
    "and live web search. For informational purposes only — "
    "always consult a qualified healthcare professional."
)
DISCLAIMER = (
    "⚠️ **Disclaimer:** MediBot provides general health information only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment."
)
