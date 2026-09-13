import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DIR = Path(
    os.getenv(
        "CHROMA_DIR",
        str(DATA_DIR / "chroma_db"),
    )
)

DOCS_DIR = Path(
    os.getenv(
        "DOCS_DIR",
        str(PROJECT_ROOT / "docs"),
    )
)

CHUNKS_DIR = Path(
    os.getenv(
        "CHUNKS_DIR",
        str(DATA_DIR / "chunks"),
    )
)

PROCESSED_DIR = Path(
    os.getenv(
        "PROCESSED_DIR",
        str(DATA_DIR / "processed"),
    )
)


# ============================================================
# CHROMA / VECTOR DATABASE
# ============================================================

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "rag_documents",
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)


# ============================================================
# RERANKER
# ============================================================

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

SEMANTIC_TOP_K = int(
    os.getenv(
        "SEMANTIC_TOP_K",
        "5",
    )
)

BM25_TOP_K = int(
    os.getenv(
        "BM25_TOP_K",
        "5",
    )
)

RRF_K = int(
    os.getenv(
        "RRF_K",
        "60",
    )
)

RERANK_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        "3",
    )
)


# ============================================================
# RETRIEVAL CONFIDENCE
# ============================================================

MIN_RERANK_SCORE = float(
    os.getenv(
        "MIN_RERANK_SCORE",
        "-1.0",
    )
)

MIN_CONFIDENT_RESULTS = int(
    os.getenv(
        "MIN_CONFIDENT_RESULTS",
        "1",
    )
)

MIN_EVIDENCE_SCORE = float(
    os.getenv(
        "MIN_EVIDENCE_SCORE",
        "0.0",
    )
)


# ============================================================
# WEB RESEARCH
# ============================================================

WEB_MAX_RESULTS = int(
    os.getenv(
        "WEB_MAX_RESULTS",
        "5",
    )
)

WEB_MAX_WORKERS = int(
    os.getenv(
        "WEB_MAX_WORKERS",
        "5",
    )
)

WEB_FETCH_TIMEOUT = int(
    os.getenv(
        "WEB_FETCH_TIMEOUT",
        "5",
    )
)

WEB_MAX_CHARS = int(
    os.getenv(
        "WEB_MAX_CHARS",
        "4000",
    )
)

WEB_FETCH_TOP_K = int(
    os.getenv(
        "WEB_FETCH_TOP_K",
        "3",
    )
)

WEB_CACHE_TTL = int(
    os.getenv(
        "WEB_CACHE_TTL",
        "300",
    )
)


# ============================================================
# QUERY-AWARE WEB CACHE TTL
# ============================================================

WEB_CACHE_TTL_FINANCE = int(
    os.getenv(
        "WEB_CACHE_TTL_FINANCE",
        "60",
    )
)

WEB_CACHE_TTL_NEWS = int(
    os.getenv(
        "WEB_CACHE_TTL_NEWS",
        "120",
    )
)

WEB_CACHE_TTL_GOVERNMENT = int(
    os.getenv(
        "WEB_CACHE_TTL_GOVERNMENT",
        "900",
    )
)

WEB_CACHE_TTL_HR = int(
    os.getenv(
        "WEB_CACHE_TTL_HR",
        "900",
    )
)

WEB_CACHE_TTL_TECHNOLOGY = int(
    os.getenv(
        "WEB_CACHE_TTL_TECHNOLOGY",
        "600",
    )
)

WEB_CACHE_TTL_RESEARCH = int(
    os.getenv(
        "WEB_CACHE_TTL_RESEARCH",
        "1800",
    )
)

WEB_CACHE_TTL_GENERAL = int(
    os.getenv(
        "WEB_CACHE_TTL_GENERAL",
        "300",
    )
)


# ============================================================
# WEB SOURCE RELIABILITY
# ============================================================

WEB_MAX_DOMAIN_FAILURES = int(
    os.getenv(
        "WEB_MAX_DOMAIN_FAILURES",
        "2",
    )
)

WEB_DOMAIN_FAILURE_TTL = int(
    os.getenv(
        "WEB_DOMAIN_FAILURE_TTL",
        "900",
    )
)

WEB_MIN_CONTENT_CHARS = int(
    os.getenv(
        "WEB_MIN_CONTENT_CHARS",
        "200",
    )
)


# ============================================================
# WEB RESEARCH TIMEOUT
# ============================================================
#
# Maximum amount of time allocated to web source fetching
# during one research request.
#
# This prevents slow/unresponsive websites from making the
# complete assistant request hang for an excessive amount
# of time.
# ============================================================

WEB_RESEARCH_TIMEOUT = int(
    os.getenv(
        "WEB_RESEARCH_TIMEOUT",
        "15",
    )
)


# ============================================================
# WEB CACHE DIRECTORY
# ============================================================

WEB_CACHE_DIR = DATA_DIR / "web_cache"


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

def ensure_data_directories():
    """
    Create all required project directories.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WEB_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )