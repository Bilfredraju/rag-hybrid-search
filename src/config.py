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

load_dotenv(
    PROJECT_ROOT / ".env"
)


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
# VECTOR DATABASE
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

# Number of results returned by the web search engine.
WEB_MAX_RESULTS = int(
    os.getenv(
        "WEB_MAX_RESULTS",
        "5",
    )
)


# Maximum number of concurrent web requests.
WEB_MAX_WORKERS = int(
    os.getenv(
        "WEB_MAX_WORKERS",
        "5",
    )
)


# Maximum time allowed for an individual web request.
WEB_FETCH_TIMEOUT = int(
    os.getenv(
        "WEB_FETCH_TIMEOUT",
        "5",
    )
)


# Maximum amount of text extracted from a web page.
WEB_MAX_CHARS = int(
    os.getenv(
        "WEB_MAX_CHARS",
        "4000",
    )
)


# Number of top-ranked web sources that should
# actually be fetched.
#
# Search can return 5 results, but we only fetch
# the best 3 to reduce latency.
WEB_FETCH_TOP_K = int(
    os.getenv(
        "WEB_FETCH_TOP_K",
        "3",
    )
)


# Web research cache lifetime in seconds.
#
# 300 seconds = 5 minutes.
#
# Set to 0 to disable caching.
WEB_CACHE_TTL = int(
    os.getenv(
        "WEB_CACHE_TTL",
        "300",
    )
)


# Directory used to store cached web research.
WEB_CACHE_DIR = DATA_DIR / "web_cache"


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

def ensure_data_directories():
    """
    Create all required project data directories.

    This function is safe to call multiple times.
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