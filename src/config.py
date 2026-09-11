from pathlib import Path
import os

from dotenv import load_dotenv

# Project root is the directory containing the src/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / os.getenv("DOCS_DIR", "docs")
CHUNKS_DIR = PROJECT_ROOT / os.getenv("CHUNKS_DIR", "data/chunks")
PROCESSED_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DIR", "data/processed")
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_DIR", "data/chroma_db")

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "5"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# Retrieval confidence
MIN_RERANK_SCORE = float(
    os.getenv("MIN_RERANK_SCORE", "-1.0")
)

MIN_CONFIDENT_RESULTS = int(
    os.getenv("MIN_CONFIDENT_RESULTS", "1")
)
MIN_EVIDENCE_SCORE = float(
    os.getenv("MIN_EVIDENCE_SCORE", "0.0")
)


def ensure_data_directories() -> None:
    """Create runtime data directories when they do not exist."""
    for path in (DATA_DIR, CHUNKS_DIR, PROCESSED_DIR, CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)



