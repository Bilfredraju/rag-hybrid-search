from src.config import BM25_TOP_K, SEMANTIC_TOP_K
from src.retrieval.bm25_search import BM25Search
from src.retrieval.retriever import Retriever


class HybridSearch:
    """Runs dense semantic and sparse BM25 retrieval over the same chunks."""

    def __init__(self, semantic_top_k=SEMANTIC_TOP_K, bm25_top_k=BM25_TOP_K):
        self.semantic_top_k = semantic_top_k
        self.bm25_top_k = bm25_top_k
        self.semantic = Retriever()
        self.bm25 = BM25Search()

    def search(self, query):
        return (
            self.semantic.search(query, self.semantic_top_k),
            self.bm25.search(query, self.bm25_top_k),
        )
