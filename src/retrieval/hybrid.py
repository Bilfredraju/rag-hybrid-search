from src.retrieval.retriever import Retriever
from src.retrieval.bm25_search import BM25Search


class HybridSearch:
    """
    Combines Semantic Search and BM25 Search.
    """

    def __init__(self):

        print("Loading Semantic Search...")
        self.semantic = Retriever()

        print("Loading BM25...")
        self.bm25 = BM25Search()

    def search(self, query):

        semantic_results = self.semantic.search(query)

        bm25_results = self.bm25.search(query)

        return semantic_results, bm25_results