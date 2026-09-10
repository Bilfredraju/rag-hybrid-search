from src.config import RERANK_TOP_K
from src.generation.llm import LLM
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.hybrid import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import ReciprocalRankFusion


class RAGPipeline:
    """
    Enterprise Hybrid RAG Pipeline.
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Enterprise RAG Pipeline")
        print("=" * 60)

        self.hybrid = HybridSearch()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()

        self.prompt_builder = PromptBuilder()
        self.llm = LLM()

        print("\n✅ Pipeline Ready")

    def ask(self, query):
        semantic_results, bm25_results = (
            self.hybrid.search(query)
        )

        fused_results = self.rrf.fuse(
            semantic_results,
            bm25_results,
        )

        reranked_results = self.reranker.rerank(
            query,
            fused_results,
        )

        top_documents = reranked_results[
            :RERANK_TOP_K
        ]

        prompt = self.prompt_builder.build_prompt(
            query,
            top_documents,
        )

        answer = self.llm.generate(prompt)

        sources = []

        for doc in top_documents:
            metadata = doc["metadata"]

            sources.append(
                {
                    "source": metadata["source"],
                    "page": metadata["page"],
                }
            )

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
        }

    def refresh_indexes(self):
        """
        Refresh indexes after document changes.

        ChromaDB is persistent and immediately reflects
        document changes. BM25 needs to reload its
        persisted chunk corpus.
        """

        print("\nRefreshing retrieval indexes...")

        self.hybrid.reload_bm25()

        print("✅ Retrieval indexes refreshed")