from src.retrieval.hybrid import HybridSearch
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker

from src.generation.prompt_builder import PromptBuilder
from src.generation.llm import LLM
from src.config import RERANK_TOP_K


class RAGPipeline:
    """
    Enterprise Hybrid RAG Pipeline
    """

    def __init__(self):

        print("=" * 60)
        print("Initializing Enterprise RAG Pipeline")
        print("=" * 60)

        # Retrieval Components
        self.hybrid = HybridSearch()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()

        # Generation Components
        self.prompt_builder = PromptBuilder()
        self.llm = LLM()

        print("\n✅ Pipeline Ready")

    def ask(self, query):
        """
        Process a user query using the complete RAG pipeline.
        """

        # -----------------------------
        # Hybrid Search
        # -----------------------------
        semantic_results, bm25_results = self.hybrid.search(query)

        # -----------------------------
        # Reciprocal Rank Fusion
        # -----------------------------
        fused_results = self.rrf.fuse(
            semantic_results,
            bm25_results
        )

        # -----------------------------
        # Cross-Encoder Reranking
        # -----------------------------
        reranked_results = self.reranker.rerank(
            query,
            fused_results
        )

        # Keep only the Top 3 most relevant documents
        top_documents = reranked_results[:RERANK_TOP_K]

        # -----------------------------
        # Prompt Building
        # -----------------------------
        prompt = self.prompt_builder.build_prompt(
            query,
            top_documents
        )

        # -----------------------------
        # Generate Answer
        # -----------------------------
        answer = self.llm.generate(prompt)

        # -----------------------------
        # Prepare Sources
        # -----------------------------
        sources = []

        for doc in top_documents:
            sources.append(
                {
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page"],
                }
            )

        # -----------------------------
        # Final Response
        # -----------------------------
        return {
            "question": query,
            "answer": answer,
            "sources": sources,
        }