from src.config import RERANK_TOP_K
from src.generation.llm import LLM
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.confidence import RetrievalConfidence
from src.retrieval.hybrid import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import ReciprocalRankFusion


class RAGPipeline:
    """
    Enterprise Hybrid RAG Pipeline.

    Flow:
        Query
          ↓
        Hybrid Search
          ↓
        RRF
          ↓
        CrossEncoder Reranker
          ↓
        Confidence Check
          ↓
        Evidence Filtering
          ↓
        Prompt Builder
          ↓
        LLM
          ↓
        Answer + Sources
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Enterprise RAG Pipeline")
        print("=" * 60)

        self.hybrid = HybridSearch()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()
        self.confidence = RetrievalConfidence()

        self.prompt_builder = PromptBuilder()
        self.llm = LLM()

        print("\n✅ Pipeline Ready")

    def ask(self, query):
        """
        Answer a question using retrieved document evidence.

        The pipeline:
        1. Performs hybrid retrieval.
        2. Combines semantic and BM25 results using RRF.
        3. Reranks the retrieved chunks.
        4. Checks whether the retrieval is confident enough.
        5. Filters weak evidence.
        6. Sends only sufficiently relevant evidence to the LLM.
        7. Returns the answer with deduplicated sources.
        """

        # -------------------------------------------------
        # VALIDATE QUERY
        # -------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        # -------------------------------------------------
        # HYBRID RETRIEVAL
        # -------------------------------------------------

        semantic_results, bm25_results = (
            self.hybrid.search(query)
        )

        # -------------------------------------------------
        # RECIPROCAL RANK FUSION
        # -------------------------------------------------

        fused_results = self.rrf.fuse(
            semantic_results,
            bm25_results,
        )

        # -------------------------------------------------
        # CROSS-ENCODER RERANKING
        # -------------------------------------------------

        reranked_results = self.reranker.rerank(
            query,
            fused_results,
        )

        # -------------------------------------------------
        # CONFIDENCE CHECK
        # -------------------------------------------------

        confidence = self.confidence.evaluate(
            reranked_results
        )

        if not confidence["confident"]:

            return {
                "question": query,
                "answer": (
                    "I don't have enough information "
                    "in the indexed documents to answer "
                    "this question reliably."
                ),
                "sources": [],
                "confidence": confidence,
            }

        # -------------------------------------------------
        # EVIDENCE FILTERING
        # -------------------------------------------------

        evidence_documents = (
            self.confidence.filter_evidence(
                reranked_results
            )
        )

        # Safety check in case all documents were
        # removed by the evidence filter.

        if not evidence_documents:

            return {
                "question": query,
                "answer": (
                    "I don't have enough information "
                    "in the indexed documents to answer "
                    "this question reliably."
                ),
                "sources": [],
                "confidence": {
                    **confidence,
                    "reason": (
                        "no_sufficient_evidence"
                    ),
                },
            }

        # -------------------------------------------------
        # SELECT FINAL DOCUMENTS
        # -------------------------------------------------

        top_documents = evidence_documents[
            :RERANK_TOP_K
        ]

        # -------------------------------------------------
        # BUILD PROMPT
        # -------------------------------------------------

        prompt = self.prompt_builder.build_prompt(
            query,
            top_documents,
        )

        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        answer = self.llm.generate(prompt)

        # -------------------------------------------------
        # BUILD DEDUPLICATED SOURCES
        # -------------------------------------------------

        sources = []
        seen = set()

        for doc in top_documents:

            metadata = doc.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source"
            )

            page = metadata.get(
                "page"
            )

            if source is None:
                continue

            source_key = (
                source,
                page,
            )

            if source_key in seen:
                continue

            seen.add(source_key)

            sources.append(
                {
                    "source": source,
                    "page": page,
                }
            )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    def refresh_indexes(self):
        """
        Refresh retrieval indexes after documents
        are uploaded, replaced, or deleted.

        ChromaDB is persistent and immediately reflects
        document changes.

        BM25 keeps an in-memory snapshot, so it must
        explicitly reload the persisted chunk files.
        """

        print(
            "\nRefreshing retrieval indexes..."
        )

        self.hybrid.reload_bm25()

        print(
            "✅ Retrieval indexes refreshed"
        )