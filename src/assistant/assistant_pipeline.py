from __future__ import annotations

import re

from src.config import RERANK_TOP_K
from src.generation.both_prompt import BothPromptBuilder
from src.generation.general_prompt import GeneralPromptBuilder
from src.generation.llm import LLM
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.confidence import RetrievalConfidence
from src.retrieval.hybrid import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import ReciprocalRankFusion
from src.router.query_router import QueryRoute, QueryRouter
from src.web.web_research import WebResearch


class AssistantPipeline:
    """
    Universal AI Assistant.

    Supports four query routes:

    DOCUMENT
        Answer using uploaded documents.

    GENERAL
        Answer using the LLM's general knowledge.

    CURRENT
        Answer using current web research.

    BOTH
        Combine uploaded-document evidence with
        current web research.
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        # =====================================================
        # DOCUMENT RETRIEVAL
        # =====================================================

        self.hybrid = HybridSearch()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()
        self.confidence = RetrievalConfidence()

        # =====================================================
        # PROMPT BUILDERS
        # =====================================================

        self.prompt_builder = PromptBuilder()
        self.general_prompt_builder = GeneralPromptBuilder()
        self.both_prompt_builder = BothPromptBuilder()

        # =====================================================
        # LLM
        # =====================================================

        self.llm = LLM()

        # =====================================================
        # QUERY ROUTER
        # =====================================================

        self.router = QueryRouter()

        # =====================================================
        # WEB RESEARCH
        # =====================================================

        self.web_research = WebResearch(
            max_results=5
        )

        print("\n✅ Universal Assistant Ready")

    # =========================================================
    # PUBLIC ASK METHOD
    # =========================================================

    def ask(self, query: str) -> dict:
        """
        Route and answer a user query.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        route = self.router.classify(
            query
        )

        print(
            f"\n🔀 Query Route: "
            f"{route.value.upper()}"
        )

        if route == QueryRoute.DOCUMENT:
            return self._ask_document(query)

        if route == QueryRoute.GENERAL:
            return self._ask_general(query)

        if route == QueryRoute.CURRENT:
            return self._ask_current(query)

        if route == QueryRoute.BOTH:
            return self._ask_both(query)

        raise RuntimeError(
            f"Unsupported query route: {route}"
        )

    # =========================================================
    # DOCUMENT ROUTE
    # =========================================================

    def _ask_document(
        self,
        query: str,
    ) -> dict:
        """
        Answer using uploaded-document evidence.
        """

        print(
            "📚 Using document knowledge..."
        )

        semantic_results, bm25_results = (
            self.hybrid.search(query)
        )

        fused_results = self.rrf.fuse(
            semantic_results,
            bm25_results,
        )

        reranked_results = (
            self.reranker.rerank(
                query,
                fused_results,
            )
        )

        confidence = (
            self.confidence.evaluate(
                reranked_results
            )
        )

        if not confidence["confident"]:
            return {
                "question": query,
                "answer": (
                    "I don't have enough information "
                    "in the indexed documents to answer "
                    "this question reliably."
                ),
                "route": QueryRoute.DOCUMENT.value,
                "sources": [],
                "confidence": confidence,
            }

        evidence_documents = (
            self.confidence.filter_evidence(
                reranked_results
            )
        )

        if not evidence_documents:
            return {
                "question": query,
                "answer": (
                    "I don't have enough information "
                    "in the indexed documents to answer "
                    "this question reliably."
                ),
                "route": QueryRoute.DOCUMENT.value,
                "sources": [],
                "confidence": {
                    **confidence,
                    "reason": "no_sufficient_evidence",
                },
            }

        top_documents = evidence_documents[
            :RERANK_TOP_K
        ]

        prompt = (
            self.prompt_builder.build_prompt(
                query,
                top_documents,
            )
        )

        answer = self.llm.generate(
            prompt
        )

        sources = (
            self._extract_document_sources(
                top_documents
            )
        )

        return {
            "question": query,
            "answer": answer,
            "route": QueryRoute.DOCUMENT.value,
            "sources": sources,
            "confidence": confidence,
        }

    # =========================================================
    # GENERAL ROUTE
    # =========================================================

    def _ask_general(
        self,
        query: str,
    ) -> dict:
        """
        Answer using general LLM knowledge.
        """

        print(
            "🧠 Using general LLM knowledge..."
        )

        prompt = (
            self.general_prompt_builder
            .build_prompt(query)
        )

        answer = self.llm.generate(
            prompt
        )

        return {
            "question": query,
            "answer": answer,
            "route": QueryRoute.GENERAL.value,
            "sources": [],
            "confidence": {
                "confident": True,
                "score": None,
                "reason": "general_knowledge",
            },
        }

    # =========================================================
    # CURRENT ROUTE
    # =========================================================

    def _ask_current(
        self,
        query: str,
    ) -> dict:
        """
        Answer using current web research.
        """

        print(
            "🌐 Using current web research..."
        )

        web_query = (
            self._build_web_query(
                query
            )
        )

        print(
            f"🔎 Web Query: "
            f"{web_query}"
        )

        web_result = (
            self.web_research
            .collect_evidence(
                web_query
            )
        )

        web_evidence = (
            web_result.get(
                "evidence",
                [],
            )
        )

        if not web_evidence:
            return {
                "question": query,
                "answer": (
                    "I could not find sufficient "
                    "current web information to answer "
                    "this question reliably."
                ),
                "route": QueryRoute.CURRENT.value,
                "sources": [],
                "confidence": {
                    "confident": False,
                    "score": None,
                    "reason": "web_research_unavailable",
                },
                "web_research": web_result.get(
                    "web_research",
                    {},
                ),
            }

        prompt = self._build_current_prompt(
            query,
            web_evidence,
        )

        answer = self.llm.generate(
            prompt
        )

        sources = []

        for source in web_result.get(
            "sources",
            [],
        ):
            sources.append(
                {
                    **source,
                    "type": "web",
                }
            )

        return {
            "question": query,
            "answer": answer,
            "route": QueryRoute.CURRENT.value,
            "sources": sources,
            "confidence": {
                "confident": True,
                "score": None,
                "reason": "web_evidence",
            },
            "web_research": web_result.get(
                "web_research",
                {},
            ),
        }

    # =========================================================
    # BOTH ROUTE
    # =========================================================

    def _ask_both(
        self,
        query: str,
    ) -> dict:
        """
        Answer using both:

        1. Uploaded-document evidence.
        2. Current web evidence.
        """

        print(
            "📚🌐 Combining document + web research..."
        )

        # =====================================================
        # DOCUMENT RETRIEVAL
        # =====================================================

        semantic_results, bm25_results = (
            self.hybrid.search(query)
        )

        fused_results = self.rrf.fuse(
            semantic_results,
            bm25_results,
        )

        reranked_results = (
            self.reranker.rerank(
                query,
                fused_results,
            )
        )

        confidence = (
            self.confidence.evaluate(
                reranked_results
            )
        )

        document_results = []

        if confidence["confident"]:

            evidence_documents = (
                self.confidence.filter_evidence(
                    reranked_results
                )
            )

            document_results = (
                evidence_documents[
                    :RERANK_TOP_K
                ]
            )

        # =====================================================
        # WEB RESEARCH
        # =====================================================

        web_query = (
            self._build_web_query(
                query
            )
        )

        print(
            f"🔎 Web Query: "
            f"{web_query}"
        )

        web_result = (
            self.web_research
            .collect_evidence(
                web_query
            )
        )

        web_results = (
            web_result.get(
                "evidence",
                [],
            )
        )

        # =====================================================
        # NO EVIDENCE FALLBACK
        # =====================================================

        if (
            not document_results
            and not web_results
        ):
            return {
                "question": query,
                "answer": (
                    "I don't have enough information "
                    "from the uploaded documents or "
                    "current web research to answer "
                    "this question reliably."
                ),
                "route": QueryRoute.BOTH.value,
                "sources": [],
                "confidence": confidence,
                "web_research": web_result.get(
                    "web_research",
                    {},
                ),
            }

        # =====================================================
        # COMBINED PROMPT
        # =====================================================

        prompt = (
            self.both_prompt_builder.build_prompt(
                query,
                document_results,
                web_results,
            )
        )

        # =====================================================
        # GENERATE COMBINED ANSWER
        # =====================================================

        try:

            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:

            print(
                f"⚠️ Combined generation failed: "
                f"{exc}"
            )

            # -------------------------------------------------
            # Document fallback
            # -------------------------------------------------

            if document_results:

                fallback_prompt = (
                    self.prompt_builder
                    .build_prompt(
                        query,
                        document_results,
                    )
                )

                answer = self.llm.generate(
                    fallback_prompt
                )

            # -------------------------------------------------
            # Web fallback
            # -------------------------------------------------

            elif web_results:

                fallback_prompt = (
                    self._build_current_prompt(
                        query,
                        web_results,
                    )
                )

                answer = self.llm.generate(
                    fallback_prompt
                )

            else:
                raise

        # =====================================================
        # DOCUMENT SOURCES
        # =====================================================

        document_sources = (
            self._extract_document_sources(
                document_results
            )
        )

        # =====================================================
        # WEB SOURCES
        # =====================================================
        #
        # Explicitly identify web sources using:
        #
        #     type = "web"
        #
        # This allows consumers/tests to distinguish
        # document evidence from web evidence.

        web_sources = []

        for source in web_result.get(
            "sources",
            [],
        ):

            web_sources.append(
                {
                    **source,
                    "type": "web",
                }
            )

        # =====================================================
        # COMBINE SOURCES
        # =====================================================

        sources = (
            document_sources
            + web_sources
        )

        return {
            "question": query,
            "answer": answer,
            "route": QueryRoute.BOTH.value,
            "sources": sources,
            "confidence": confidence,
            "web_research": web_result.get(
                "web_research",
                {},
            ),
        }

    # =========================================================
    # DOCUMENT SOURCE EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_document_sources(
        documents,
    ):
        """
        Convert retrieved document chunks into
        deduplicated source metadata.
        """

        sources = []
        seen = set()

        for document in documents:

            metadata = document.get(
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

            seen.add(
                source_key
            )

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "type": "document",
                }
            )

        return sources

    # =========================================================
    # CURRENT WEB PROMPT
    # =========================================================

    @staticmethod
    def _build_current_prompt(
        query,
        web_results,
    ):
        """
        Build a grounded prompt using current web evidence.
        """

        evidence_parts = []

        for index, result in enumerate(
            web_results,
            start=1,
        ):

            title = result.get(
                "title",
                "Unknown title",
            )

            url = result.get(
                "url",
                "",
            )

            content = result.get(
                "content",
                result.get(
                    "snippet",
                    "",
                ),
            )

            content = content[:6000]

            evidence_parts.append(
                f"""
Web Evidence {index}

Title: {title}

URL: {url}

Content:
{content}
"""
            )

        evidence = "\n".join(
            evidence_parts
        )

        return f"""
You are a current-information research assistant.

Answer the user's question using ONLY the supplied
web evidence.

STRICT RULES:

- Do not use unsupported outside knowledge.
- Do not invent facts.
- Use only the supplied web evidence.
- If the evidence is insufficient, clearly say so.
- Do not pretend that you accessed sources that
  are not supplied below.
- Do not create citation markers.
- Do not create a bibliography.
- Do not mention the retrieval process.
- Do not include source numbers in the answer.
- Do not include URLs in the answer.
- Keep the answer concise and useful.
- Preserve important technical terminology when
  supported by the evidence.

WEB EVIDENCE:

{evidence}

=======================================================

QUESTION:

{query}

=======================================================

ANSWER:
"""

    # =========================================================
    # WEB QUERY BUILDER
    # =========================================================

    @staticmethod
    def _build_web_query(
        query: str,
    ) -> str:
        """
        Convert an assistant query into a clean
        web-search query.

        Internal document references are removed while
        preserving the actual external/current topic.
        """

        normalized = " ".join(
            query.strip().split()
        )

        # =====================================================
        # REMOVE DOCUMENT REFERENCES
        # =====================================================

        patterns_to_remove = [
            r"\bbased on project phoenix\b",
            r"\bbased on the project phoenix\b",
            r"\bproject phoenix\b",

            r"\bbased on leave policy\b",
            r"\bbased on the leave policy\b",
            r"\bthe leave policy\b",
            r"\bleave policy\b",

            r"\baccording to the document\b",
            r"\baccording to the documents\b",

            r"\bin the document\b",
            r"\bin the documents\b",
            r"\bin this document\b",
            r"\bin these documents\b",

            r"\bfrom the document\b",
            r"\bfrom the documents\b",
        ]

        for pattern in patterns_to_remove:

            normalized = re.sub(
                pattern,
                " ",
                normalized,
                flags=re.IGNORECASE,
            )

        # =====================================================
        # REMOVE COMPARISON FILLER
        # =====================================================

        filler_patterns = [
            r"\bhow does it compare with\b",
            r"\bhow does this compare with\b",
            r"\bhow does it compare to\b",
            r"\bhow does it compare against\b",
            r"\bcompare with\b",
            r"\bcompare to\b",
            r"\bcompare against\b",
            r"\bcompare\b",
        ]

        for pattern in filler_patterns:

            normalized = re.sub(
                pattern,
                " ",
                normalized,
                flags=re.IGNORECASE,
            )

        # =====================================================
        # REMOVE LEADING / TRAILING PUNCTUATION
        # =====================================================

        normalized = re.sub(
            r"^[\s,;:!?]+",
            "",
            normalized,
        )

        normalized = re.sub(
            r"[\s,;:!?]+$",
            "",
            normalized,
        )

        # =====================================================
        # DETECT TOPIC
        # =====================================================

        lower = normalized.lower()

        ai_topic = any(
            term in lower
            for term in [
                "ai",
                "artificial intelligence",
                "ai assistant",
                "ai assistants",
                "knowledge assistant",
                "knowledge assistants",
                "rag",
                "retrieval augmented generation",
                "llm",
                "large language model",
            ]
        )

        leave_topic = any(
            term in lower
            for term in [
                "leave",
                "employee",
                "employees",
                "leave management",
                "employee leave",
                "hr",
                "human resources",
            ]
        )

        # =====================================================
        # ADD AI SEARCH CONTEXT
        # =====================================================

        if ai_topic:

            normalized = (
                f"{normalized} "
                "enterprise AI assistants "
                "RAG retrieval augmented generation trends"
            )

        # =====================================================
        # ADD HR SEARCH CONTEXT
        # =====================================================

        elif leave_topic:

            normalized = (
                f"{normalized} "
                "employee leave management trends"
            )

        # =====================================================
        # ADD FRESHNESS
        # =====================================================

        freshness_terms = [
            "latest",
            "current",
            "today",
            "recent",
            "recently",
            "this week",
            "this month",
            "this year",
        ]

        # Recalculate after cleanup/topic processing.
        lower = normalized.lower()

        if not any(
            term in lower
            for term in freshness_terms
        ):

            normalized = (
                "latest "
                + normalized
            )

        # =====================================================
        # FINAL CLEANUP
        # =====================================================

        normalized = " ".join(
            normalized.split()
        )

        normalized = re.sub(
            r"^[\s,;:!?]+",
            "",
            normalized,
        )

        normalized = re.sub(
            r"[\s,;:!?]+$",
            "",
            normalized,
        )

        return normalized.strip()

    # =========================================================
    # REFRESH INDEXES
    # =========================================================

    def refresh_indexes(self):
        """
        Refresh retrieval indexes after document changes.
        """

        print(
            "\nRefreshing retrieval indexes..."
        )

        self.hybrid.reload_bm25()

        print(
            "✅ Retrieval indexes refreshed"
        )