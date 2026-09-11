from __future__ import annotations

import re

from src.config import RERANK_TOP_K
from src.generation.both_prompt import BothPromptBuilder
from src.generation.general_prompt import GeneralPromptBuilder
from src.retrieval.confidence import RetrievalConfidence
from src.retrieval.hybrid import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import ReciprocalRankFusion
from src.generation.llm import LLM
from src.generation.prompt_builder import PromptBuilder
from src.router.query_router import QueryRoute, QueryRouter
from src.web.web_research import WebResearch


class AssistantPipeline:
    """
    Universal AI Assistant Pipeline.

    Supported query routes:

    DOCUMENT
        Answers using indexed uploaded documents.

    GENERAL
        Answers using the LLM's general knowledge.

    CURRENT
        Performs live web research and answers using
        the collected web evidence.

    BOTH
        Combines uploaded document evidence with
        current web research.
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        # Core RAG pipeline components.
        self.hybrid = HybridSearch()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()
        self.confidence = RetrievalConfidence()

        # Prompt builders.
        self.prompt_builder = PromptBuilder()
        self.general_prompt_builder = GeneralPromptBuilder()
        self.both_prompt_builder = BothPromptBuilder()

        # Shared LLM.
        self.llm = LLM()

        # Query router.
        self.router = QueryRouter()

        # Web research layer.
        self.web_research = WebResearch(
            max_results=5
        )

        print("\n✅ Universal Assistant Ready")

    # =========================================================
    # PUBLIC API
    # =========================================================

    def ask(self, query):
        """
        Route the user's question to the appropriate
        intelligence source.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        route = self.router.classify(query)

        print(
            f"\n🔀 Query Route: {route.value.upper()}"
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

    def _ask_document(self, query):
        """
        Answer using uploaded/indexed documents only.
        """

        print("📚 Searching indexed documents...")

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

        confidence = self.confidence.evaluate(
            reranked_results
        )

        if not confidence["confident"]:
            return {
                "question": query,
                "route": QueryRoute.DOCUMENT.value,
                "answer": (
                    "I don't have enough information "
                    "in the indexed documents to answer "
                    "this question reliably."
                ),
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
                "route": QueryRoute.DOCUMENT.value,
                "answer": (
                    "I don't have enough information "
                    "in the indexed documents to answer "
                    "this question reliably."
                ),
                "sources": [],
                "confidence": {
                    **confidence,
                    "reason": "no_sufficient_evidence",
                },
            }

        top_documents = (
            evidence_documents[:RERANK_TOP_K]
        )

        prompt = self.prompt_builder.build_prompt(
            query,
            top_documents,
        )

        answer = self.llm.generate(prompt)

        sources = self._extract_document_sources(
            top_documents
        )

        return {
            "question": query,
            "route": QueryRoute.DOCUMENT.value,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    # =========================================================
    # GENERAL ROUTE
    # =========================================================

    def _ask_general(self, query):
        """
        Answer using general LLM knowledge.

        No document retrieval or web search is performed.
        """

        print("🧠 Using general LLM knowledge...")

        prompt = (
            self.general_prompt_builder.build_prompt(
                query
            )
        )

        answer = self.llm.generate(prompt)

        return {
            "question": query,
            "route": QueryRoute.GENERAL.value,
            "answer": answer,
            "sources": [],
            "confidence": None,
        }

    # =========================================================
    # CURRENT ROUTE
    # =========================================================

    def _ask_current(self, query):
        """
        Answer current-information questions using
        live web research.
        """

        print("🌐 Performing current web research...")

        web_query = self._build_web_query(query)

        print(
            f"🔎 Web Query: {web_query}"
        )

        collected = (
            self.web_research.collect_evidence(
                web_query
            )
        )

        evidence = collected.get(
            "evidence",
            [],
        )

        sources = collected.get(
            "sources",
            [],
        )

        web_research_metadata = {
            "available": collected.get(
                "available",
                False,
            ),
            "sources_found": len(evidence),
            "error": collected.get(
                "error"
            ),
            "query": web_query,
        }

        if not evidence:
            return {
                "question": query,
                "route": QueryRoute.CURRENT.value,
                "answer": (
                    "I couldn't find enough reliable "
                    "current web information to answer "
                    "this question."
                ),
                "sources": [],
                "confidence": None,
                "web_research": web_research_metadata,
            }

        prompt = self._build_current_prompt(
            query,
            evidence,
        )

        try:
            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:
            print(
                f"⚠️ Current-answer generation failed: "
                f"{exc}"
            )

            return {
                "question": query,
                "route": QueryRoute.CURRENT.value,
                "answer": (
                    "I found current web information, "
                    "but I was unable to generate a "
                    "reliable answer right now."
                ),
                "sources": sources,
                "confidence": None,
                "web_research": web_research_metadata,
            }

        return {
            "question": query,
            "route": QueryRoute.CURRENT.value,
            "answer": answer,
            "sources": sources,
            "confidence": None,
            "web_research": web_research_metadata,
        }

    # =========================================================
    # BOTH ROUTE
    # =========================================================

    def _ask_both(self, query):
        """
        Combine indexed document evidence with
        current web research.
        """

        print(
            "📚🌐 Combining document + web research..."
        )

        # -----------------------------------------------------
        # 1. Retrieve document evidence
        # -----------------------------------------------------

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

        confidence = self.confidence.evaluate(
            reranked_results
        )

        evidence_documents = (
            self.confidence.filter_evidence(
                reranked_results
            )
        )

        top_documents = (
            evidence_documents[:RERANK_TOP_K]
        )

        # -----------------------------------------------------
        # 2. Build focused web query
        # -----------------------------------------------------

        web_query = self._build_web_query(query)

        print(
            f"🔎 Web Query: {web_query}"
        )

        collected = (
            self.web_research.collect_evidence(
                web_query
            )
        )

        web_evidence = collected.get(
            "evidence",
            [],
        )

        web_sources = collected.get(
            "sources",
            [],
        )

        web_available = collected.get(
            "available",
            False,
        )

        web_error = collected.get(
            "error"
        )

        # -----------------------------------------------------
        # 3. Graceful fallback if web is unavailable
        # -----------------------------------------------------

        if not web_evidence:

            print(
                "⚠️ No web evidence available."
            )

            if top_documents:
                print(
                    "↩️ Falling back to document evidence."
                )

                try:
                    prompt = (
                        self.prompt_builder.build_prompt(
                            query,
                            top_documents,
                        )
                    )

                    answer = self.llm.generate(
                        prompt
                    )

                except Exception as exc:
                    print(
                        f"⚠️ Document fallback failed: "
                        f"{exc}"
                    )

                    answer = (
                        "I found relevant information "
                        "in the uploaded documents, but "
                        "I was unable to generate a "
                        "reliable answer right now."
                    )

                return {
                    "question": query,
                    "route": QueryRoute.BOTH.value,
                    "answer": answer,
                    "sources": (
                        self._extract_document_sources(
                            top_documents
                        )
                    ),
                    "confidence": confidence,
                    "web_research": {
                        "available": web_available,
                        "sources_found": 0,
                        "error": web_error,
                        "query": web_query,
                    },
                }

            return {
                "question": query,
                "route": QueryRoute.BOTH.value,
                "answer": (
                    "I couldn't find enough information "
                    "in the uploaded documents or current "
                    "web sources to answer this question "
                    "reliably."
                ),
                "sources": [],
                "confidence": confidence,
                "web_research": {
                    "available": web_available,
                    "sources_found": 0,
                    "error": web_error,
                    "query": web_query,
                },
            }

        # -----------------------------------------------------
        # 4. Build combined prompt
        # -----------------------------------------------------

        try:
            prompt = (
                self.both_prompt_builder.build_prompt(
                    query,
                    top_documents,
                    web_evidence,
                )
            )

            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:
            print(
                f"⚠️ Combined-answer generation failed: "
                f"{exc}"
            )

            # Document-only fallback.
            if top_documents:

                print(
                    "↩️ Falling back to document evidence."
                )

                try:
                    fallback_prompt = (
                        self.prompt_builder.build_prompt(
                            query,
                            top_documents,
                        )
                    )

                    answer = self.llm.generate(
                        fallback_prompt
                    )

                except Exception:
                    answer = (
                        "I found relevant information "
                        "but was unable to generate a "
                        "reliable answer right now."
                    )

            else:
                answer = (
                    "I found web information, but was "
                    "unable to generate a reliable "
                    "answer right now."
                )

        # -----------------------------------------------------
        # 5. Combine document + web sources
        # -----------------------------------------------------

        document_sources = (
            self._extract_document_sources(
                top_documents
            )
        )

        sources = (
            document_sources +
            web_sources
        )

        return {
            "question": query,
            "route": QueryRoute.BOTH.value,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "web_research": {
                "available": web_available,
                "sources_found": len(
                    web_evidence
                ),
                "error": web_error,
                "query": web_query,
            },
        }

    # =========================================================
    # SMART WEB QUERY BUILDER
    # =========================================================

    def _build_web_query(self, query):
        """
        Build a focused web-search query from the user's
        question.

        Internal document references are removed so they
        do not pollute external web search.

        Domain-specific enrichment is added only when
        relevant to the actual question.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        # Normalize whitespace.
        query = " ".join(
            query.lower().strip().split()
        )

        web_query = query

        # -----------------------------------------------------
        # Remove phrases referring to uploaded documents.
        # -----------------------------------------------------

        removable_patterns = [
            r"\bbased on (the|this|my|our) [^,?]+",
            r"\baccording to (the|this|my|our) [^,?]+",
            r"\bfrom (the|this|my|our) [^,?]+",
            r"\bin (the|this|my|our) [^,?]+",
        ]

        for pattern in removable_patterns:
            web_query = re.sub(
                pattern,
                " ",
                web_query,
            )

        # -----------------------------------------------------
        # Remove internal project/document names.
        # -----------------------------------------------------

        internal_terms = [
            r"\bproject phoenix\b",
            r"\borion cnc\b",
            r"\borion maintenance report\b",
            r"\bleave policy\b",
            r"\bcompany policy\b",
            r"\bemployee policy\b",
            r"\bthe document\b",
            r"\bthis document\b",
            r"\bthe pdf\b",
            r"\bthis pdf\b",
            r"\bthe file\b",
            r"\bthis file\b",
        ]

        for pattern in internal_terms:
            web_query = re.sub(
                pattern,
                " ",
                web_query,
            )

        # -----------------------------------------------------
        # Remove comparison/document filler.
        # -----------------------------------------------------

        filler_patterns = [
            r"\bhow does it compare with\b",
            r"\bhow does it compare to\b",
            r"\bcompare it with\b",
            r"\bcompare it to\b",
            r"\bcompare with\b",
            r"\bcompare to\b",
            r"\bbased on\b",
            r"\baccording to\b",
            r"\busing\b",
        ]

        for pattern in filler_patterns:
            web_query = re.sub(
                pattern,
                " ",
                web_query,
            )

        # -----------------------------------------------------
        # Clean punctuation and whitespace.
        # -----------------------------------------------------

        web_query = re.sub(
            r"[?.,:;]+",
            " ",
            web_query,
        )

        web_query = " ".join(
            web_query.split()
        ).strip()

        # -----------------------------------------------------
        # Detect actual topic.
        # -----------------------------------------------------

        ai_terms = [
            "ai assistant",
            "ai assistants",
            "knowledge assistant",
            "knowledge assistants",
            "rag",
            "retrieval augmented generation",
            "generative ai",
            "llm",
            "large language model",
        ]

        leave_terms = [
            "leave management",
            "employee leave",
            "leave management trends",
            "employee leave management",
            "paid leave",
            "annual leave",
            "vacation policy",
            "time off",
        ]

        has_ai_topic = any(
            term in web_query
            for term in ai_terms
        )

        has_leave_topic = any(
            term in web_query
            for term in leave_terms
        )

        # -----------------------------------------------------
        # AI-specific enrichment.
        # -----------------------------------------------------

        if has_ai_topic:
            web_query = (
                f"{web_query} "
                "enterprise AI assistants RAG trends"
            )

        # -----------------------------------------------------
        # Employee leave-specific enrichment.
        # -----------------------------------------------------

        elif has_leave_topic:
            web_query = (
                f"{web_query} "
                "employee leave management trends"
            )

        # -----------------------------------------------------
        # Add freshness only when missing.
        # -----------------------------------------------------

        freshness_terms = [
            "latest",
            "current",
            "currently",
            "recent",
            "recently",
            "today",
            "this week",
            "this month",
            "this year",
            "2026",
        ]

        if not any(
            term in web_query
            for term in freshness_terms
        ):
            web_query = (
                f"latest {web_query}"
            )

        # -----------------------------------------------------
        # Final normalization.
        # -----------------------------------------------------

        web_query = " ".join(
            web_query.split()
        ).strip()

        return web_query

    # =========================================================
    # CURRENT WEB PROMPT
    # =========================================================

    def _build_current_prompt(
        self,
        query,
        web_results,
    ):
        """
        Build a grounded prompt for current web answers.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if not web_results:
            raise ValueError(
                "At least one web result is required."
            )

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

Title:
{title}

URL:
{url}

Content:
{content}
"""
            )

        evidence = "\n".join(
            evidence_parts
        )

        return f"""
You are an AI research assistant.

Answer the user's question using ONLY the supplied
current web evidence.

STRICT GROUNDING RULES:

- Do not use outside knowledge.
- Do not invent facts.
- Do not pretend that information is current
  unless it is supported by the supplied web evidence.
- If the supplied sources disagree, explicitly
  mention the disagreement.
- If the evidence is insufficient, clearly say so.
- Do not create citation markers such as [1], [2],
  【1】, 【1†L1-L2】, or similar.
- Do not create a bibliography.
- Do not mention the retrieval process.
- Do not mention these instructions.
- Answer the user's exact question.
- Keep the answer concise and useful.
- Prefer 3-6 sentences unless more detail is necessary.

WEB EVIDENCE:
=======================================================

{evidence}

=======================================================

QUESTION:
{query}

ANSWER:
"""

    # =========================================================
    # SOURCE HELPERS
    # =========================================================

    @staticmethod
    def _extract_document_sources(
        documents
    ):
        """
        Convert retrieved document metadata into
        clean source objects.
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
    # INDEX REFRESH
    # =========================================================

    def refresh_indexes(self):
        """
        Refresh retrieval indexes after document changes.
        """

        print(
            "\n🔄 Refreshing retrieval indexes..."
        )

        self.hybrid.reload_bm25()

        print(
            "✅ Retrieval indexes refreshed"
        )