from src.generation.both_prompt import BothPromptBuilder
from src.generation.general_prompt import GeneralPromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline
from src.router.query_router import QueryRoute, QueryRouter
from src.web.web_research import WebResearch


class AssistantPipeline:
    """
    Universal AI Assistant.

    Routes questions to:
    - document RAG
    - general LLM knowledge
    - current web research
    - document + web research
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        self.router = QueryRouter()

        self.rag_pipeline = RAGPipeline()

        self.general_prompt = GeneralPromptBuilder()

        self.both_prompt = BothPromptBuilder()

        self.llm = self.rag_pipeline.llm

        self.web_research = WebResearch(
            max_results=5
        )

        print("\n✅ Universal AI Assistant Ready")

    def ask(self, query):
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        query = query.strip()

        route = self.router.classify(query)

        print(
            f"\nQuery route: {route.value}"
        )

        # --------------------------------------------------
        # DOCUMENT ROUTE
        # --------------------------------------------------

        if route == QueryRoute.DOCUMENT:

            result = self.rag_pipeline.ask(query)

            result["route"] = route.value

            return result

        # --------------------------------------------------
        # GENERAL ROUTE
        # --------------------------------------------------

        if route == QueryRoute.GENERAL:

            prompt = self.general_prompt.build_prompt(
                query
            )

            answer = self.llm.generate(prompt)

            return {
                "question": query,
                "answer": answer,
                "sources": [],
                "confidence": None,
                "route": route.value,
            }

        # --------------------------------------------------
        # CURRENT ROUTE
        # --------------------------------------------------

        if route == QueryRoute.CURRENT:

            result = self.web_research.research(
                query
            )

            result["confidence"] = None
            result["route"] = route.value

            return result

        # --------------------------------------------------
        # BOTH ROUTE
        # --------------------------------------------------

        if route == QueryRoute.BOTH:

            return self._ask_both(query)

        raise RuntimeError(
            f"Unsupported query route: {route}"
        )

    def _ask_both(self, query):

        # ==================================================
        # DOCUMENT RESEARCH
        # ==================================================

        print("\nRunning document research...")

        semantic_results, bm25_results = (
            self.rag_pipeline.hybrid.search(query)
        )

        fused_results = self.rag_pipeline.rrf.fuse(
            semantic_results,
            bm25_results,
        )

        reranked_results = (
            self.rag_pipeline.reranker.rerank(
                query,
                fused_results,
            )
        )

        document_evidence = (
            self.rag_pipeline.confidence
            .filter_evidence(
                reranked_results
            )[:3]
        )

        print(
            f"Document evidence: "
            f"{len(document_evidence)} chunks"
        )

        # ==================================================
        # DOCUMENT SOURCES
        # ==================================================

        document_sources = []

        seen_documents = set()

        for result in document_evidence:

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get("source")

            page = metadata.get("page")

            if source is None:
                continue

            key = (
                source,
                page,
            )

            if key in seen_documents:
                continue

            seen_documents.add(key)

            document_sources.append(
                {
                    "type": "document",
                    "source": source,
                    "page": page,
                }
            )

        # ==================================================
        # WEB RESEARCH
        # ==================================================

        web_query = self._build_web_query(query)

        print(
            f"\nWeb research query: "
            f"{web_query}"
        )

        web_data = self.web_research.collect_evidence(
            web_query
        )

        web_evidence = web_data["evidence"]

        web_sources = web_data["sources"]

        web_available = web_data["available"]

        web_error = web_data["error"]

        print(
            f"Web evidence: "
            f"{len(web_evidence)} sources"
        )

        if not web_available:

            print(
                "⚠️ Web research unavailable. "
                "Continuing with document evidence."
            )

        # ==================================================
        # NO EVIDENCE
        # ==================================================

        if (
            not document_evidence
            and not web_evidence
        ):

            if not web_available:

                answer = (
                    "I couldn't find enough information "
                    "in the provided documents, and "
                    "current web research is temporarily "
                    "unavailable."
                )

            else:

                answer = (
                    "I couldn't find enough reliable "
                    "information in the provided "
                    "documents or on the web."
                )

            return {
                "question": query,
                "answer": answer,
                "sources": [],
                "confidence": None,
                "route": QueryRoute.BOTH.value,
                "web_research": {
                    "available": web_available,
                    "sources_found": len(
                        web_evidence
                    ),
                    "error": web_error,
                },
            }

        # ==================================================
        # BUILD PROMPT
        # ==================================================

        prompt = self.both_prompt.build_prompt(
            query=query,
            document_results=document_evidence,
            web_results=web_evidence,
        )

        # ==================================================
        # GENERATE ANSWER
        # ==================================================

        try:

            answer = self.llm.generate(prompt)

        except Exception as exc:

            print(
                f"⚠️ BOTH-route answer generation "
                f"failed: {exc}"
            )

            # ----------------------------------------------
            # Graceful document-only fallback
            # ----------------------------------------------

            if document_evidence:

                fallback_parts = []

                for result in document_evidence:

                    content = result.get(
                        "document",
                        "",
                    ).strip()

                    if not content:
                        continue

                    fallback_parts.append(
                        content
                    )

                if fallback_parts:

                    answer = (
                        "I found relevant information "
                        "in the provided documents, but "
                        "the AI answer generation service "
                        "is temporarily unavailable.\n\n"
                        + "\n\n".join(
                            fallback_parts[:2]
                        )
                    )

                else:

                    answer = (
                        "I found relevant document "
                        "evidence, but the AI answer "
                        "generation service is "
                        "temporarily unavailable."
                    )

            elif web_evidence:

                answer = (
                    "I found relevant web sources, "
                    "but the AI answer generation "
                    "service is temporarily "
                    "unavailable."
                )

            else:

                answer = (
                    "The AI answer generation "
                    "service is temporarily "
                    "unavailable."
                )

        # ==================================================
        # SOURCES
        # ==================================================

        sources = []

        sources.extend(
            document_sources
        )

        seen_web_urls = set()

        for source in web_sources:

            url = source.get(
                "url"
            )

            if not url:
                continue

            if url in seen_web_urls:
                continue

            seen_web_urls.add(url)

            sources.append(
                {
                    "type": "web",
                    "title": source.get(
                        "title",
                        "",
                    ),
                    "url": url,
                    "fetched": source.get(
                        "fetched",
                        False,
                    ),
                }
            )

        # ==================================================
        # DOCUMENT CONFIDENCE
        # ==================================================

        confidence = None

        if document_evidence:

            scores = [
                float(
                    item.get(
                        "rerank_score",
                        0.0,
                    )
                )
                for item in document_evidence
                if item.get(
                    "rerank_score"
                ) is not None
            ]

            if scores:

                confidence = {
                    "confident": True,
                    "score": max(scores),
                    "reason": (
                        "document_evidence_available"
                    ),
                }

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "route": QueryRoute.BOTH.value,
            "web_research": {
                "available": web_available,
                "sources_found": len(
                    web_evidence
                ),
                "error": web_error,
            },
        }

    @staticmethod
    def _build_web_query(query):

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        normalized = query.lower().strip()

        internal_phrases = [
            "based on our project phoenix",
            "based on project phoenix",
            "according to our project phoenix",
            "according to project phoenix",
            "from our project phoenix",
            "from project phoenix",
            "in our project phoenix",
            "in project phoenix",
            "project phoenix",
        ]

        for phrase in internal_phrases:

            normalized = normalized.replace(
                phrase,
                " ",
            )

        comparison_phrases = [
            "how does it compare with",
            "how does it compare to",
            "how does this compare with",
            "how does this compare to",
            "how does it compare",
            "how does this compare",
            "compare it with",
            "compare it to",
            "compare this with",
            "compare this to",
            "compare with",
            "compare to",
            "based on",
            "according to",
        ]

        for phrase in comparison_phrases:

            normalized = normalized.replace(
                phrase,
                " ",
            )

        punctuation = [
            ",",
            ".",
            "?",
            "!",
            ":",
            ";",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]

        for character in punctuation:

            normalized = normalized.replace(
                character,
                " ",
            )

        filler_words = {
            "it",
            "this",
            "that",
            "the",
            "with",
            "for",
            "and",
            "from",
            "our",
            "my",
            "does",
            "how",
        }

        words = [
            word
            for word in normalized.split()
            if word not in filler_words
        ]

        normalized = " ".join(words)

        if (
            "latest" not in normalized
            and "current" not in normalized
            and "recent" not in normalized
        ):

            normalized = (
                "latest "
                + normalized
            )

        query_parts = [
            normalized,
            "enterprise AI assistants",
            "knowledge assistants",
            "RAG",
        ]

        return " ".join(
            part.strip()
            for part in query_parts
            if part.strip()
        )

    def refresh_indexes(self):

        self.rag_pipeline.refresh_indexes()