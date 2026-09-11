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

    The pipeline uses a shared LLM instance from the
    RAG pipeline to avoid creating duplicate Groq clients.
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        self.router = QueryRouter()

        self.rag_pipeline = RAGPipeline()

        self.general_prompt = GeneralPromptBuilder()

        self.both_prompt = BothPromptBuilder()

        # Reuse the LLM created by RAGPipeline.
        self.llm = self.rag_pipeline.llm

        # WebResearch is responsible only for search
        # and evidence collection.
        self.web_research = WebResearch(
            max_results=5
        )

        print("\n✅ Universal AI Assistant Ready")

    # ========================================================
    # MAIN QUERY ROUTER
    # ========================================================

    def ask(self, query):
        """
        Route the user query to the appropriate
        assistant capability.
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
            f"\nQuery route: {route.value}"
        )

        # ----------------------------------------------------
        # DOCUMENT ROUTE
        # ----------------------------------------------------

        if route == QueryRoute.DOCUMENT:

            result = self.rag_pipeline.ask(
                query
            )

            result["route"] = route.value

            return result

        # ----------------------------------------------------
        # GENERAL ROUTE
        # ----------------------------------------------------

        if route == QueryRoute.GENERAL:

            prompt = self.general_prompt.build_prompt(
                query
            )

            answer = self.llm.generate(
                prompt
            )

            return {
                "question": query,
                "answer": answer,
                "sources": [],
                "confidence": None,
                "route": route.value,
            }

        # ----------------------------------------------------
        # CURRENT ROUTE
        # ----------------------------------------------------

        if route == QueryRoute.CURRENT:

            return self._ask_current(
                query
            )

        # ----------------------------------------------------
        # BOTH ROUTE
        # ----------------------------------------------------

        if route == QueryRoute.BOTH:

            return self._ask_both(
                query
            )

        raise RuntimeError(
            f"Unsupported query route: {route}"
        )

    # ========================================================
    # CURRENT INFORMATION ROUTE
    # ========================================================

    def _ask_current(self, query):
        """
        Answer a current-information question using
        web evidence and the shared LLM.

        WebResearch handles search and page fetching.
        This method handles answer generation.
        """

        print(
            "\nRunning web research..."
        )

        web_data = (
            self.web_research.collect_evidence(
                query
            )
        )

        evidence = web_data[
            "evidence"
        ]

        sources = web_data[
            "sources"
        ]

        web_available = web_data[
            "available"
        ]

        web_error = web_data[
            "error"
        ]

        print(
            f"Web evidence: "
            f"{len(evidence)} sources"
        )

        # ----------------------------------------------------
        # Web search unavailable
        # ----------------------------------------------------

        if not web_available:

            return {
                "question": query,
                "answer": (
                    "Web research is temporarily "
                    "unavailable. I couldn't retrieve "
                    "reliable current information "
                    "for this question."
                ),
                "sources": [],
                "confidence": None,
                "route": QueryRoute.CURRENT.value,
                "web_research": {
                    "available": False,
                    "sources_found": 0,
                    "error": web_error,
                },
            }

        # ----------------------------------------------------
        # No usable web evidence
        # ----------------------------------------------------

        if not evidence:

            return {
                "question": query,
                "answer": (
                    "I couldn't find reliable web "
                    "sources for this question."
                ),
                "sources": [],
                "confidence": None,
                "route": QueryRoute.CURRENT.value,
                "web_research": {
                    "available": True,
                    "sources_found": 0,
                    "error": None,
                },
            }

        # ----------------------------------------------------
        # Build web evidence context
        # ----------------------------------------------------

        context_parts = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            title = item.get(
                "title",
                "Unknown title",
            )

            url = item.get(
                "url",
                "",
            )

            content = item.get(
                "content",
                "",
            )

            context_parts.append(
                f"""
Web Evidence {index}
Title: {title}
URL: {url}
Content:
{content[:4000]}
"""
            )

        context = "\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # Build LLM prompt
        # ----------------------------------------------------

        prompt = f"""
You are a current-information research assistant.

Answer the user's question using ONLY the
web evidence provided below.

STRICT RULES:

- Use only the supplied web evidence.
- Do not invent facts.
- Do not use outside knowledge.
- Treat the supplied web evidence as the
  source of current information.
- If sources disagree, clearly state that.
- If the evidence is insufficient, clearly
  say so.
- Do not create citation markers such as
  [1], [2], 【1】, or similar.
- Do not create a bibliography.
- Do not mention the retrieval process.
- Do not claim that you personally visited
  a website.
- Answer the user's exact question.
- Keep the answer concise.
- Prefer 2-5 sentences unless more detail
  is necessary.

The application will return source metadata
separately.

Therefore, DO NOT write URLs, source numbers,
or citation markers inside the answer.

================ WEB EVIDENCE ================

{context}

===============================================

QUESTION:
{query}

ANSWER:
"""

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        try:

            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:

            print(
                f"⚠️ Current-route answer "
                f"generation failed: {exc}"
            )

            return {
                "question": query,
                "answer": (
                    "I found relevant current web "
                    "sources, but the AI answer "
                    "generation service is temporarily "
                    "unavailable."
                ),
                "sources": sources,
                "confidence": None,
                "route": QueryRoute.CURRENT.value,
                "web_research": {
                    "available": True,
                    "sources_found": len(
                        evidence
                    ),
                    "error": str(exc),
                },
            }

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "confidence": None,
            "route": QueryRoute.CURRENT.value,
            "web_research": {
                "available": True,
                "sources_found": len(
                    evidence
                ),
                "error": None,
            },
        }

    # ========================================================
    # BOTH ROUTE
    # ========================================================

    def _ask_both(self, query):
        """
        Answer questions that require both:
        - information from uploaded documents
        - current web information

        Web research is treated as optional evidence.
        If web research fails, document evidence can
        still be used.
        """

        # ====================================================
        # DOCUMENT RESEARCH
        # ====================================================

        print(
            "\nRunning document research..."
        )

        semantic_results, bm25_results = (
            self.rag_pipeline.hybrid.search(
                query
            )
        )

        fused_results = (
            self.rag_pipeline.rrf.fuse(
                semantic_results,
                bm25_results,
            )
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

        # ====================================================
        # DOCUMENT SOURCES
        # ====================================================

        document_sources = []

        seen_documents = set()

        for result in document_evidence:

            metadata = result.get(
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

            key = (
                source,
                page,
            )

            if key in seen_documents:
                continue

            seen_documents.add(
                key
            )

            document_sources.append(
                {
                    "type": "document",
                    "source": source,
                    "page": page,
                }
            )

        # ====================================================
        # WEB RESEARCH
        # ====================================================

        web_query = self._build_web_query(
            query
        )

        print(
            f"\nWeb research query: "
            f"{web_query}"
        )

        web_data = (
            self.web_research.collect_evidence(
                web_query
            )
        )

        web_evidence = web_data[
            "evidence"
        ]

        web_sources = web_data[
            "sources"
        ]

        web_available = web_data[
            "available"
        ]

        web_error = web_data[
            "error"
        ]

        print(
            f"Web evidence: "
            f"{len(web_evidence)} sources"
        )

        if not web_available:

            print(
                "⚠️ Web research unavailable. "
                "Continuing with document evidence."
            )

        # ====================================================
        # NO EVIDENCE
        # ====================================================

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

        # ====================================================
        # BUILD COMBINED PROMPT
        # ====================================================

        prompt = (
            self.both_prompt.build_prompt(
                query=query,
                document_results=document_evidence,
                web_results=web_evidence,
            )
        )

        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        try:

            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:

            print(
                f"⚠️ BOTH-route answer "
                f"generation failed: {exc}"
            )

            # ------------------------------------------------
            # Document-only fallback
            # ------------------------------------------------

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
                        "generation service is temporarily "
                        "unavailable."
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

        # ====================================================
        # COMBINE SOURCES
        # ====================================================

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

            seen_web_urls.add(
                url
            )

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

        # ====================================================
        # DOCUMENT CONFIDENCE
        # ====================================================

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
                    "score": max(
                        scores
                    ),
                    "reason": (
                        "document_evidence_available"
                    ),
                }

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

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

    # ========================================================
    # BUILD WEB QUERY
    # ========================================================

    @staticmethod
    def _build_web_query(query):
        """
        Convert a BOTH-route question into a cleaner
        web-search query by removing internal document
        references and comparison filler.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        normalized = query.lower().strip()

        # ----------------------------------------------------
        # Internal document references
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Comparison phrases
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Remove punctuation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Remove filler words
        # ----------------------------------------------------

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

        normalized = " ".join(
            words
        )

        # ----------------------------------------------------
        # Encourage current results
        # ----------------------------------------------------

        if (
            "latest" not in normalized
            and "current" not in normalized
            and "recent" not in normalized
        ):

            normalized = (
                "latest "
                + normalized
            )

        # ----------------------------------------------------
        # Add domain context
        # ----------------------------------------------------

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

    # ========================================================
    # REFRESH RETRIEVAL INDEXES
    # ========================================================

    def refresh_indexes(self):
        """
        Refresh BM25/retrieval indexes after document
        ingestion or deletion.
        """

        self.rag_pipeline.refresh_indexes()