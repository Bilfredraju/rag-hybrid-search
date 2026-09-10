from src.generation.both_prompt import BothPromptBuilder
from src.generation.general_prompt import GeneralPromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline
from src.router.query_router import QueryRoute, QueryRouter
from src.web.web_research import WebResearch


class AssistantPipeline:
    """
    High-level universal AI assistant.

    Routes:

        DOCUMENT → Hybrid RAG

        GENERAL → General LLM

        CURRENT → Live Web Research

        BOTH → Document RAG + Live Web Research + Synthesis
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        self.router = QueryRouter()

        self.rag_pipeline = RAGPipeline()

        self.general_prompt = GeneralPromptBuilder()

        self.both_prompt = BothPromptBuilder()

        # Reuse the LLM instance already created by RAGPipeline.
        self.llm = self.rag_pipeline.llm

        self.web_research = WebResearch(
            max_results=5
        )

        print("\n✅ Universal AI Assistant Ready")

    # ======================================================
    # MAIN ENTRY POINT
    # ======================================================

    def ask(self, query):
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

        # ==================================================
        # DOCUMENT
        # ==================================================

        if route == QueryRoute.DOCUMENT:
            result = self.rag_pipeline.ask(
                query
            )

            result["route"] = route.value

            return result

        # ==================================================
        # GENERAL
        # ==================================================

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

        # ==================================================
        # CURRENT
        # ==================================================

        if route == QueryRoute.CURRENT:
            result = self.web_research.research(
                query
            )

            result["confidence"] = None

            result["route"] = route.value

            return result

        # ==================================================
        # BOTH
        # ==================================================

        if route == QueryRoute.BOTH:
            return self._ask_both(
                query
            )

        raise RuntimeError(
            f"Unsupported query route: {route}"
        )

    # ======================================================
    # BOTH ROUTE
    # ======================================================

    def _ask_both(self, query):
        """
        Combine internal document evidence with
        current web evidence.

        The document query and web query are kept
        logically separate so ambiguous document
        names do not pollute the web search.
        """

        print(
            "\nRunning document research..."
        )

        # --------------------------------------------------
        # DOCUMENT EVIDENCE
        # --------------------------------------------------

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
            )
        )

        document_evidence = document_evidence[
            :3
        ]

        print(
            f"Document evidence: "
            f"{len(document_evidence)} chunks"
        )

        # --------------------------------------------------
        # DOCUMENT SOURCES
        # --------------------------------------------------

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

            seen_documents.add(key)

            document_sources.append(
                {
                    "type": "document",
                    "source": source,
                    "page": page,
                }
            )

        # --------------------------------------------------
        # WEB QUERY
        # --------------------------------------------------

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

        print(
            f"Web evidence: "
            f"{len(web_evidence)} sources"
        )

        # --------------------------------------------------
        # NO EVIDENCE
        # --------------------------------------------------

        if (
            not document_evidence
            and not web_evidence
        ):
            return {
                "question": query,
                "answer": (
                    "I couldn't find enough reliable "
                    "information in the provided documents "
                    "or on the web."
                ),
                "sources": [],
                "confidence": None,
                "route": QueryRoute.BOTH.value,
            }

        # --------------------------------------------------
        # SYNTHESIS
        # --------------------------------------------------

        prompt = (
            self.both_prompt.build_prompt(
                query=query,
                document_results=document_evidence,
                web_results=web_evidence,
            )
        )

        answer = self.llm.generate(
            prompt
        )

        # --------------------------------------------------
        # COMBINE SOURCES
        # --------------------------------------------------

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

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

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

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "route": QueryRoute.BOTH.value,
        }

    # ======================================================
    # WEB QUERY BUILDER
    # ======================================================

    @staticmethod
    def _build_web_query(query):
        """
        Convert a combined document + current query
        into a clean web-focused search query.

        Example:

            Based on Project Phoenix, how does it compare
            with the latest AI assistant trends?

        becomes:

            latest AI assistant trends enterprise
            knowledge assistants RAG
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        normalized = (
            query.lower()
            .strip()
        )

        # --------------------------------------------------
        # Remove internal-document references.
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Remove comparison language.
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Remove punctuation.
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Remove filler words that don't help search.
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Explicitly preserve current/trend intent.
        # --------------------------------------------------

        if (
            "latest" not in normalized
            and "current" not in normalized
            and "recent" not in normalized
        ):
            normalized = (
                "latest " + normalized
            )

        # --------------------------------------------------
        # Add enterprise AI context.
        # --------------------------------------------------

        query_parts = [
            normalized,
            "enterprise AI assistants",
            "knowledge assistants",
            "RAG",
        ]

        final_query = " ".join(
            part.strip()
            for part in query_parts
            if part.strip()
        )

        return final_query

    # ======================================================
    # INDEX REFRESH
    # ======================================================

    def refresh_indexes(self):
        self.rag_pipeline.refresh_indexes()