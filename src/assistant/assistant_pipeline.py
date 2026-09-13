from src.generation.both_prompt import BothPromptBuilder
from src.generation.general_prompt import GeneralPromptBuilder
from src.generation.llm import LLM
from src.generation.prompt_builder import PromptBuilder
from src.router.query_router import QueryRoute, QueryRouter
from src.web.web_research import WebResearch

from src.retrieval.confidence import RetrievalConfidence
from src.retrieval.hybrid import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import ReciprocalRankFusion

from src.config import RERANK_TOP_K


class AssistantPipeline:
    """
    Universal AI Assistant.

    Routes queries to:

    DOCUMENT
        Hybrid document RAG.

    GENERAL
        General LLM knowledge.

    CURRENT
        Current web research.

    BOTH
        Uploaded documents + current web research.

    The assistant supports graceful degradation when
    external web research is temporarily unavailable.
    """

    def __init__(self):

        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        # -----------------------------------------------------
        # DOCUMENT RAG
        # -----------------------------------------------------

        self.hybrid = HybridSearch()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()
        self.confidence = RetrievalConfidence()

        self.prompt_builder = PromptBuilder()

        # -----------------------------------------------------
        # LLM
        # -----------------------------------------------------

        self.llm = LLM()

        # -----------------------------------------------------
        # GENERAL
        # -----------------------------------------------------

        self.general_prompt_builder = (
            GeneralPromptBuilder()
        )

        # -----------------------------------------------------
        # BOTH
        # -----------------------------------------------------

        self.both_prompt_builder = (
            BothPromptBuilder()
        )

        # -----------------------------------------------------
        # ROUTER
        # -----------------------------------------------------

        self.router = QueryRouter()

        # -----------------------------------------------------
        # WEB
        # -----------------------------------------------------

        self.web_research = WebResearch()

        print(
            "\n✅ Universal Assistant Ready"
        )

    # =========================================================
    # DOCUMENT SEARCH
    # =========================================================

    def _document_search(
        self,
        query: str,
    ):

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
                "documents": [],
                "confidence": confidence,
            }

        evidence_documents = (
            self.confidence.filter_evidence(
                reranked_results
            )
        )

        if not evidence_documents:

            return {
                "documents": [],
                "confidence": {
                    **confidence,
                    "reason": (
                        "no_sufficient_evidence"
                    ),
                },
            }

        return {
            "documents": evidence_documents[
                :RERANK_TOP_K
            ],
            "confidence": confidence,
        }

    # =========================================================
    # DOCUMENT SOURCES
    # =========================================================

    @staticmethod
    def _document_sources(
        documents,
    ):

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

            key = (
                source,
                page,
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "type": "document",
                }
            )

        return sources

    # =========================================================
    # CURRENT WEB QUERY
    # =========================================================

    @staticmethod
    def _build_current_web_query(
        query: str,
    ) -> str:

        normalized = (
            query.lower().strip()
        )

        if (
            "project phoenix"
            in normalized
        ):
            return (
                "the latest AI assistant trends "
                "enterprise AI assistants RAG "
                "retrieval augmented generation trends"
            )

        if (
            "leave policy"
            in normalized
            or "employee leave"
            in normalized
        ):
            return (
                "what are the latest trends in "
                "employee leave management "
                "employee leave management trends"
            )

        if (
            "ai news"
            in normalized
            or "artificial intelligence news"
            in normalized
        ):
            return (
                query
                + " enterprise AI assistants "
                "RAG retrieval augmented generation trends"
            )

        return query

    # =========================================================
    # GENERAL
    # =========================================================

    def _ask_general(
        self,
        query: str,
    ):

        prompt = (
            self.general_prompt_builder
            .build_prompt(query)
        )

        answer = self.llm.generate(
            prompt
        )

        return {
            "question": query,
            "route": QueryRoute.GENERAL.value,
            "answer": answer,
            "sources": [],
            "confidence": None,
        }

    # =========================================================
    # DOCUMENT
    # =========================================================

    def _ask_document(
        self,
        query: str,
    ):

        result = self._document_search(
            query
        )

        documents = result[
            "documents"
        ]

        confidence = result[
            "confidence"
        ]

        if not documents:

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

        prompt = (
            self.prompt_builder.build_prompt(
                query,
                documents,
            )
        )

        answer = self.llm.generate(
            prompt
        )

        return {
            "question": query,
            "route": QueryRoute.DOCUMENT.value,
            "answer": answer,
            "sources": self._document_sources(
                documents
            ),
            "confidence": confidence,
        }

    # =========================================================
    # CURRENT
    # =========================================================

    def _ask_current(
        self,
        query: str,
    ):

        web_query = (
            self._build_current_web_query(
                query
            )
        )

        print(
            f"🔎 Web Query: {web_query}"
        )

        web_result = (
            self.web_research.research(
                web_query
            )
        )

        web_evidence = web_result.get(
            "evidence",
            [],
        )

        web_metadata = web_result.get(
            "web_research",
            {},
        )

        if not web_evidence:

            return {
                "question": query,
                "route": QueryRoute.CURRENT.value,
                "answer": (
                    "I’m unable to obtain reliable "
                    "current web information right now. "
                    "Please try again shortly."
                ),
                "sources": [],
                "confidence": {
                    "web_available": False,
                    "reason": web_metadata.get(
                        "error",
                        "no_web_evidence",
                    ),
                },
            }

        # -----------------------------------------------------
        # CURRENT PROMPT
        # -----------------------------------------------------

        evidence_parts = []

        for index, evidence in enumerate(
            web_evidence,
            start=1,
        ):

            evidence_type = evidence.get(
                "evidence_type",
                "unknown",
            )

            if (
                evidence_type
                == "verified"
            ):
                status = (
                    "VERIFIED WEB PAGE CONTENT"
                )

            else:
                status = (
                    "SEARCH RESULT FALLBACK — "
                    "PAGE NOT VERIFIED"
                )

            evidence_parts.append(
                f"""
WEB SOURCE {index}

Verification:
{status}

Title:
{evidence.get("title", "")}

URL:
{evidence.get("url", "")}

Content:
{evidence.get("content", "")[:5000]}
"""
            )

        evidence = "\n".join(
            evidence_parts
        )

        prompt = f"""
You are a current-information AI assistant.

Answer the user's question using the supplied web
evidence.

IMPORTANT EVIDENCE RULES:

1. Prefer VERIFIED WEB PAGE CONTENT.

2. SEARCH RESULT FALLBACK evidence comes only from
   search-engine snippets and the actual page was not
   verified.

3. Never claim that an unverified search result was
   directly fetched or verified.

4. If sources disagree, report the disagreement.

5. Do not invent facts that are not supported by the
   supplied evidence.

6. For prices, rates, statistics, or other changing
   values, identify the source and acknowledge that
   values can change.

7. Keep the answer concise and directly answer the
   question.

WEB EVIDENCE:

{evidence}

QUESTION:

{query}

ANSWER:
"""

        try:

            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:

            print(
                f"⚠️ Current answer generation failed: "
                f"{exc}"
            )

            return {
                "question": query,
                "route": QueryRoute.CURRENT.value,
                "answer": (
                    "I obtained current web information, "
                    "but I was unable to generate the "
                    "final answer right now. Please try "
                    "again shortly."
                ),
                "sources": [],
                "confidence": {
                    "web_available": True,
                    "reason": "generation_failed",
                },
            }

        sources = []

        for source in web_result.get(
            "sources",
            [],
        ):

            sources.append(
                {
                    "source": source.get(
                        "title",
                        "",
                    ),
                    "url": source.get(
                        "url",
                        "",
                    ),
                    "type": "web",
                    "verified": source.get(
                        "verified",
                        False,
                    ),
                    "evidence_type": source.get(
                        "evidence_type",
                        "unknown",
                    ),
                }
            )

        return {
            "question": query,
            "route": QueryRoute.CURRENT.value,
            "answer": answer,
            "sources": sources,
            "confidence": {
                "web_available": True,
                "verified_sources": web_metadata.get(
                    "verified_sources",
                    0,
                ),
                "search_fallback_sources": web_metadata.get(
                    "search_fallback_sources",
                    0,
                ),
            },
        }

    # =========================================================
    # BOTH
    # =========================================================

    def _ask_both(
        self,
        query: str,
    ):

        # -----------------------------------------------------
        # DOCUMENT
        # -----------------------------------------------------

        document_result = (
            self._document_search(
                query
            )
        )

        documents = document_result[
            "documents"
        ]

        document_confidence = (
            document_result[
                "confidence"
            ]
        )

        # -----------------------------------------------------
        # WEB
        # -----------------------------------------------------

        web_query = (
            self._build_current_web_query(
                query
            )
        )

        print(
            f"🔎 Web Query: {web_query}"
        )

        web_result = (
            self.web_research.research(
                web_query
            )
        )

        web_evidence = web_result.get(
            "evidence",
            [],
        )

        web_metadata = web_result.get(
            "web_research",
            {},
        )

        # -----------------------------------------------------
        # NO DOCUMENT + NO WEB
        # -----------------------------------------------------

        if (
            not documents
            and not web_evidence
        ):

            return {
                "question": query,
                "route": QueryRoute.BOTH.value,
                "answer": (
                    "I don't have enough reliable "
                    "information from the uploaded documents "
                    "or current web sources to answer "
                    "this question."
                ),
                "sources": [],
                "confidence": {
                    "document": document_confidence,
                    "web": web_metadata,
                },
            }

        # -----------------------------------------------------
        # DOCUMENT ONLY
        # -----------------------------------------------------

        if documents and not web_evidence:

            print(
                "⚠️ Web research unavailable; "
                "using document evidence only."
            )

            prompt = (
                self.prompt_builder.build_prompt(
                    query,
                    documents,
                )
            )

            answer = self.llm.generate(
                prompt
            )

            answer += (
                "\n\nNote: Current web research "
                "was unavailable for this request."
            )

            return {
                "question": query,
                "route": QueryRoute.BOTH.value,
                "answer": answer,
                "sources": self._document_sources(
                    documents
                ),
                "confidence": {
                    "document": document_confidence,
                    "web_available": False,
                },
            }

        # -----------------------------------------------------
        # WEB ONLY
        # -----------------------------------------------------

        if (
            not documents
            and web_evidence
        ):

            print(
                "⚠️ Document evidence unavailable; "
                "using web evidence only."
            )

            # Reuse the current-route generation logic
            # through a concise combined prompt.

            evidence_parts = []

            for index, evidence in enumerate(
                web_evidence,
                start=1,
            ):

                evidence_parts.append(
                    f"""
Web Evidence {index}
Type: {evidence.get("evidence_type", "unknown")}
Title: {evidence.get("title", "")}
Content:
{evidence.get("content", "")[:5000]}
"""
                )

            web_text = "\n".join(
                evidence_parts
            )

            prompt = f"""
Answer the user's question using the supplied
current web evidence.

Prefer verified web page content.

Search-result fallback evidence is unverified and
must not be described as directly fetched.

Do not invent facts.

QUESTION:
{query}

WEB EVIDENCE:
{web_text}

ANSWER:
"""

            try:
                answer = self.llm.generate(
                    prompt
                )

            except Exception as exc:

                print(
                    f"⚠️ Web-only generation failed: "
                    f"{exc}"
                )

                answer = (
                    "I obtained current web evidence, "
                    "but I could not generate the final "
                    "answer right now."
                )

            return {
                "question": query,
                "route": QueryRoute.BOTH.value,
                "answer": answer,
                "sources": [
                    {
                        "source": source.get(
                            "title",
                            "",
                        ),
                        "url": source.get(
                            "url",
                            "",
                        ),
                        "type": "web",
                        "verified": source.get(
                            "verified",
                            False,
                        ),
                    }
                    for source in web_result.get(
                        "sources",
                        [],
                    )
                ],
                "confidence": {
                    "document_available": False,
                    "web_available": True,
                    "verified_sources": web_metadata.get(
                        "verified_sources",
                        0,
                    ),
                    "search_fallback_sources": web_metadata.get(
                        "search_fallback_sources",
                        0,
                    ),
                },
            }

        # -----------------------------------------------------
        # DOCUMENT + WEB
        # -----------------------------------------------------

        try:

            prompt = (
                self.both_prompt_builder
                .build_prompt(
                    query,
                    documents,
                    web_evidence,
                )
            )

            answer = self.llm.generate(
                prompt
            )

        except Exception as exc:

            print(
                f"⚠️ Combined generation failed: "
                f"{exc}"
            )

            # Graceful fallback to document evidence.
            prompt = (
                self.prompt_builder.build_prompt(
                    query,
                    documents,
                )
            )

            try:

                answer = self.llm.generate(
                    prompt
                )

                answer += (
                    "\n\nNote: The current web "
                    "comparison could not be completed."
                )

            except Exception:

                answer = (
                    "I found relevant information in "
                    "the uploaded documents, but I was "
                    "unable to generate the final answer "
                    "right now."
                )

        document_sources = (
            self._document_sources(
                documents
            )
        )

        web_sources = [
            {
                "source": source.get(
                    "title",
                    "",
                ),
                "url": source.get(
                    "url",
                    "",
                ),
                "type": "web",
                "verified": source.get(
                    "verified",
                    False,
                ),
                "evidence_type": source.get(
                    "evidence_type",
                    "unknown",
                ),
            }
            for source in web_result.get(
                "sources",
                [],
            )
        ]

        return {
            "question": query,
            "route": QueryRoute.BOTH.value,
            "answer": answer,
            "sources": (
                document_sources
                + web_sources
            ),
            "confidence": {
                "document": document_confidence,
                "web_available": True,
                "verified_sources": web_metadata.get(
                    "verified_sources",
                    0,
                ),
                "search_fallback_sources": web_metadata.get(
                    "search_fallback_sources",
                    0,
                ),
            },
        }

    # =========================================================
    # PUBLIC ASK
    # =========================================================

    def ask(
        self,
        query: str,
    ):

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

            print(
                "📚 Using document knowledge..."
            )

            return self._ask_document(
                query
            )

        if route == QueryRoute.GENERAL:

            print(
                "🧠 Using general LLM knowledge..."
            )

            return self._ask_general(
                query
            )

        if route == QueryRoute.CURRENT:

            print(
                "🌐 Using current web research..."
            )

            return self._ask_current(
                query
            )

        if route == QueryRoute.BOTH:

            print(
                "📚🌐 Combining document + "
                "web research..."
            )

            return self._ask_both(
                query
            )

        raise RuntimeError(
            f"Unsupported query route: {route}"
        )

    # =========================================================
    # REFRESH
    # =========================================================

    def refresh_indexes(
        self,
    ):

        print(
            "\nRefreshing retrieval indexes..."
        )

        self.hybrid.reload_bm25()

        print(
            "✅ Retrieval indexes refreshed"
        )