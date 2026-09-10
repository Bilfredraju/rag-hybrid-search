from src.generation.general_prompt import GeneralPromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline
from src.router.query_router import QueryRoute, QueryRouter
from src.web.web_research import WebResearch


class AssistantPipeline:
    """
    High-level universal AI assistant.

    Routes:
        DOCUMENT → Hybrid RAG
        GENERAL  → General LLM
        CURRENT  → Live Web Research
        BOTH     → Document + Web research (next phase)
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing Universal AI Assistant")
        print("=" * 60)

        self.router = QueryRouter()

        self.rag_pipeline = RAGPipeline()

        self.general_prompt = GeneralPromptBuilder()

        # Reuse the LLM already initialized by RAGPipeline.
        self.llm = self.rag_pipeline.llm

        # Live web research component.
        self.web_research = WebResearch(
            max_results=5
        )

        print("\n✅ Universal AI Assistant Ready")

    def ask(self, query):
        """
        Route and answer a user question.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        route = self.router.classify(query)

        print(
            f"\nQuery route: {route.value}"
        )

        # -------------------------------------------------
        # DOCUMENT
        # -------------------------------------------------

        if route == QueryRoute.DOCUMENT:

            result = self.rag_pipeline.ask(
                query
            )

            result["route"] = route.value

            return result

        # -------------------------------------------------
        # GENERAL
        # -------------------------------------------------

        if route == QueryRoute.GENERAL:

            prompt = (
                self.general_prompt
                .build_prompt(query)
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

        # -------------------------------------------------
        # CURRENT
        # -------------------------------------------------

        if route == QueryRoute.CURRENT:

            result = (
                self.web_research
                .research(query)
            )

            result["confidence"] = None
            result["route"] = route.value

            return result

        # -------------------------------------------------
        # BOTH
        # -------------------------------------------------

        if route == QueryRoute.BOTH:

            return {
                "question": query,
                "answer": (
                    "Combined document and web "
                    "research is not implemented yet."
                ),
                "sources": [],
                "confidence": None,
                "route": route.value,
            }

        raise RuntimeError(
            f"Unsupported query route: {route}"
        )

    def refresh_indexes(self):
        """
        Refresh document retrieval indexes.
        """

        self.rag_pipeline.refresh_indexes()