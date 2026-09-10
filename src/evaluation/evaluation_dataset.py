from dataclasses import dataclass
from src.router.query_router import QueryRoute


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_route: QueryRoute
    expected_sources: tuple[str, ...] = ()
    expected_concepts: tuple[tuple[str, ...], ...] = ()
    description: str = ""


EVALUATION_DATASET = [
    EvaluationCase(
        question="Who manages Project Phoenix?",
        expected_route=QueryRoute.DOCUMENT,
        expected_sources=("Project_Phoenix.pdf",),
        expected_concepts=(
            ("AI Engineering Team",),
        ),
        description="Tests retrieval from the Project Phoenix document.",
    ),

    EvaluationCase(
        question="What is the leave policy?",
        expected_route=QueryRoute.DOCUMENT,
        expected_sources=("LeavePolicy.pdf",),
        expected_concepts=(
            ("leave",),
            ("policy",),
        ),
        description="Tests policy retrieval.",
    ),

    EvaluationCase(
        question="What is the capital of France?",
        expected_route=QueryRoute.GENERAL,
        expected_concepts=(
            ("Paris",),
        ),
        description="Tests general LLM knowledge.",
    ),

    EvaluationCase(
        question="Explain machine learning in simple terms.",
        expected_route=QueryRoute.GENERAL,
        expected_concepts=(
            ("data",),
            ("learning",),
        ),
        description="Tests general explanatory capability.",
    ),

    EvaluationCase(
        question="What is the latest AI news?",
        expected_route=QueryRoute.CURRENT,
        description="Tests current web research.",
    ),

    EvaluationCase(
        question="What is the current Bitcoin price?",
        expected_route=QueryRoute.CURRENT,
        description="Tests current information routing.",
    ),

    EvaluationCase(
        question=(
            "Based on Project Phoenix, how does it compare "
            "with the latest AI assistant trends?"
        ),
        expected_route=QueryRoute.BOTH,
        expected_sources=("Project_Phoenix.pdf",),
        expected_concepts=(
            ("Project Phoenix", "Phoenix"),
            (
                "AI assistant",
                "knowledge assistant",
                "document-grounded assistant",
            ),
            (
                "RAG",
                "retrieval-augmented generation",
            ),
        ),
        description="Tests document + current web synthesis.",
    ),

    EvaluationCase(
        question=(
            "Based on the leave policy, what are the latest "
            "trends in employee leave management?"
        ),
        expected_route=QueryRoute.BOTH,
        expected_sources=("LeavePolicy.pdf",),
        expected_concepts=(
            ("leave",),
            ("employee",),
        ),
        description=(
            "Tests policy evidence combined with "
            "current web research."
        ),
    ),
]