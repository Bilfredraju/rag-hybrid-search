from src.config import (
    MIN_CONFIDENT_RESULTS,
    MIN_EVIDENCE_SCORE,
    MIN_RERANK_SCORE,
)


class RetrievalConfidence:
    """
    Evaluates retrieval quality and filters weak evidence.
    """

    def __init__(
        self,
        min_score=MIN_RERANK_SCORE,
        min_results=MIN_CONFIDENT_RESULTS,
        evidence_score=MIN_EVIDENCE_SCORE,
    ):
        self.min_score = float(min_score)
        self.min_results = int(min_results)
        self.evidence_score = float(evidence_score)

    def filter_evidence(self, documents):
        """
        Keep only documents whose reranker score meets
        the minimum evidence threshold.
        """

        if not documents:
            return []

        return [
            doc
            for doc in documents
            if doc.get("rerank_score") is not None
            and float(doc["rerank_score"])
            >= self.evidence_score
        ]

    def evaluate(self, documents):
        """
        Evaluate whether retrieved documents provide
        sufficient evidence.
        """

        if not documents:
            return {
                "confident": False,
                "score": None,
                "reason": "no_results",
            }

        scores = [
            float(doc["rerank_score"])
            for doc in documents
            if doc.get("rerank_score") is not None
        ]

        if not scores:
            return {
                "confident": False,
                "score": None,
                "reason": "no_scores",
            }

        best_score = max(scores)

        confident_documents = [
            doc
            for doc in documents
            if doc.get("rerank_score") is not None
            and float(doc["rerank_score"])
            >= self.min_score
        ]

        if len(confident_documents) < self.min_results:
            return {
                "confident": False,
                "score": best_score,
                "reason": "low_confidence",
            }

        return {
            "confident": True,
            "score": best_score,
            "reason": "sufficient_evidence",
        }