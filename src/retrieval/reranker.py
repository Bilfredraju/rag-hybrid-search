from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL


class Reranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """

    def __init__(self):

        print("Loading CrossEncoder model...")

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

        print("✅ CrossEncoder Loaded")

    def rerank(self, query, documents):

        pairs = []

        for doc in documents:
            pairs.append((query, doc["document"]))

        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return documents