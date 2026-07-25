from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """

    def __init__(self):

        print("Loading CrossEncoder model...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
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