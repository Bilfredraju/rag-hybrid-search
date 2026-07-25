class ReciprocalRankFusion:
    """
    Combines rankings from Semantic Search and BM25 Search.
    """

    def __init__(self, k=60):
        self.k = k

    def fuse(self, semantic_results, bm25_results):

        scores = {}

        # Semantic Search Results
        documents = semantic_results["documents"][0]
        metadatas = semantic_results["metadatas"][0]

        for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):

            doc_id = f"{meta['source']}_{meta['page']}"

            if doc_id not in scores:
                scores[doc_id] = {
                    "score": 0,
                    "document": doc,
                    "metadata": meta
                }

            scores[doc_id]["score"] += 1 / (self.k + rank)

        # BM25 Results
        for rank, (_, chunk) in enumerate(bm25_results, start=1):

            doc_id = f"{chunk['source']}_{chunk['page']}"

            if doc_id not in scores:
                scores[doc_id] = {
                    "score": 0,
                    "document": chunk["text"],
                    "metadata": {
                        "source": chunk["source"],
                        "page": chunk["page"]
                    }
                }

            scores[doc_id]["score"] += 1 / (self.k + rank)

        fused = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return fused