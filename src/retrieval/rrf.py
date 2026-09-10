from src.config import RRF_K


class ReciprocalRankFusion:
    """
    Combines semantic and BM25 rankings using stable chunk IDs.
    """

    def __init__(
        self,
        k=RRF_K,
        semantic_weight=0.7,
        bm25_weight=0.3,
    ):
        self.k = k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def fuse(
        self,
        semantic_results,
        bm25_results,
    ):
        """
        Fuse semantic and BM25 search results.

        Semantic results come directly from ChromaDB.
        BM25 results use the standardized dictionary format.
        """

        scores = {}

        # =====================================================
        # SEMANTIC RESULTS
        # =====================================================

        documents = semantic_results.get(
            "documents",
            [[]],
        )[0]

        metadatas = semantic_results.get(
            "metadatas",
            [[]],
        )[0]

        ids = semantic_results.get(
            "ids",
            [[]],
        )[0]

        for rank, (doc, meta) in enumerate(
            zip(documents, metadatas),
            start=1,
        ):

            meta = meta or {}

            doc_id = str(
                meta.get("chunk_id")
                or (
                    ids[rank - 1]
                    if ids and rank - 1 < len(ids)
                    else ""
                )
            )

            if not doc_id:
                raise ValueError(
                    "Semantic result is missing chunk_id"
                )

            scores.setdefault(
                doc_id,
                {
                    "score": 0.0,
                    "document": doc,
                    "metadata": meta,
                },
            )

            scores[doc_id]["score"] += (
                self.semantic_weight
                / (self.k + rank)
            )

        # =====================================================
        # BM25 RESULTS
        # =====================================================

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            if not isinstance(result, dict):
                raise ValueError(
                    "BM25 result must be a dictionary"
                )

            document = result.get("document")
            metadata = result.get("metadata", {})

            if not document:
                raise ValueError(
                    "BM25 result is missing document"
                )

            if not metadata:
                raise ValueError(
                    "BM25 result is missing metadata"
                )

            doc_id = str(
                metadata.get("chunk_id", "")
            )

            if not doc_id:
                raise ValueError(
                    "BM25 result is missing chunk_id"
                )

            if doc_id not in scores:

                scores[doc_id] = {
                    "score": 0.0,
                    "document": document,
                    "metadata": metadata,
                }

            scores[doc_id]["score"] += (
                self.bm25_weight
                / (self.k + rank)
            )

        # =====================================================
        # SORT BY FUSED SCORE
        # =====================================================

        return sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )