from src.retrieval.retriever import Retriever
from src.retrieval.bm25_search import BM25Search
from src.retrieval.rrf import ReciprocalRankFusion


def main():

    query = "What is the leave policy?"

    print("=" * 60)
    print("HYBRID SEARCH TEST")
    print("=" * 60)

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    retriever = Retriever()

    semantic_results = retriever.search(
        query,
        top_k=5,
    )

    print("\n")
    print("=" * 60)
    print("SEMANTIC RESULTS")
    print("=" * 60)

    semantic_documents = semantic_results.get(
        "documents",
        [[]],
    )[0]

    semantic_metadatas = semantic_results.get(
        "metadatas",
        [[]],
    )[0]

    semantic_ids = semantic_results.get(
        "ids",
        [[]],
    )[0]

    for index, (
        document,
        metadata,
    ) in enumerate(
        zip(
            semantic_documents,
            semantic_metadatas,
        ),
        start=1,
    ):

        print(f"\nResult {index}")

        print(
            f"Source : "
            f"{metadata.get('source')}"
        )

        print(
            f"Page   : "
            f"{metadata.get('page')}"
        )

        print(
            f"Chunk  : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            document[:500]
        )

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    bm25 = BM25Search()

    bm25_results = bm25.search(
        query,
        top_k=5,
    )

    print("\n")
    print("=" * 60)
    print("BM25 RESULTS")
    print("=" * 60)

    for index, result in enumerate(
        bm25_results,
        start=1,
    ):

        metadata = result["metadata"]

        print(f"\nResult {index}")

        print(
            f"Score  : "
            f"{result['score']:.4f}"
        )

        print(
            f"Source : "
            f"{metadata.get('source')}"
        )

        print(
            f"Page   : "
            f"{metadata.get('page')}"
        )

        print(
            f"Chunk  : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            result["document"][:500]
        )

    # ========================================================
    # RRF FUSION
    # ========================================================

    rrf = ReciprocalRankFusion()

    fused_results = rrf.fuse(
        semantic_results,
        bm25_results,
    )

    print("\n")
    print("=" * 60)
    print("RRF FUSED RESULTS")
    print("=" * 60)

    for index, result in enumerate(
        fused_results[:5],
        start=1,
    ):

        metadata = result["metadata"]

        print(f"\nResult {index}")

        print(
            f"RRF Score : "
            f"{result['score']:.6f}"
        )

        print(
            f"Source    : "
            f"{metadata.get('source')}"
        )

        print(
            f"Page      : "
            f"{metadata.get('page')}"
        )

        print(
            f"Chunk     : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            result["document"][:500]
        )

    print("\n")
    print("=" * 60)
    print("HYBRID SEARCH TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()