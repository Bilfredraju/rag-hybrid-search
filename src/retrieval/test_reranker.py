from src.retrieval.hybrid import HybridSearch
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker


def main():

    query = "leave policy"

    hybrid = HybridSearch()

    semantic, bm25 = hybrid.search(query)

    rrf = ReciprocalRankFusion()

    fused = rrf.fuse(semantic, bm25)

    reranker = Reranker()

    final_results = reranker.rerank(query, fused)

    print("\n" + "=" * 60)
    print("FINAL RERANKED RESULTS")
    print("=" * 60)

    for i, item in enumerate(final_results[:5], start=1):

        meta = item["metadata"]

        print(f"\nResult {i}")
        print("-" * 60)
        print(f"Source : {meta['source']}")
        print(f"Page   : {meta['page']}")
        print(f"Score  : {item['rerank_score']:.4f}")
        print()
        print(item["document"][:300])


if __name__ == "__main__":
    main()