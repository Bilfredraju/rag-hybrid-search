from src.retrieval.hybrid import HybridSearch
from src.retrieval.rrf import ReciprocalRankFusion


def main():

    hybrid = HybridSearch()
    rrf = ReciprocalRankFusion()

    query = "leave policy"

    semantic, bm25 = hybrid.search(query)

    fused = rrf.fuse(semantic, bm25)

    print("\n" + "=" * 60)
    print("RRF RESULTS")
    print("=" * 60)

    for i, item in enumerate(fused[:5], start=1):

        meta = item["metadata"]

        print(f"\nResult {i}")
        print("-" * 60)
        print(f"Source : {meta['source']}")
        print(f"Page   : {meta['page']}")
        print(f"RRF Score : {item['score']:.5f}")
        print()
        print(item["document"][:300])


if __name__ == "__main__":
    main()