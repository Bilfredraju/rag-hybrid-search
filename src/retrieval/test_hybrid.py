from src.retrieval.hybrid import HybridSearch


def main():

    hybrid = HybridSearch()

    query = "leave policy"

    semantic, bm25 = hybrid.search(query)

    print("\n")
    print("=" * 60)
    print("SEMANTIC RESULTS")
    print("=" * 60)

    docs = semantic["documents"][0]
    metas = semantic["metadatas"][0]

    for i in range(len(docs)):

        print(f"\nResult {i+1}")
        print(f"Source : {metas[i]['source']}")
        print(f"Page   : {metas[i]['page']}")
        print(docs[i][:300])

    print("\n")
    print("=" * 60)
    print("BM25 RESULTS")
    print("=" * 60)

    for score, chunk in bm25:

        print("\n")
        print(f"Score : {score:.2f}")
        print(f"Source : {chunk['source']}")
        print(f"Page : {chunk['page']}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()