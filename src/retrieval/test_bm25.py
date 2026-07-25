from src.retrieval.bm25_search import BM25Search


def main():

    bm25 = BM25Search()

    query = "leave policy"

    results = bm25.search(query)

    print("\nBM25 Results\n")

    for score, chunk in results:

        print("=" * 60)
        print("Score :", score)
        print("Source:", chunk["source"])
        print("Page:", chunk["page"])
        print()
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()