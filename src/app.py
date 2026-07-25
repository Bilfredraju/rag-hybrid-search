from src.pipeline.rag_pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline()

    while True:

        query = input("\nAsk a question (type exit to quit): ")

        if query.lower() == "exit":
            break

        results = pipeline.retrieve(query)

        print("\n" + "=" * 80)
        print("TOP RETRIEVED DOCUMENTS")
        print("=" * 80)

        for i, result in enumerate(results, start=1):

            meta = result["metadata"]

            print(f"\nResult {i}")
            print("-" * 60)
            print(f"Source : {meta['source']}")
            print(f"Page   : {meta['page']}")
            print(f"Score  : {result['rerank_score']:.4f}\n")
            print(result["document"][:400])


if __name__ == "__main__":
    main()