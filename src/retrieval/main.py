from src.retrieval.retriever import Retriever


def main():

    print("=" * 60)
    print("RAG SEMANTIC SEARCH TEST")
    print("=" * 60)

    print("\nCreating Retriever...\n")

    retriever = Retriever()

    query = "How many casual leave days are employees entitled to?"

    print(f"\nQuery : {query}\n")

    results = retriever.search(query)

    print("\n")
    print("=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        print(f"\nResult {i+1}")
        print("-" * 60)
        print(f"Source   : {metadatas[i]['source']}")
        print(f"Page     : {metadatas[i]['page']}")
        print(f"Distance : {distances[i]:.4f}")
        print("\nDocument:")
        print(documents[i][:500])

    print("\n")
    print("=" * 60)
    print("RETRIEVAL COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()