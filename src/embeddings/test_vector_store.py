from src.embeddings.vector_store import VectorStore


def main():
    print("=" * 60)
    print("VECTOR STORE DOCUMENT SYNC TEST")
    print("=" * 60)

    store = VectorStore()

    print(f"\nCurrent total vectors: {store.count()}")

    test_document_id = "phase2_test_document"

    chunks = [
        {
            "chunk_id": "phase2_test_document_chunk_000001",
            "document_id": test_document_id,
            "source": "Phase2Test.pdf",
            "page": 1,
            "total_pages": 1,
            "text": "This is a Phase 2 test document.",
        },
        {
            "chunk_id": "phase2_test_document_chunk_000002",
            "document_id": test_document_id,
            "source": "Phase2Test.pdf",
            "page": 1,
            "total_pages": 1,
            "text": "This is the second test chunk.",
        },
    ]

    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
    ]

    print("\nAdding test document...")

    store.add_documents(
        chunks,
        embeddings,
    )

    print(
        f"Test document chunks: "
        f"{store.count_document(test_document_id)}"
    )

    print("\nDeleting test document...")

    deleted = store.delete_document(
        test_document_id
    )

    print(
        f"Deleted chunks: {deleted}"
    )

    print(
        f"Remaining test chunks: "
        f"{store.count_document(test_document_id)}"
    )

    print(
        f"\nFinal total vectors: {store.count()}"
    )

    print("\n✅ Vector store synchronization test completed.")


if __name__ == "__main__":
    main()