from src.chunking.chunk_saver import ChunkSaver


def main():
    print("=" * 60)
    print("CHUNK SAVER TEST")
    print("=" * 60)

    saver = ChunkSaver()

    test_source = "Phase2Test.pdf"

    test_chunks = [
        {
            "chunk_id": "phase2_test_document_chunk_000001",
            "document_id": "phase2_test_document",
            "source": test_source,
            "page": 1,
            "total_pages": 1,
            "text": "Phase 2 test chunk.",
        }
    ]

    print("\nSaving test chunks...")

    path = saver.save(
        test_source,
        test_chunks,
    )

    print(f"Created: {path}")

    print("\nDeleting test chunks...")

    deleted = saver.delete(
        test_source
    )

    print(
        f"Deleted: {deleted}"
    )

    print("\n✅ Chunk saver test completed.")


if __name__ == "__main__":
    main()