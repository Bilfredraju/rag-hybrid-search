from src.documents.registry import DocumentRegistry


def main():
    registry = DocumentRegistry()

    print("=" * 60)
    print("DOCUMENT REGISTRY TEST")
    print("=" * 60)

    document = registry.add_document(
        document_id="test_document_001",
        filename="Test.pdf",
        pages=5,
        chunks=20,
    )

    print("\nAdded:")
    print(document)

    print("\nDocuments:")
    for item in registry.list_documents():
        print(item)

    registry.remove_document("test_document_001")

    print("\nAfter removal:")
    print(registry.list_documents())

    print("\n✅ Registry test completed")


if __name__ == "__main__":
    main()