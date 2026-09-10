from pathlib import Path

from src.documents.document_manager import DocumentManager
from src.config import DOCS_DIR


def main():

    print("=" * 60)
    print("DOCUMENT MANAGER TEST")
    print("=" * 60)

    manager = DocumentManager()

    test_pdf = Path(DOCS_DIR) / "Handbook.pdf"

    if not test_pdf.exists():
        raise FileNotFoundError(
            f"Test PDF not found: {test_pdf}"
        )

    # ========================================================
    # INGEST
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST: INGEST HANDBOOK")
    print("=" * 60)

    document = manager.ingest_pdf(
        test_pdf
    )

    print("\nRegistered Document:")
    print(document)

    # ========================================================
    # LIST
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST: LIST DOCUMENTS")
    print("=" * 60)

    documents = manager.list_documents()

    for item in documents:
        print(item)

    # ========================================================
    # VERIFY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST: VERIFY DOCUMENT")
    print("=" * 60)

    document_id = document["document_id"]

    stored = manager.registry.get_document(
        document_id
    )

    if stored is None:
        raise RuntimeError(
            "Document was not found in registry."
        )

    print(
        f"Document ID : {stored['document_id']}"
    )

    print(
        f"Filename    : {stored['filename']}"
    )

    print(
        f"Pages       : {stored['pages']}"
    )

    print(
        f"Chunks      : {stored['chunks']}"
    )

    print(
        f"Status      : {stored['status']}"
    )

    # ========================================================
    # VECTOR COUNT
    # ========================================================

    vector_count = (
        manager.vector_store.count_document(
            document_id
        )
    )

    print(
        f"\nChromaDB chunks for document: "
        f"{vector_count}"
    )

    if vector_count != document["chunks"]:
        raise RuntimeError(
            "ChromaDB chunk count does not match "
            "registry chunk count."
        )

    print(
        "\n✅ Document Manager test completed successfully."
    )


if __name__ == "__main__":
    main()