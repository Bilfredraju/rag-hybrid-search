import json
from pathlib import Path

from src.config import CHUNKS_DIR
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore


def main():
    print("\n========== EMBEDDING & VECTOR INDEXING ==========\n")

    chunks_folder = Path(CHUNKS_DIR)

    if not chunks_folder.exists():
        print(f"❌ Chunks directory not found: {chunks_folder}")
        return

    chunk_files = sorted(
        chunks_folder.glob("*_chunks.json")
    )

    if not chunk_files:
        print("❌ No chunk files found.")
        return

    generator = EmbeddingGenerator()
    vector_store = VectorStore()

    total_indexed = 0

    for chunk_file in chunk_files:

        print("\n" + "=" * 60)
        print(f"Processing: {chunk_file.name}")
        print("=" * 60)

        # -------------------------------------------------
        # Load chunks for this document
        # -------------------------------------------------

        with chunk_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(file)

        if not chunks:
            print("⚠️ No chunks found. Skipping.")
            continue

        source = chunks[0]["source"]

        document_id = chunks[0].get(
            "document_id"
        )

        print(f"Source      : {source}")
        print(f"Document ID : {document_id}")
        print(f"Chunks      : {len(chunks)}")

        # -------------------------------------------------
        # Remove previous version
        # -------------------------------------------------

        print("\nRemoving previous vectors...")

        deleted = vector_store.delete_source(
            source
        )

        print(
            f"Previous vectors removed: {deleted}"
        )

        # -------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("\nGenerating embeddings...")

        embeddings = (
            generator.generate_embeddings(
                texts
            )
        )

        print(
            f"✅ Generated {len(embeddings)} embeddings"
        )

        # -------------------------------------------------
        # Store vectors
        # -------------------------------------------------

        print("\nSaving to ChromaDB...")

        vector_store.add_documents(
            chunks,
            embeddings,
        )

        total_indexed += len(chunks)

        print(
            f"✅ Indexed {len(chunks)} chunks"
        )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("VECTOR INDEXING COMPLETED")
    print("=" * 60)

    print(
        f"Chunks indexed this run : {total_indexed}"
    )

    print(
        f"Total vectors in ChromaDB: "
        f"{vector_store.count()}"
    )


if __name__ == "__main__":
    main()