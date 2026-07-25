import json
from pathlib import Path

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore


def main():

    chunks_folder = Path("data/chunks")

    all_chunks = []

    # Load every *_chunks.json file
    for chunk_file in chunks_folder.glob("*_chunks.json"):

        print(f"\nLoading {chunk_file.name}...")

        with open(chunk_file, "r", encoding="utf-8") as f:

            chunks = json.load(f)

            print(f"Loaded {len(chunks)} chunks")

            all_chunks.extend(chunks)

    print("\n===================================")
    print(f"Total Chunks Loaded : {len(all_chunks)}")
    print("===================================")

    generator = EmbeddingGenerator()

    texts = [chunk["text"] for chunk in all_chunks]

    print("\nGenerating embeddings...")

    embeddings = generator.generate_embeddings(texts)

    print("✅ Embeddings Generated")

    vector_store = VectorStore()

    print("\nSaving to ChromaDB...")

    vector_store.add_documents(all_chunks, embeddings)

    print("✅ Saved Successfully")

    print(f"\nTotal Vectors : {vector_store.count()}")


if __name__ == "__main__":
    main()