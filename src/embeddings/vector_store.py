import chromadb


class VectorStore:
    """
    Handles storage and retrieval of document embeddings using ChromaDB.
    """

    def __init__(self):

        self.client = chromadb.PersistentClient(path="data/chroma_db")

        collection_name = "rag_documents"

        # Delete old collection before rebuilding
        try:
            self.client.delete_collection(collection_name)
            print("🗑️ Old collection deleted.")
        except Exception:
            print("No previous collection found.")

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, chunks, embeddings):
        """
        Store chunks and embeddings in ChromaDB.
        """

        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):

            # Remove ".pdf" from filename
            source_name = chunk["source"].replace(".pdf", "")

            # Create unique ID
            unique_id = f"{source_name}_{chunk['chunk_id']}"

            ids.append(unique_id)

            documents.append(chunk["text"])

            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "total_pages": chunk["total_pages"],
                "chunk_id": chunk["chunk_id"]
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"✅ Stored {len(ids)} chunks in ChromaDB.")

    def count(self):
        return self.collection.count()