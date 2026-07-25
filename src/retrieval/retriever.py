from chromadb import PersistentClient
from src.embeddings.embedding_generator import EmbeddingGenerator


class Retriever:
    """
    Performs semantic search on the ChromaDB vector store.
    """

    def __init__(self):

        print("Step 1: Connecting to ChromaDB...")

        self.client = PersistentClient(path="data/chroma_db")

        print("✅ Connected to ChromaDB")

        print("Step 2: Loading collection...")

        self.collection = self.client.get_collection("rag_documents")

        print("✅ Collection loaded")

        print("Step 3: Loading embedding model...")

        self.embedding_generator = EmbeddingGenerator()

        print("✅ Embedding model loaded")

    def search(self, query, top_k=5):

        print("\nGenerating query embedding...")

        query_embedding = self.embedding_generator.generate_embedding(query)

        print("✅ Query embedding generated")

        print("Searching ChromaDB...")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        print("✅ Search completed")

        return results