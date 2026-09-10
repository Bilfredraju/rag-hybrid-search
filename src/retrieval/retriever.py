from chromadb import PersistentClient

from src.config import CHROMA_COLLECTION, CHROMA_DIR, SEMANTIC_TOP_K
from src.embeddings.embedding_generator import EmbeddingGenerator


class Retriever:
    """Performs semantic search against the persistent ChromaDB collection."""

    def __init__(self):
        print("Connecting to ChromaDB...")
        self.client = PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(CHROMA_COLLECTION)
        self.embedding_generator = EmbeddingGenerator()
        print("✅ Semantic retriever ready")

    def search(self, query, top_k=SEMANTIC_TOP_K):
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        top_k = max(1, int(top_k))
        available = self.collection.count()
        if available == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        top_k = min(top_k, available)
        query_embedding = self.embedding_generator.generate_embedding(query)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
