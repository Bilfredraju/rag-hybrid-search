from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class EmbeddingGenerator:
    """Generates embeddings using the configured SentenceTransformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Model Loaded Successfully")

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        return self.model.encode(text).tolist()

    def generate_embeddings(self, texts):
        if not texts:
            return []
        return self.model.encode(texts).tolist()
