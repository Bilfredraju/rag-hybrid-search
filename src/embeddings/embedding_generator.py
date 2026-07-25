from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings using Sentence Transformers.
    """

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("✅ Model Loaded Successfully")

    def generate_embedding(self, text):
        """
        Generate embedding for a single text.

        Args:
            text (str)

        Returns:
            list
        """

        embedding = self.model.encode(text)

        return embedding.tolist()

    def generate_embeddings(self, texts):
        """
        Generate embeddings for multiple texts.

        Args:
            texts (list)

        Returns:
            list
        """

        embeddings = self.model.encode(texts)

        return embeddings.tolist()