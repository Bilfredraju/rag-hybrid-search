import json
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Search:
    """
    BM25 keyword search over all chunk files.
    """

    def __init__(self):

        self.documents = []
        self.tokenized_documents = []

        chunks_folder = Path("data/chunks")

        print("Loading chunks for BM25...")

        for file in chunks_folder.glob("*_chunks.json"):

            with open(file, "r", encoding="utf-8") as f:

                chunks = json.load(f)

                for chunk in chunks:

                    self.documents.append(chunk)

                    self.tokenized_documents.append(
                        chunk["text"].lower().split()
                    )

        self.bm25 = BM25Okapi(self.tokenized_documents)

        print(f"Loaded {len(self.documents)} chunks.")

    def search(self, query, top_k=5):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self.documents),
            reverse=True,
            key=lambda x: x[0]
        )

        return ranked[:top_k]