import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_DIR


class BM25Search:
    """
    BM25 keyword search over persisted document chunks.
    """

    def __init__(self, chunks_dir: Path | str | None = None):
        self.chunks_dir = (
            Path(chunks_dir)
            if chunks_dir
            else CHUNKS_DIR
        )

        self.chunks = []
        self.bm25 = None

        self._load_chunks()

    @staticmethod
    def _tokenize(text):
        """
        Basic normalized tokenizer.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def _load_chunks(self):
        print("Loading chunks for BM25...")

        self.chunks = []

        if not self.chunks_dir.exists():
            print("No chunks directory found.")
            return

        chunk_files = sorted(
            self.chunks_dir.glob("*_chunks.json")
        )

        for chunk_file in chunk_files:
            try:
                with chunk_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    data = json.load(file)

                if isinstance(data, list):
                    self.chunks.extend(data)

            except (OSError, json.JSONDecodeError) as exc:
                print(
                    f"⚠️ Could not load {chunk_file.name}: {exc}"
                )

        if not self.chunks:
            print("No chunks available for BM25.")
            return

        tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        print(
            f"Loaded {len(self.chunks)} chunks."
        )

    def search(self, query, top_k=5):
        """
        Search the BM25 index.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if self.bm25 is None:
            return []

        top_k = max(1, int(top_k))

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indexes:
            chunk = self.chunks[index]

            results.append(
                {
                    "document": chunk["text"],
                    "metadata": {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "total_pages": chunk["total_pages"],
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                    },
                    "score": float(scores[index]),
                }
            )

        return results

    def reload(self):
        """
        Reload the BM25 corpus after documents are added,
        replaced, or removed.
        """

        self._load_chunks()
        print("✅ BM25 index reloaded.")