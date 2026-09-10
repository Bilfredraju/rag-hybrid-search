import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR


REGISTRY_FILE = DATA_DIR / "documents.json"


class DocumentRegistry:
    """
    Persistent registry for ingested documents.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else REGISTRY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._save([])

    def _load(self):
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError("Document registry must contain a list.")

            return data

        except (json.JSONDecodeError, OSError, ValueError):
            return []

    def _save(self, documents):
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(
                documents,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def list_documents(self):
        return self._load()

    def get_document(self, document_id):
        for document in self._load():
            if document["document_id"] == document_id:
                return document

        return None

    def add_document(
        self,
        document_id,
        filename,
        pages,
        chunks,
        status="indexed",
    ):
        documents = self._load()

        # Prevent duplicate document IDs.
        documents = [
            document
            for document in documents
            if document["document_id"] != document_id
        ]

        document = {
            "document_id": document_id,
            "filename": filename,
            "pages": int(pages),
            "chunks": int(chunks),
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        documents.append(document)
        self._save(documents)

        return document

    def update_status(self, document_id, status):
        documents = self._load()

        for document in documents:
            if document["document_id"] == document_id:
                document["status"] = status
                self._save(documents)
                return document

        return None

    def remove_document(self, document_id):
        documents = self._load()

        remaining = [
            document
            for document in documents
            if document["document_id"] != document_id
        ]

        removed = len(documents) != len(remaining)

        if removed:
            self._save(remaining)

        return removed