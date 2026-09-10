from pathlib import Path

import chromadb

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    ensure_data_directories,
)


class VectorStore:
    """
    Persistent ChromaDB vector store.

    Supports safe document-level synchronization:
    - upsert document chunks
    - remove all chunks belonging to a document
    - remove all chunks belonging to a source filename
    """

    def __init__(
        self,
        path: Path | str | None = None,
        collection_name: str | None = None,
    ):
        ensure_data_directories()

        db_path = Path(path) if path else CHROMA_DIR

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name or CHROMA_COLLECTION
        )

    def add_documents(self, chunks, embeddings):
        """
        Add or update chunks in ChromaDB.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have the same length"
            )

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])

            ids.append(chunk_id)
            documents.append(chunk["text"])

            metadatas.append(
                {
                    "source": chunk["source"],
                    "page": int(chunk["page"]),
                    "total_pages": int(chunk["total_pages"]),
                    "chunk_id": chunk_id,
                    "document_id": chunk["document_id"],
                }
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"✅ Upserted {len(ids)} chunks into ChromaDB."
        )

    def delete_document(self, document_id):
        """
        Remove every chunk belonging to a document.
        """

        document_id = str(document_id)

        results = self.collection.get(
            where={
                "document_id": document_id
            },
            include=["metadatas"],
        )

        ids = results.get("ids", [])

        if not ids:
            print(
                f"ℹ️ No ChromaDB chunks found for "
                f"document: {document_id}"
            )
            return 0

        self.collection.delete(ids=ids)

        print(
            f"🗑️ Deleted {len(ids)} chunks from ChromaDB "
            f"for document: {document_id}"
        )

        return len(ids)

    def delete_source(self, source):
        """
        Remove every chunk belonging to a source filename.

        This is useful when replacing or re-indexing a document.
        It also supports older ChromaDB records that may not contain
        the newer document_id metadata.
        """

        source = str(source)

        results = self.collection.get(
            where={
                "source": source
            },
            include=["metadatas"],
        )

        ids = results.get("ids", [])

        if not ids:
            print(
                f"ℹ️ No ChromaDB chunks found for source: {source}"
            )
            return 0

        self.collection.delete(ids=ids)

        print(
            f"🗑️ Deleted {len(ids)} old chunks from ChromaDB "
            f"for source: {source}"
        )

        return len(ids)

    def count(self):
        """
        Return total number of vectors in the collection.
        """

        return self.collection.count()

    def count_document(self, document_id):
        """
        Return number of chunks belonging to a document.
        """

        results = self.collection.get(
            where={
                "document_id": str(document_id)
            },
            include=["metadatas"],
        )

        return len(results.get("ids", []))