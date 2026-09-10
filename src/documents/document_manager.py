from pathlib import Path

from src.chunking.chunk_saver import ChunkSaver
from src.chunking.chunker import DocumentChunker
from src.documents.registry import DocumentRegistry
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.ingestion.parser import PDFParser


class DocumentManager:
    """
    Coordinates the complete document lifecycle.

    Responsibilities:
    - Parse PDF
    - Create chunks
    - Generate embeddings
    - Synchronize ChromaDB
    - Persist chunk files
    - Maintain document registry
    - Delete documents safely
    """

    def __init__(self):
        print("Initializing Document Manager...")

        self.parser = PDFParser()
        self.chunker = DocumentChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.chunk_saver = ChunkSaver()
        self.registry = DocumentRegistry()

        print("✅ Document Manager Ready")

    # =========================================================
    # INGEST DOCUMENT
    # =========================================================

    def ingest_pdf(self, pdf_path):
        """
        Parse, chunk, embed, and index a PDF.

        If a document with the same filename already exists,
        its previous vectors and chunk file are replaced.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise ValueError(
                f"Path is not a file: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        filename = pdf_path.name

        print("\n" + "=" * 60)
        print(f"INGESTING DOCUMENT: {filename}")
        print("=" * 60)

        # -----------------------------------------------------
        # 1. Parse PDF
        # -----------------------------------------------------

        print("\nParsing PDF...")

        pages = self.parser.parse_document(
            pdf_path
        )

        if not pages:
            raise ValueError(
                f"No pages could be extracted from {filename}"
            )

        print(
            f"✅ Extracted {len(pages)} pages"
        )

        # -----------------------------------------------------
        # 2. Create chunks
        # -----------------------------------------------------

        print("\nCreating chunks...")

        chunks = self.chunker.chunk_documents(
            pages
        )

        if not chunks:
            raise ValueError(
                f"No chunks were created for {filename}"
            )

        document_id = chunks[0]["document_id"]

        print(
            f"✅ Created {len(chunks)} chunks"
        )

        print(
            f"Document ID: {document_id}"
        )

        # -----------------------------------------------------
        # 3. Generate embeddings BEFORE deleting old data
        # -----------------------------------------------------
        # This is important for safer replacement.
        #
        # If embedding generation fails, the old document
        # remains untouched.
        # -----------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("\nGenerating embeddings...")

        embeddings = (
            self.embedding_generator
            .generate_embeddings(texts)
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Number of embeddings does not match "
                "number of chunks."
            )

        print(
            f"✅ Generated {len(embeddings)} embeddings"
        )

        # -----------------------------------------------------
        # 4. Remove previous version
        # -----------------------------------------------------

        print(
            "\nSynchronizing previous document version..."
        )

        old_vectors = (
            self.vector_store.delete_source(
                filename
            )
        )

        old_chunks = self.chunk_saver.delete(
            filename
        )

        existing_document = (
            self.registry.get_document(
                document_id
            )
        )

        if existing_document:
            self.registry.remove_document(
                document_id
            )

        print(
            f"Previous vectors removed: {old_vectors}"
        )

        print(
            f"Previous chunk file removed: "
            f"{old_chunks}"
        )

        # -----------------------------------------------------
        # 5. Save new chunks
        # -----------------------------------------------------

        self.chunk_saver.save(
            filename,
            chunks
        )

        # -----------------------------------------------------
        # 6. Add vectors to ChromaDB
        # -----------------------------------------------------

        print("\nIndexing vectors...")

        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        # -----------------------------------------------------
        # 7. Register document
        # -----------------------------------------------------

        document = self.registry.add_document(
            document_id=document_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            status="indexed",
        )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("DOCUMENT INGESTION COMPLETE")
        print("=" * 60)

        print(
            f"Document : {filename}"
        )

        print(
            f"Pages    : {len(pages)}"
        )

        print(
            f"Chunks   : {len(chunks)}"
        )

        print(
            f"ID       : {document_id}"
        )

        print(
            f"Status   : {document['status']}"
        )

        return document

    # =========================================================
    # DELETE DOCUMENT
    # =========================================================

    def delete_document(self, document_id):
        """
        Delete a document from:

        - ChromaDB
        - persisted chunk storage
        - document registry
        """

        document = self.registry.get_document(
            document_id
        )

        if not document:
            raise ValueError(
                f"Document not found: {document_id}"
            )

        filename = document["filename"]

        print("\n" + "=" * 60)
        print(f"DELETING DOCUMENT: {filename}")
        print("=" * 60)

        # -----------------------------------------------------
        # 1. Delete ChromaDB vectors
        # -----------------------------------------------------

        chroma_deleted = (
            self.vector_store.delete_source(
                filename
            )
        )

        # -----------------------------------------------------
        # 2. Delete persisted chunks
        # -----------------------------------------------------

        chunks_deleted = (
            self.chunk_saver.delete(
                filename
            )
        )

        # -----------------------------------------------------
        # 3. Remove registry entry
        # -----------------------------------------------------

        self.registry.remove_document(
            document_id
        )

        print("\n" + "=" * 60)
        print("DOCUMENT DELETED")
        print("=" * 60)

        print(
            f"Document : {filename}"
        )

        print(
            f"Vectors  : {chroma_deleted}"
        )

        print(
            f"Chunk File Removed: {chunks_deleted}"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chroma_chunks_deleted": chroma_deleted,
            "chunk_file_deleted": chunks_deleted,
        }

    # =========================================================
    # LIST DOCUMENTS
    # =========================================================

    def list_documents(self):
        """
        Return all registered documents.
        """

        return self.registry.list_documents()