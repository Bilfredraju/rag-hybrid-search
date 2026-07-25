from src.ingestion.loader import DocumentLoader
from src.ingestion.parser import PDFParser
from src.ingestion.cleaner import TextCleaner
from src.chunking.chunker import DocumentChunker
from src.chunking.chunk_saver import ChunkSaver


def main():

    print("\n========== DOCUMENT INGESTION PIPELINE ==========\n")

    # Initialize components
    loader = DocumentLoader()
    parser = PDFParser()
    cleaner = TextCleaner()
    chunker = DocumentChunker()
    chunk_saver = ChunkSaver()

    # Load PDF documents
    pdf_files = loader.load_documents()

    # Process each document
    for pdf in pdf_files:

        print("\n" + "=" * 60)
        print(f"Processing: {pdf.name}")
        print("=" * 60)

        # Step 1: Parse PDF
        pages = parser.parse_document(pdf)

        print(f"📄 Total Pages: {len(pages)}")

        # Step 2: Clean pages
        cleaned_pages = []

        for page in pages:

            cleaned_text = cleaner.clean_text(page["text"])

            cleaned_pages.append({
                "source": page["source"],
                "page": page["page"],
                "total_pages": page["total_pages"],
                "text": cleaned_text
            })

        # Step 3: Save cleaned document
        cleaner.save_document(pdf.name, cleaned_pages)

        # Step 4: Create chunks
        chunks = chunker.chunk_documents(cleaned_pages)

        print(f"📚 Total Chunks Created: {len(chunks)}")

        # Step 5: Save chunks
        chunk_saver.save_chunks(pdf.name, chunks)

        # Step 6: Display sample chunk
        if chunks:

            print("\nSample Chunk")
            print("-" * 50)

            print(chunks[0]["text"][:300])

            print("\nChunk Metadata")
            print("-" * 50)

            print(f"Chunk ID     : {chunks[0]['chunk_id']}")
            print(f"Source       : {chunks[0]['source']}")
            print(f"Page         : {chunks[0]['page']}")
            print(f"Total Pages  : {chunks[0]['total_pages']}")

    print("\n" + "=" * 60)
    print("🎉 DOCUMENT INGESTION & CHUNKING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()