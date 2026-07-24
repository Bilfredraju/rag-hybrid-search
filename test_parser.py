from src.ingestion.loader import DocumentLoader
from src.ingestion.parser import PDFParser
from src.ingestion.cleaner import TextCleaner
from src.chunking.chunker import TextChunker


# Initialize classes
loader = DocumentLoader()
parser = PDFParser()
cleaner = TextCleaner()
chunker = TextChunker()


# Load PDF files
documents = loader.load_documents()

# Parse the first PDF
pages = parser.parse(documents[0])

# Clean every page
for page in pages:
    page["text"] = cleaner.clean_text(page["text"])

# Create chunks
chunks = chunker.chunk(pages)

# Display results
print("\n" + "=" * 60)
print(f"Document : {documents[0].name}")
print(f"Total Pages : {len(pages)}")
print(f"Total Chunks : {len(chunks)}")
print("=" * 60)

print("\nFirst Chunk Metadata")
print("-" * 30)

print(f"Chunk ID : {chunks[0]['chunk_id']}")
print(f"Source   : {chunks[0]['source']}")
print(f"Page     : {chunks[0]['page']}")

print("\nChunk Preview")
print("-" * 30)

print(chunks[0]["text"][:500])