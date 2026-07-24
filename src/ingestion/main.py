from src.ingestion.loader import DocumentLoader
from src.ingestion.parser import PDFParser
from src.ingestion.cleaner import TextCleaner


def main():

    loader = DocumentLoader()
    parser = PDFParser()
    cleaner = TextCleaner()

    pdf_files = loader.load_documents()

    for pdf in pdf_files:

        print(f"\nProcessing: {pdf.name}")

        pages = parser.parse_document(pdf)

        cleaner.save_processed_text(pages)

    print("\n🎉 All documents processed successfully!")


if __name__ == "__main__":
    main()