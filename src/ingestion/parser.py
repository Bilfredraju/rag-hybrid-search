import fitz  # PyMuPDF
from pathlib import Path


class PDFParser:
    """
    Extract text from PDF files page by page.
    """

    def parse_document(self, pdf_path: Path):
        """
        Parse a PDF file and extract text from each page.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            list: List of page dictionaries.
        """

        pages = []

        with fitz.open(pdf_path) as pdf:

            total_pages = len(pdf)

            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                pages.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "total_pages": total_pages,
                        "text": text,
                    }
                )

        return pages