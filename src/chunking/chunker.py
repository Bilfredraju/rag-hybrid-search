from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits cleaned documents into smaller chunks while preserving metadata.
    """

    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_documents(self, pages):
        """
        Split cleaned pages into chunks.

        Args:
            pages (list): List of page dictionaries.

        Returns:
            list: List of chunk dictionaries.
        """

        chunks = []
        chunk_id = 1

        for page in pages:

            split_text = self.text_splitter.split_text(page["text"])

            for text in split_text:

                chunks.append({
                    "chunk_id": chunk_id,
                    "source": page["source"],
                    "page": page["page"],
                    "total_pages": page["total_pages"],
                    "text": text
                })

                chunk_id += 1

        return chunks