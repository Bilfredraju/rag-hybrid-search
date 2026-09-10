import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits documents into chunks with stable unique IDs.

    Chunk IDs are unique across the entire document,
    not reset for every page.
    """

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=100,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
                "",
            ],
        )

    @staticmethod
    def _document_id(source: str) -> str:
        """
        Create a stable identifier for a source document.
        """

        return hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()[:16]

    def chunk_documents(self, pages):
        """
        Split all pages into chunks.

        The chunk counter is global for the entire document,
        ensuring every chunk_id is unique.
        """

        chunks = []

        if not pages:
            return chunks

        document_id = self._document_id(
            pages[0]["source"]
        )

        chunk_index = 0

        for page in pages:

            split_text = self.text_splitter.split_text(
                page["text"]
            )

            for text in split_text:

                chunk_index += 1

                chunk_id = (
                    f"{document_id}_chunk_"
                    f"{chunk_index:06d}"
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source": page["source"],
                        "page": page["page"],
                        "total_pages": page["total_pages"],
                        "text": text,
                    }
                )

        return chunks