from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Splits cleaned text into smaller chunks.
    """

    def __init__(self, chunk_size=500, chunk_overlap=100):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, pages):

        chunks = []
        chunk_id = 1

        for page in pages:

            split_texts = self.text_splitter.split_text(page["text"])

            for text in split_texts:

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": page["filename"],
                        "page": page["page"],
                        "text": text,
                    }
                )

                chunk_id += 1

        return chunks