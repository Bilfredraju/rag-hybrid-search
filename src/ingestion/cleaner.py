import re
import json
from pathlib import Path


class TextCleaner:
    """
    Cleans extracted PDF text and saves processed documents.
    """

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text (str): Raw extracted text.

        Returns:
            str: Cleaned text.
        """

        # Replace multiple spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Replace multiple blank lines
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Remove leading/trailing spaces
        text = text.strip()

        return text

    def save_document(self, filename: str, pages: list, output_dir=None):
        """
        Save cleaned pages as a JSON file.

        Args:
            filename (str): Original PDF filename.
            pages (list): Cleaned pages.
            output_dir (str): Output directory.
        """

        if output_dir is None:
            from src.config import PROCESSED_DIR
            output_dir = PROCESSED_DIR
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{Path(filename).stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved cleaned document: {output_file}")