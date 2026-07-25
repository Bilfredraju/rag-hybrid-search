import json
from pathlib import Path


class ChunkSaver:
    """
    Saves document chunks as JSON files.
    """

    def __init__(self, output_dir="data/chunks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_chunks(self, filename, chunks):
        """
        Save chunks to a JSON file.

        Args:
            filename (str): Original PDF filename
            chunks (list): List of chunk dictionaries
        """

        output_file = self.output_dir / f"{Path(filename).stem}_chunks.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)

        print(f"✅ Chunks saved: {output_file}")