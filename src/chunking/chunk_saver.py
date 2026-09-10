import json
from pathlib import Path

from src.config import CHUNKS_DIR


class ChunkSaver:
    """
    Persists and removes document chunk files.
    """

    def __init__(
        self,
        chunks_dir: Path | str | None = None,
    ):
        self.chunks_dir = (
            Path(chunks_dir)
            if chunks_dir
            else CHUNKS_DIR
        )

        self.chunks_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(self, source, chunks):
        """
        Save chunks for one document.
        """

        source_name = Path(source).stem

        output_file = (
            self.chunks_dir
            / f"{source_name}_chunks.json"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                chunks,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"✅ Saved {len(chunks)} chunks to "
            f"{output_file}"
        )

        return output_file

    def save_chunks(self, source, chunks):
        """
        Backward-compatible alias used by the existing
        ingestion pipeline.
        """

        return self.save(source, chunks)

    def delete(self, source):
        """
        Delete persisted chunks for one document.
        """

        source_name = Path(source).stem

        chunk_file = (
            self.chunks_dir
            / f"{source_name}_chunks.json"
        )

        if not chunk_file.exists():
            print(
                f"ℹ️ No chunk file found for {source}"
            )
            return False

        chunk_file.unlink()

        print(
            f"🗑️ Deleted chunk file: "
            f"{chunk_file.name}"
        )

        return True

    def delete_chunks(self, source):
        """
        Backward-compatible alias.
        """

        return self.delete(source)