"""
Corpus loading utilities for the retrieval service.
"""

from __future__ import annotations

import os

import orjson
from tqdm import tqdm

from src.decorators import log_execution
from src.logging import get_logger
from src.types import Document

logger = get_logger(__name__)


class JSONLCorpusLoader:
    """
    Corpus loader for JSONL format files.

    This loader reads documents from a JSONL file and parses them into
    Document objects with progress feedback.
    """

    def __init__(self, corpus_path: str) -> None:
        """
        Initialize the corpus loader.

        Args:
            corpus_path: Path to the JSONL corpus file.
        """
        self._corpus_path = corpus_path

        logger.info(f"Initialized JSONLCorpusLoader with path={corpus_path}")

    @log_execution()
    def load(self) -> list[Document]:
        """
        Load documents from the corpus file.

        Returns:
            List of Document objects.
        """
        logger.info(f"Loading corpus from {self._corpus_path}")

        file_size = os.path.getsize(self._corpus_path)
        documents: list[Document] = []

        with open(self._corpus_path, "rb") as file:
            with tqdm(
                total=file_size,
                desc="Loading corpus",
                unit="B",
                unit_scale=True,
                ncols=100,
            ) as progress_bar:
                for raw_line in file:
                    progress_bar.update(len(raw_line))

                    line_data = orjson.loads(raw_line)
                    document_id = str(line_data.get("id", ""))
                    contents = line_data.get("contents", "")

                    if not isinstance(contents, str):
                        contents = str(contents)

                    title, text = self._split_contents(contents)

                    documents.append(
                        Document(
                            id=document_id,
                            title=title,
                            text=text,
                            contents=contents,
                        )
                    )

        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents

    @staticmethod
    def _split_contents(contents: str) -> tuple[str, str]:
        """Split a corpus contents field into title and body text."""
        parts = contents.split("\n", 1)
        title = parts[0].strip()
        text = parts[1] if len(parts) > 1 else ""
        return title, text
