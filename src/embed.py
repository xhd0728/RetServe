"""
Embedding processor for corpus vectorization.

This module provides functionality for generating embeddings from text corpora
through an OpenAI-compatible embeddings API such as vLLM.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterator

import jsonlines
import numpy as np
from tqdm import tqdm

from src.config_loader import config_loader
from src.decorators import log_execution, measure_time
from src.embedding_client import OpenAIEmbeddingClient
from src.logging import get_logger
from src.settings import EmbedSettings, EmbeddingSettings

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Corpus Reader
# =============================================================================

class CorpusReader:
    """
    Reader for JSONL corpus files.
    
    This class handles reading and extracting text content from
    JSONL-formatted corpus files.
    
    Attributes:
        file_path: Path to the corpus file.
    """
    
    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the corpus reader.
        
        Args:
            file_path: Path to the JSONL corpus file.
        """
        self._file_path = Path(file_path)
        
        logger.info(f"Initialized CorpusReader with path={self._file_path}")
    
    @property
    def file_path(self) -> Path:
        """Get the corpus file path."""
        return self._file_path
    
    @log_execution()
    def count(self) -> int:
        """
        Count documents in the corpus file without loading them.

        Returns:
            Number of JSONL records.

        Raises:
            FileNotFoundError: If the corpus file does not exist.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self._file_path}")

        with open(self._file_path, "rb") as file:
            total = sum(1 for _ in file)

        logger.info(f"Counted {total} documents in corpus")
        return total

    def iter_batches(self, batch_size: int) -> Iterator[list[str]]:
        """
        Stream text contents from the corpus file in batches.

        Args:
            batch_size: Number of documents per yielded batch.

        Yields:
            Lists of text content strings.

        Raises:
            FileNotFoundError: If the corpus file does not exist.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self._file_path}")

        batch: list[str] = []
        with jsonlines.open(self._file_path, mode="r") as reader:
            for item in reader:
                content = item.get("contents", "")
                if not isinstance(content, str):
                    content = str(content)
                batch.append(content)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch


class OpenAIEmbeddingGenerator:
    """
    Streaming corpus encoder using an OpenAI-compatible embeddings API.

    This generator is designed for large corpora: it counts the JSONL records,
    streams texts in bounded batches, sends concurrent API requests through the
    shared OpenAIEmbeddingClient, and writes directly to a .npy memmap.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """
        Initialize the OpenAI-compatible generator.

        Args:
            settings: Embedding API settings.
        """
        self._settings = settings
        self._stream_batch_size = (
            settings.encode_batch_size
            or settings.batch_size * settings.concurrency_limit * 4
        )

        logger.info(
            f"Initialized OpenAIEmbeddingGenerator: url={settings.base_url}, "
            f"model={settings.model_name}, api_batch_size={settings.batch_size}, "
            f"concurrency={settings.concurrency_limit}, "
            f"stream_batch_size={self._stream_batch_size}, "
            f"normalize={settings.normalize}"
        )

    @measure_time()
    @log_execution()
    async def generate_to_file(
        self,
        corpus_reader: CorpusReader,
        output_path: str | Path,
    ) -> None:
        """
        Generate embeddings and write them directly to a .npy file.

        Args:
            corpus_reader: Streaming corpus reader.
            output_path: Destination .npy path.
        """
        output_path = Path(output_path)
        total_texts = corpus_reader.count()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f"{output_path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        if total_texts == 0:
            with open(tmp_path, "wb") as file:
                np.save(file, np.empty((0, 0), dtype=np.float32))
            tmp_path.replace(output_path)
            logger.warning("Corpus is empty, saved empty embeddings array")
            return

        logger.info(
            f"Generating {total_texts} embeddings via OpenAI-compatible API"
        )

        output_array: np.memmap | None = None
        offset = 0

        try:
            async with OpenAIEmbeddingClient(
                base_url=self._settings.base_url,
                model_name=self._settings.model_name,
                api_key=self._settings.resolved_api_key,
                batch_size=self._settings.batch_size,
                concurrency_limit=self._settings.concurrency_limit,
                request_timeout=self._settings.request_timeout,
                max_retries=self._settings.max_retries,
                normalize=self._settings.normalize,
                dimensions=self._settings.dimensions,
            ) as client:
                with tqdm(
                    total=total_texts,
                    desc="[openai] Embedding",
                    unit="doc",
                ) as progress_bar:
                    for batch_texts in corpus_reader.iter_batches(
                        self._stream_batch_size
                    ):
                        batch_embeddings = await client.embed(batch_texts)

                        if batch_embeddings.ndim != 2:
                            raise RuntimeError(
                                "Embedding API returned a non-matrix result: "
                                f"shape={batch_embeddings.shape}"
                            )

                        batch_size = batch_embeddings.shape[0]
                        if offset + batch_size > total_texts:
                            raise RuntimeError(
                                "Corpus changed while embedding: received more rows "
                                f"than counted ({offset + batch_size} > {total_texts})"
                            )

                        if output_array is None:
                            dimension = batch_embeddings.shape[1]
                            output_array = np.lib.format.open_memmap(
                                str(tmp_path),
                                mode="w+",
                                dtype=np.float32,
                                shape=(total_texts, dimension),
                            )
                            logger.info(
                                f"Created embedding memmap at {tmp_path}: "
                                f"shape={(total_texts, dimension)}"
                            )

                        output_array[offset:offset + batch_size] = batch_embeddings
                        offset += batch_size
                        output_array.flush()
                        progress_bar.update(batch_size)

            if offset != total_texts:
                raise RuntimeError(
                    "Corpus changed while embedding: received fewer rows than counted "
                    f"({offset} != {total_texts})"
                )
        except Exception:
            if output_array is not None:
                del output_array
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        if output_array is not None:
            output_array.flush()
            del output_array

        tmp_path.replace(output_path)

        logger.info(
            f"Generated and saved embeddings to {output_path} "
            f"(rows={offset}, dtype=float32)"
        )


# =============================================================================
# Embedding Processor
# =============================================================================

class EmbeddingProcessor:
    """
    Complete embedding pipeline processor.
    
    This class orchestrates the full embedding pipeline: reading corpus,
    generating embeddings, and saving results.
    
    Attributes:
        settings: Embedding configuration settings.
    """
    
    def __init__(self, settings: EmbedSettings) -> None:
        """
        Initialize the embedding processor.
        
        Args:
            settings: Embedding configuration settings.
        """
        self._settings = settings
        
        # Initialize components
        self._corpus_reader = CorpusReader(settings.data.corpus_path)
        self._embedding_generator = OpenAIEmbeddingGenerator(settings.embedding)
        
        logger.info("Initialized OpenAI-only EmbeddingProcessor")
    
    @property
    def settings(self) -> EmbedSettings:
        """Get the settings."""
        return self._settings
    
    @measure_time()
    @log_execution()
    async def process(self) -> None:
        """
        Execute the complete embedding pipeline.
        
        This method reads the corpus, generates embeddings, saves them,
        and performs cleanup.
        
        Raises:
            Exception: If any step of the pipeline fails.
        """
        try:
            logger.info("Step 1/1: Streaming corpus embeddings")
            await self._embedding_generator.generate_to_file(
                corpus_reader=self._corpus_reader,
                output_path=self._settings.data.embedding_path,
            )
            logger.info("Embedding pipeline completed successfully")
            
        except Exception as exc:
            logger.exception(f"Embedding pipeline failed: {exc}")
            raise


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Corpus Embedding Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="embed",
        help="Configuration file name (without .yaml extension)",
    )
    
    return parser.parse_args()


async def async_main() -> None:
    """
    Async main entry point.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    settings = config_loader.load_embed_settings(args.config)
    
    # Create and run processor
    processor = EmbeddingProcessor(settings)
    await processor.process()


def main() -> None:
    """
    Main entry point for the embedding tool.
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
