"""
Embedding processor for corpus vectorization.

This module provides functionality for generating embeddings from text corpora
using the infinity_emb library for efficient batch processing.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
from infinity_emb import AsyncEngineArray, EngineArgs
from tqdm import tqdm

from src.config_loader import config_loader
from src.decorators import log_execution, measure_time
from src.logging import get_logger
from src.settings import EmbedSettings

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
    def read(self) -> list[str]:
        """
        Read text contents from the corpus file.
        
        Returns:
            List of text content strings.
            
        Raises:
            FileNotFoundError: If the corpus file does not exist.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self._file_path}")
        
        logger.info(f"Reading corpus from {self._file_path}")
        
        contents: list[str] = []
        
        with jsonlines.open(self._file_path, mode="r") as reader:
            for item in reader:
                content = item.get("contents", "")
                if not isinstance(content, str):
                    content = str(content)
                contents.append(content)
        
        logger.info(f"Read {len(contents)} documents from corpus")
        return contents


# =============================================================================
# Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """
    Generator for text embeddings using infinity_emb.
    
    This class provides efficient batch embedding generation with
    GPU support and progress tracking.
    
    Attributes:
        model_path: Path to the embedding model.
        batch_size: Batch size for embedding generation.
        device: Compute device (cuda/cpu).
    """
    
    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 4,
        gpu_device_ids: str = "0",
        pooling_method: str = "auto",
        better_transformer: bool = False,
        model_warmup: bool = False,
        trust_remote_code: bool = True,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the embedding generator.
        
        Args:
            model_path: Path to the embedding model directory.
            batch_size: Batch size for processing.
            gpu_device_ids: Comma-separated GPU device IDs.
            pooling_method: Pooling method for embeddings.
            better_transformer: Whether to use BetterTransformer.
            model_warmup: Whether to warm up the model.
            trust_remote_code: Whether to trust remote code.
            device: Compute device (cuda/cpu).
        """
        self._model_path = Path(model_path)
        self._batch_size = batch_size
        self._gpu_device_ids = gpu_device_ids
        self._pooling_method = pooling_method
        self._better_transformer = better_transformer
        self._model_warmup = model_warmup
        self._trust_remote_code = trust_remote_code
        self._device = device
        
        # Set CUDA visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device_ids
        
        logger.info(
            f"Initialized EmbeddingGenerator: "
            f"model={model_path}, batch_size={batch_size}, "
            f"device={device}, gpu_ids={gpu_device_ids}"
        )
    
    @property
    def model_path(self) -> Path:
        """Get the model path."""
        return self._model_path
    
    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch_size
    
    @property
    def effective_batch_size(self) -> int:
        """Get the effective batch size considering GPU count."""
        gpu_count = len(self._gpu_device_ids.split(","))
        return self._batch_size * gpu_count
    
    def _create_engine_args(self) -> EngineArgs:
        """
        Create engine arguments for infinity_emb.
        
        Returns:
            Configured EngineArgs instance.
        """
        return EngineArgs(
            model_name_or_path=str(self._model_path),
            batch_size=self._batch_size,
            bettertransformer=self._better_transformer,
            pooling_method=self._pooling_method,
            device=self._device,
            model_warmup=self._model_warmup,
            trust_remote_code=self._trust_remote_code,
        )
    
    @measure_time()
    @log_execution()
    async def generate(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            Embedding vectors with shape (len(texts), dimension).
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        total_texts = len(texts)
        effective_batch_size = self.effective_batch_size
        
        logger.info(
            f"Generating embeddings for {total_texts} texts "
            f"with effective batch size {effective_batch_size}"
        )
        
        # Create engine
        engine_args = self._create_engine_args()
        engine = AsyncEngineArray.from_args([engine_args])[0]
        
        embeddings: list[np.ndarray] = []
        
        async with engine:
            with tqdm(
                total=total_texts,
                desc="[infinity] Embedding",
                unit="doc",
            ) as progress_bar:
                for batch_start in range(0, total_texts, effective_batch_size):
                    batch_end = min(batch_start + effective_batch_size, total_texts)
                    batch_texts = texts[batch_start:batch_end]
                    
                    # Generate embeddings for batch
                    batch_embeddings, _ = await engine.embed(sentences=batch_texts)
                    embeddings.extend(batch_embeddings)
                    
                    progress_bar.update(len(batch_texts))
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        logger.info(f"Generated embeddings with shape {embeddings_array.shape}")
        return embeddings_array


# =============================================================================
# Embedding Saver
# =============================================================================

class EmbeddingSaver:
    """
    Saver for embedding arrays.
    
    This class handles saving embeddings to disk in numpy format.
    """
    
    def __init__(self, output_path: str | Path) -> None:
        """
        Initialize the embedding saver.
        
        Args:
            output_path: Path for saving embeddings.
        """
        self._output_path = Path(output_path)
        
        logger.info(f"Initialized EmbeddingSaver with path={self._output_path}")
    
    @property
    def output_path(self) -> Path:
        """Get the output path."""
        return self._output_path
    
    @log_execution()
    def save(self, embeddings: np.ndarray) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embedding array to save.
        """
        # Ensure output directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(self._output_path, embeddings)
        
        logger.info(
            f"Saved embeddings to {self._output_path} "
            f"(shape={embeddings.shape}, dtype={embeddings.dtype})"
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
        
        self._embedding_generator = EmbeddingGenerator(
            model_path=settings.model.path,
            batch_size=settings.model.batch_size,
            gpu_device_ids=settings.model.gpu_device_ids,
            pooling_method=settings.model.pooling_method,
            better_transformer=settings.model.better_transformer,
            model_warmup=settings.model.model_warmup,
            trust_remote_code=settings.model.trust_remote_code,
            device=settings.model.device,
        )
        
        self._embedding_saver = EmbeddingSaver(settings.data.embedding_path)
        
        logger.info("Initialized EmbeddingProcessor")
    
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
            # Step 1: Read corpus
            logger.info("Step 1/3: Reading corpus")
            corpus_texts = self._corpus_reader.read()
            
            if not corpus_texts:
                logger.warning("Corpus is empty, nothing to process")
                return
            
            # Step 2: Generate embeddings
            logger.info("Step 2/3: Generating embeddings")
            embeddings = await self._embedding_generator.generate(corpus_texts)
            
            # Step 3: Save embeddings
            logger.info("Step 3/3: Saving embeddings")
            self._embedding_saver.save(embeddings)
            
            # Cleanup
            logger.info("Cleaning up resources")
            del embeddings
            del corpus_texts
            gc.collect()
            
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
