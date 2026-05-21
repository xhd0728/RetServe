"""
FAISS index builder for vector similarity search.

This module provides functionality for building and saving FAISS indices
from embedding vectors, supporting efficient batch processing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from src.config_loader import config_loader
from src.decorators import log_execution, measure_time
from src.logging import get_logger
from src.settings import IndexBuildSettings

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Embedding Loader
# =============================================================================

class EmbeddingLoader:
    """
    Loader for embedding arrays from disk.
    
    This class handles loading pre-computed embeddings from numpy files.
    
    Attributes:
        file_path: Path to the embeddings file.
    """
    
    def __init__(self, file_path: str | Path, mmap: bool = True) -> None:
        """
        Initialize the embedding loader.
        
        Args:
            file_path: Path to the numpy embeddings file.
            mmap: Whether to memory-map the array instead of loading it fully.
        """
        self._file_path = Path(file_path)
        self._mmap = mmap
        
        logger.info(
            f"Initialized EmbeddingLoader with path={self._file_path}, mmap={mmap}"
        )
    
    @property
    def file_path(self) -> Path:
        """Get the embeddings file path."""
        return self._file_path
    
    @log_execution()
    def load(self) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Returns:
            Embedding array with shape (num_docs, dimension).
            
        Raises:
            FileNotFoundError: If the embeddings file does not exist.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self._file_path}")
        
        logger.info(f"Loading embeddings from {self._file_path}")
        
        embeddings = np.load(
            self._file_path,
            mmap_mode="r" if self._mmap else None,
        )
        
        logger.info(
            f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )
        
        return embeddings


# =============================================================================
# FAISS Index Builder
# =============================================================================

class FAISSIndexBuilder:
    """
    Builder for FAISS vector indices.
    
    This class handles the construction of FAISS indices from embedding
    vectors with support for batch processing and progress tracking.
    
    Attributes:
        chunk_size: Number of vectors to add per batch.
        use_inner_product: Whether to use inner product (True) or L2 (False).
    """
    
    def __init__(
        self,
        chunk_size: int = 50000,
        use_inner_product: bool = True,
        normalize: bool = False,
    ) -> None:
        """
        Initialize the index builder.
        
        Args:
            chunk_size: Number of vectors to add per batch.
            use_inner_product: Whether to use inner product similarity.
            normalize: Whether to L2-normalize vectors before adding.
        """
        self._chunk_size = chunk_size
        self._use_inner_product = use_inner_product
        self._normalize = normalize
        
        logger.info(
            f"Initialized FAISSIndexBuilder: "
            f"chunk_size={chunk_size}, use_inner_product={use_inner_product}, "
            f"normalize={normalize}"
        )
    
    @property
    def chunk_size(self) -> int:
        """Get the chunk size."""
        return self._chunk_size
    
    def _create_base_index(self, dimension: int) -> faiss.Index:
        """
        Create the base FAISS index.
        
        Args:
            dimension: Vector dimension.
            
        Returns:
            Base FAISS index (flat index).
        """
        if self._use_inner_product:
            # Inner product similarity (cosine similarity for normalized vectors)
            return faiss.IndexFlatIP(dimension)
        else:
            # L2 distance
            return faiss.IndexFlatL2(dimension)
    
    @measure_time()
    @log_execution()
    def build(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Embedding vectors with shape (num_docs, dimension).
            
        Returns:
            Built FAISS index with ID mapping.
        """
        num_vectors, dimension = embeddings.shape
        
        logger.info(
            f"Building FAISS index: num_vectors={num_vectors}, dimension={dimension}"
        )
        
        # Create base index
        base_index = self._create_base_index(dimension)
        
        # Wrap with ID mapping for direct document ID lookup
        index = faiss.IndexIDMap2(base_index)
        
        # Generate vector IDs (sequential integers)
        vector_ids = np.arange(num_vectors, dtype=np.int64)
        
        # Add vectors in chunks with progress bar
        with tqdm(
            total=num_vectors,
            desc="[faiss] Indexing",
            unit="vec",
        ) as progress_bar:
            for chunk_start in range(0, num_vectors, self._chunk_size):
                chunk_end = min(chunk_start + self._chunk_size, num_vectors)
                
                if self._normalize:
                    chunk_embeddings = np.array(
                        embeddings[chunk_start:chunk_end],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                    faiss.normalize_L2(chunk_embeddings)
                else:
                    chunk_embeddings = np.ascontiguousarray(
                        embeddings[chunk_start:chunk_end],
                        dtype=np.float32,
                    )
                chunk_ids = vector_ids[chunk_start:chunk_end]
                
                index.add_with_ids(chunk_embeddings, chunk_ids)
                
                progress_bar.update(chunk_end - chunk_start)
        
        logger.info(f"Built index with {index.ntotal} vectors")
        return index


# =============================================================================
# Index Saver
# =============================================================================

class IndexSaver:
    """
    Saver for FAISS indices.
    
    This class handles saving FAISS indices to disk, including
    automatic GPU to CPU conversion when necessary.
    """
    
    def __init__(self, output_path: str | Path) -> None:
        """
        Initialize the index saver.
        
        Args:
            output_path: Path for saving the index.
        """
        self._output_path = Path(output_path)
        
        logger.info(f"Initialized IndexSaver with path={self._output_path}")
    
    @property
    def output_path(self) -> Path:
        """Get the output path."""
        return self._output_path
    
    def _convert_to_cpu_if_needed(self, index: faiss.Index) -> faiss.Index:
        """
        Convert GPU index to CPU if necessary.
        
        Args:
            index: FAISS index (CPU or GPU).
            
        Returns:
            CPU FAISS index.
        """
        try:
            # Check if it's a GPU index by class name
            class_name = index.__class__.__name__
            
            if "Gpu" in class_name:
                logger.info("Converting GPU index to CPU for saving")
                return faiss.index_gpu_to_cpu(index)
            
            # Also check wrapped indices
            if hasattr(index, "index"):
                inner_class_name = index.index.__class__.__name__
                if "Gpu" in inner_class_name:
                    logger.info("Converting wrapped GPU index to CPU for saving")
                    return faiss.index_gpu_to_cpu(index)
                    
        except Exception as exc:
            logger.warning(f"Could not determine index type: {exc}")
            logger.info("Treating as CPU index")
        
        return index
    
    @log_execution()
    def save(self, index: faiss.Index) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            index: FAISS index to save.
        """
        # Ensure output directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to CPU if needed (GPU indices cannot be saved directly)
        cpu_index = self._convert_to_cpu_if_needed(index)
        
        # Save index
        faiss.write_index(cpu_index, str(self._output_path))
        
        logger.info(
            f"Saved index to {self._output_path} "
            f"(ntotal={cpu_index.ntotal})"
        )


# =============================================================================
# Index Build Pipeline
# =============================================================================

class IndexBuildPipeline:
    """
    Complete index building pipeline.
    
    This class orchestrates the full index building pipeline: loading
    embeddings, building the index, and saving results.
    
    Attributes:
        settings: Index building configuration settings.
    """
    
    def __init__(self, settings: IndexBuildSettings) -> None:
        """
        Initialize the index build pipeline.
        
        Args:
            settings: Index building configuration settings.
        """
        self._settings = settings
        
        # Initialize components
        self._embedding_loader = EmbeddingLoader(
            settings.data.embedding_path,
            mmap=settings.index.mmap_embeddings,
        )
        
        self._index_builder = FAISSIndexBuilder(
            chunk_size=settings.index.chunk_size,
            use_inner_product=True,  # Use cosine similarity
            normalize=settings.index.normalize,
        )
        
        self._index_saver = IndexSaver(settings.data.index_path)
        
        logger.info("Initialized IndexBuildPipeline")
    
    @property
    def settings(self) -> IndexBuildSettings:
        """Get the settings."""
        return self._settings
    
    @measure_time()
    @log_execution()
    def build(self) -> None:
        """
        Execute the complete index building pipeline.
        
        This method loads embeddings, builds the index, and saves it.
        
        Raises:
            Exception: If any step of the pipeline fails.
        """
        try:
            # Step 1: Load embeddings
            logger.info("Step 1/3: Loading embeddings")
            embeddings = self._embedding_loader.load()
            
            if embeddings.size == 0:
                logger.warning("Embeddings are empty, nothing to index")
                return
            
            # Step 2: Build index
            logger.info("Step 2/3: Building FAISS index")
            index = self._index_builder.build(embeddings)
            
            # Step 3: Save index
            logger.info("Step 3/3: Saving index")
            self._index_saver.save(index)
            
            logger.info("Index building pipeline completed successfully")
            
        except Exception as exc:
            logger.exception(f"Index building pipeline failed: {exc}")
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
        description="FAISS Index Building Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="index",
        help="Configuration file name (without .yaml extension)",
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the index building tool.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    settings = config_loader.load_index_settings(args.config)
    
    # Create and run pipeline
    pipeline = IndexBuildPipeline(settings)
    pipeline.build()


if __name__ == "__main__":
    main()
