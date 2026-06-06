"""
FAISS-backed vector index implementation for online retrieval.
"""

from __future__ import annotations

import asyncio
from typing import Any

import faiss
import numpy as np

from src.decorators import log_execution, measure_time
from src.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorIndex:
    """
    Vector index implementation using FAISS.

    This class wraps a FAISS index and provides async search operations
    with support for GPU acceleration.
    """

    def __init__(
        self,
        index_path: str,
        use_gpu: bool = False,
        gpu_device_ids: str = "0",
        search_concurrency_limit: int = 1,
    ) -> None:
        """
        Initialize the vector index.

        Args:
            index_path: Path to the FAISS index file.
            use_gpu: Whether to use GPU acceleration.
            gpu_device_ids: Comma-separated GPU device IDs.
            search_concurrency_limit: Maximum concurrent search operations.
        """
        self._index_path = index_path
        self._use_gpu = use_gpu
        self._gpu_device_ids = gpu_device_ids

        if use_gpu:
            search_concurrency_limit = 1

        self._search_semaphore = asyncio.Semaphore(search_concurrency_limit)
        self._index: faiss.Index | None = None
        self._gpu_resources: Any | None = None
        self._dimension: int = -1

        logger.info(
            f"Initializing FAISSVectorIndex from {index_path}, "
            f"use_gpu={use_gpu}, gpu_ids={gpu_device_ids}"
        )

    @property
    def dimension(self) -> int:
        """Get the vector dimension."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_loaded(self) -> bool:
        """Check if the index is loaded."""
        return self._index is not None

    @log_execution()
    def load(self) -> None:
        """
        Load the FAISS index from disk.

        Raises:
            RuntimeError: If index loading fails.
        """
        logger.info(f"Loading FAISS index from {self._index_path}")

        try:
            self._index = faiss.read_index(self._index_path)
            self._dimension = self._index.d

            logger.info(
                f"Index loaded successfully: ntotal={self._index.ntotal}, "
                f"dimension={self._dimension}"
            )

            if self._use_gpu:
                self._move_to_gpu()

        except Exception as exc:
            logger.error(f"Failed to load index: {exc}")
            raise RuntimeError(f"Failed to load FAISS index: {exc}") from exc

    def _move_to_gpu(self) -> None:
        """
        Move the index to GPU.

        Falls back to CPU if GPU initialization fails.
        """
        try:
            logger.info(
                f"Moving index to GPU (CUDA_VISIBLE_DEVICES={self._gpu_device_ids})"
            )

            self._gpu_resources = faiss.StandardGpuResources()
            logger.debug("GPU resources initialized")

            gpu_options = faiss.GpuClonerOptions()
            gpu_options.useFloat16 = True

            self._index = faiss.index_cpu_to_gpu(
                self._gpu_resources, 0, self._index, gpu_options
            )

            logger.info("Index successfully moved to GPU")

        except Exception as exc:
            logger.warning(f"Failed to move index to GPU: {exc}")
            logger.info("Falling back to CPU index")
            self._use_gpu = False
            self._gpu_resources = None
            self._search_semaphore = asyncio.Semaphore(128)

    @measure_time(threshold_ms=50)
    async def search(
        self,
        query_vectors: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query_vectors: Query vectors with shape (num_queries, dimension).
            top_k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

        async with self._search_semaphore:
            logger.debug(
                f"Searching index: num_queries={query_vectors.shape[0]}, top_k={top_k}"
            )
            return await asyncio.to_thread(self._index.search, query_vectors, top_k)
