"""
FAISS-backed vector index implementation for online retrieval.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
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
        search_workers: int | None = None,
        omp_threads: int | None = None,
    ) -> None:
        """
        Initialize the vector index.

        Args:
            index_path: Path to the FAISS index file.
            use_gpu: Whether to use GPU acceleration.
            gpu_device_ids: Comma-separated GPU device IDs.
            search_concurrency_limit: Maximum concurrent search operations.
            search_workers: Fixed worker count for CPU FAISS search calls.
            omp_threads: Optional FAISS OpenMP thread count.
        """
        self._index_path = index_path
        self._use_gpu = use_gpu
        self._gpu_device_ids = gpu_device_ids
        self._cpu_search_workers = search_workers or search_concurrency_limit
        self._omp_threads = omp_threads

        if use_gpu:
            search_concurrency_limit = 1

        self._search_semaphore = asyncio.Semaphore(search_concurrency_limit)
        self._executor = ThreadPoolExecutor(
            max_workers=search_concurrency_limit,
            thread_name_prefix="retserve-faiss",
        )
        self._index: faiss.Index | None = None
        self._gpu_resources: list[Any] | None = None
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
            if self._omp_threads is not None and hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(self._omp_threads)
                logger.info(f"Set FAISS OMP threads to {self._omp_threads}")

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
        Move the index to one GPU or shard it across multiple GPUs.

        Falls back to CPU if GPU initialization fails.
        """
        try:
            physical_ids, logical_ids = self._resolve_gpu_device_ids()
            logger.info(
                f"Moving index to {len(logical_ids)} GPU(s): "
                f"configured_ids={physical_ids}, logical_ids={logical_ids}"
            )

            resources = [faiss.StandardGpuResources() for _ in logical_ids]
            logger.debug(f"Initialized {len(resources)} GPU resource object(s)")

            if len(logical_ids) == 1:
                gpu_options = faiss.GpuClonerOptions()
                gpu_options.useFloat16 = True
                gpu_index = faiss.index_cpu_to_gpu(
                    resources[0],
                    logical_ids[0],
                    self._index,
                    gpu_options,
                )
            else:
                gpu_options = faiss.GpuMultipleClonerOptions()
                gpu_options.shard = True
                gpu_options.useFloat16 = True
                gpu_index = faiss.index_cpu_to_gpu_multiple_py(
                    resources,
                    self._index,
                    co=gpu_options,
                    gpus=logical_ids,
                )

            self._gpu_resources = resources
            self._index = gpu_index

            if len(logical_ids) == 1:
                logger.info(f"Index successfully moved to GPU {physical_ids[0]}")
            else:
                logger.info(f"Index successfully sharded across GPUs {physical_ids}")

        except Exception as exc:
            logger.warning(f"Failed to move index to GPU: {exc}")
            logger.info("Falling back to CPU index")
            self._use_gpu = False
            self._gpu_resources = None
            self._replace_search_executor(self._cpu_search_workers)

    def _resolve_gpu_device_ids(self) -> tuple[list[int], list[int]]:
        """Resolve configured physical IDs to the IDs visible to FAISS."""
        physical_ids = self._parse_gpu_device_ids()
        available_gpu_count = faiss.get_num_gpus()
        if available_gpu_count == 0:
            raise RuntimeError("FAISS did not detect any CUDA GPUs")

        visible_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_physical_ids: list[int] | None = None
        if visible_ids:
            try:
                visible_physical_ids = self._parse_gpu_device_ids(visible_ids)
            except ValueError:
                logger.debug("CUDA_VISIBLE_DEVICES uses non-numeric device identifiers")

        if visible_physical_ids == physical_ids and available_gpu_count == len(
            physical_ids
        ):
            logical_ids = list(range(len(physical_ids)))
        elif all(device_id < available_gpu_count for device_id in physical_ids):
            logical_ids = physical_ids
        else:
            raise RuntimeError(
                f"Configured GPUs {physical_ids} are not available; "
                f"FAISS detected {available_gpu_count} visible GPU(s)"
            )

        return physical_ids, logical_ids

    def _parse_gpu_device_ids(self, value: str | None = None) -> list[int]:
        """Parse and validate a comma-separated GPU ID list."""
        raw_value = self._gpu_device_ids if value is None else value
        tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("At least one GPU device ID must be configured")

        try:
            device_ids = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"Invalid GPU device IDs: {raw_value!r}") from exc

        if any(device_id < 0 for device_id in device_ids):
            raise ValueError("GPU device IDs must be non-negative")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError(f"Duplicate GPU device IDs: {raw_value!r}")

        return device_ids

    def _replace_search_executor(self, workers: int) -> None:
        """Replace search concurrency controls after backend changes."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._search_semaphore = asyncio.Semaphore(workers)
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="retserve-faiss",
        )

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
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._index.search,
                query_vectors,
                top_k,
            )

    def close(self) -> None:
        """Release search executor resources."""
        self._executor.shutdown(wait=False, cancel_futures=True)
