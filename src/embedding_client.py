"""
OpenAI-compatible embedding client utilities.

This module is shared by the offline corpus encoder and the online retrieval
service. It targets OpenAI-compatible servers such as vLLM's /v1/embeddings
endpoint while keeping batching, concurrency, ordering, and normalization in
one place.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from src.decorators import measure_time
from src.logging import get_logger

logger = get_logger(__name__)


def normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize embedding rows to unit length in-place when possible.

    Args:
        embeddings: Embedding matrix with shape (n, dim).
        eps: Lower bound for norms to avoid division by zero.

    Returns:
        Float32 embedding matrix with L2-normalized rows.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.size == 0:
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, np.maximum(norms, eps), out=embeddings)
    return embeddings


class OpenAIEmbeddingClient:
    """
    Async embedding client for OpenAI-compatible APIs.

    The client splits input texts into API-sized batches, sends those batches
    concurrently with a shared semaphore, preserves input order, and optionally
    normalizes vectors for cosine/IP FAISS indexes.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "None",
        batch_size: int = 16,
        concurrency_limit: int = 32,
        request_timeout: float = 60.0,
        max_retries: int = 2,
        normalize: bool = False,
        dimensions: int | None = None,
    ) -> None:
        """
        Initialize the embedding client.

        Args:
            base_url: Base URL of the embedding API, usually ending in /v1.
            model_name: Served embedding model name.
            api_key: API key; vLLM accepts any non-empty value by default.
            batch_size: Maximum texts per embeddings API request.
            concurrency_limit: Maximum in-flight embeddings API requests.
            request_timeout: Per-request timeout in seconds.
            max_retries: Retries delegated to the OpenAI client.
            normalize: Whether to L2-normalize returned vectors.
            dimensions: Optional dimensions parameter for compatible APIs.
        """
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        self._model_name = model_name
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._normalize = normalize
        self._dimensions = dimensions

        logger.info(
            f"Initialized OpenAIEmbeddingClient: model={model_name}, "
            f"batch_size={batch_size}, concurrency_limit={concurrency_limit}, "
            f"timeout={request_timeout}s, max_retries={max_retries}, "
            f"normalize={normalize}"
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def batch_size(self) -> int:
        """Get the API request batch size."""
        return self._batch_size

    async def __aenter__(self) -> "OpenAIEmbeddingClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close underlying HTTP resources."""
        close_result = self._client.close()
        if inspect.isawaitable(close_result):
            await close_result

    async def _embed_batch(self, batch_texts: list[str]) -> np.ndarray:
        """
        Embed one API-sized batch.

        Args:
            batch_texts: Texts for a single embeddings API request.

        Returns:
            Embedding matrix for the batch.
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "input": batch_texts,
        }
        if self._dimensions is not None:
            request_kwargs["dimensions"] = self._dimensions

        async with self._semaphore:
            response = await self._client.embeddings.create(**request_kwargs)

        response_data = list(response.data)
        if response_data and hasattr(response_data[0], "index"):
            response_data.sort(key=lambda item: item.index)

        vectors = np.asarray(
            [item.embedding for item in response_data],
            dtype=np.float32,
        )

        if vectors.shape[0] != len(batch_texts):
            raise RuntimeError(
                "Embedding API returned an unexpected number of vectors: "
                f"got {vectors.shape[0]}, expected {len(batch_texts)}"
            )

        if self._normalize:
            vectors = normalize_embeddings(vectors)

        return vectors

    @measure_time(threshold_ms=100)
    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Embedding vectors with shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches = [
            texts[start:start + self._batch_size]
            for start in range(0, len(texts), self._batch_size)
        ]
        batch_arrays = await asyncio.gather(
            *(self._embed_batch(batch) for batch in batches)
        )

        return np.vstack(batch_arrays)
