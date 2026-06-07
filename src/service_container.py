"""
Service container and dependency lifecycle for online retrieval.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.decorators import measure_time
from src.logging import get_logger
from src.protocols import EmbeddingClient, VectorIndex
from src.settings import ServiceSettings
from src.types import Document, SearchResponse

logger = get_logger(__name__)


class ServiceContainer:
    """
    Dependency injection container for the retrieval service.

    This container manages all service dependencies including the embedding
    client, vector index, and corpus data.
    """

    def __init__(
        self,
        settings: ServiceSettings,
        embedding_client: EmbeddingClient,
        vector_index: VectorIndex,
        corpus: list[Document],
    ) -> None:
        """
        Initialize the service container.

        Args:
            settings: Service configuration settings.
            embedding_client: Client for generating embeddings.
            vector_index: Vector index for similarity search.
            corpus: List of corpus documents.
        """
        self._settings = settings
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        self._corpus = corpus
        self._corpus_payloads: tuple[Mapping[str, Any], ...] = tuple(
            MappingProxyType(document.model_dump()) for document in corpus
        )
        self._query_cache_size = settings.embedding.effective_query_cache_size
        self._query_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = asyncio.Lock()

        logger.info(
            f"ServiceContainer initialized with {len(corpus)} documents, "
            f"index_dim={vector_index.dimension}, "
            f"query_cache_size={self._query_cache_size}"
        )

    @property
    def settings(self) -> ServiceSettings:
        """Get service settings."""
        return self._settings

    @property
    def embedding_client(self) -> EmbeddingClient:
        """Get the embedding client."""
        return self._embedding_client

    @property
    def vector_index(self) -> VectorIndex:
        """Get the vector index."""
        return self._vector_index

    @property
    def corpus(self) -> list[Document]:
        """Get the corpus documents."""
        return self._corpus

    @property
    def corpus_size(self) -> int:
        """Get the number of documents in the corpus."""
        return len(self._corpus)

    @measure_time(threshold_ms=100)
    async def search(
        self,
        queries: list[str],
        top_k: int,
    ) -> SearchResponse:
        """
        Perform similarity search for the given queries.

        Args:
            queries: List of query strings.
            top_k: Number of top results per query.

        Returns:
            SearchResponse with results for all queries.
        """
        effective_top_k = self._resolve_top_k(top_k)

        query_embeddings = await self._get_query_embeddings(queries)

        if query_embeddings.ndim != 2:
            raise ValueError(
                "Embedding API returned a non-matrix result: "
                f"shape={query_embeddings.shape}"
            )

        if query_embeddings.shape[1] != self._vector_index.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: got {query_embeddings.shape[1]}, "
                f"expected {self._vector_index.dimension}"
            )

        if effective_top_k == 0:
            return SearchResponse(
                contents=[[] for _ in queries],
                scores=[[] for _ in queries],
            )

        distances, indices = await self._vector_index.search(
            query_embeddings,
            effective_top_k,
        )
        return self._build_search_response(indices, distances)

    def _resolve_top_k(self, requested_top_k: int) -> int:
        """Limit top_k to the number of indexed vectors."""
        index_size = max(self._vector_index.size, 0)
        if requested_top_k > index_size:
            logger.warning(
                f"Requested top_k={requested_top_k} exceeds index_size={index_size}, "
                "using index_size instead"
            )
            return index_size

        return requested_top_k

    async def _get_query_embeddings(self, queries: list[str]) -> np.ndarray:
        """Get query embeddings with an optional per-process LRU cache."""
        if not queries:
            return np.empty((0, 0), dtype=np.float32)

        if self._query_cache_size <= 0:
            return await self._embed_queries(queries)

        rows_by_query: dict[str, np.ndarray] = {}
        missing_queries: list[str] = []
        missing_seen: set[str] = set()

        async with self._query_cache_lock:
            for query in queries:
                cached_row = self._query_embedding_cache.get(query)
                if cached_row is not None:
                    self._query_embedding_cache.move_to_end(query)
                    rows_by_query[query] = cached_row
                elif query not in missing_seen:
                    missing_seen.add(query)
                    missing_queries.append(query)

        if missing_queries:
            missing_embeddings = await self._embed_queries(missing_queries)
            if missing_embeddings.shape[0] != len(missing_queries):
                raise RuntimeError(
                    "Embedding client returned an unexpected number of query vectors: "
                    f"got {missing_embeddings.shape[0]}, expected {len(missing_queries)}"
                )

            for query, row in zip(missing_queries, missing_embeddings):
                rows_by_query[query] = np.ascontiguousarray(
                    row, dtype=np.float32
                ).copy()

            async with self._query_cache_lock:
                for query in missing_queries:
                    self._query_embedding_cache[query] = rows_by_query[query]
                    self._query_embedding_cache.move_to_end(query)

                while len(self._query_embedding_cache) > self._query_cache_size:
                    self._query_embedding_cache.popitem(last=False)

        return np.ascontiguousarray(
            np.vstack([rows_by_query[query] for query in queries]),
            dtype=np.float32,
        )

    async def _embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed queries and normalize the result for FAISS search."""
        query_embeddings = await self._embedding_client.embed(queries)
        return np.ascontiguousarray(query_embeddings, dtype=np.float32)

    def _build_search_response(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
    ) -> SearchResponse:
        """
        Build search response from search results.

        Args:
            indices: Document indices from search.
            distances: Similarity scores from search.

        Returns:
            Formatted SearchResponse.
        """
        contents_batch: list[list[dict[str, Any]]] = []
        scores_batch: list[list[float]] = []
        corpus_size = len(self._corpus_payloads)

        for query_indices, query_distances in zip(indices, distances):
            current_contents: list[dict[str, Any]] = []
            current_scores: list[float] = []

            for document_index, score in zip(query_indices, query_distances):
                document_index = int(document_index)
                if document_index == -1:
                    continue

                if 0 <= document_index < corpus_size:
                    current_contents.append(dict(self._corpus_payloads[document_index]))
                    current_scores.append(float(score))

            contents_batch.append(current_contents)
            scores_batch.append(current_scores)

        return SearchResponse(contents=contents_batch, scores=scores_batch)


_service_container: ServiceContainer | None = None


def get_service_container() -> ServiceContainer:
    """
    Get the global service container.

    Returns:
        The initialized ServiceContainer.

    Raises:
        RuntimeError: If the service container is not initialized.
    """
    if _service_container is None:
        raise RuntimeError("Service container not initialized")
    return _service_container


def initialize_service(settings: ServiceSettings) -> None:
    """
    Initialize the global service container.

    Args:
        settings: Service configuration settings.
    """
    global _service_container

    from src.corpus import JSONLCorpusLoader
    from src.embedding_client import OpenAIEmbeddingClient
    from src.vector_index import FAISSVectorIndex

    logger.info("Initializing retrieval service...")

    if settings.index.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.index.gpu_device_ids
        logger.info(f"Set CUDA_VISIBLE_DEVICES={settings.index.gpu_device_ids}")

    embedding_client = OpenAIEmbeddingClient(
        base_url=settings.embedding.base_url,
        model_name=settings.embedding.model_name,
        api_key=settings.embedding.resolved_api_key,
        batch_size=settings.embedding.batch_size,
        concurrency_limit=settings.embedding.concurrency_limit,
        request_timeout=settings.embedding.request_timeout,
        max_retries=settings.embedding.max_retries,
        normalize=settings.embedding.normalize,
        dimensions=settings.embedding.dimensions,
    )

    vector_index = FAISSVectorIndex(
        index_path=settings.index.path,
        use_gpu=settings.index.use_gpu,
        gpu_device_ids=settings.index.gpu_device_ids,
        search_concurrency_limit=settings.index.search_concurrency_limit,
    )
    vector_index.load()

    corpus_loader = JSONLCorpusLoader(settings.data.corpus_path)
    corpus = corpus_loader.load()

    _service_container = ServiceContainer(
        settings=settings,
        embedding_client=embedding_client,
        vector_index=vector_index,
        corpus=corpus,
    )

    logger.info(
        f"Service initialized: "
        f"index_path={settings.index.path}, "
        f"corpus_size={len(corpus)}, "
        f"gpu_enabled={settings.index.use_gpu}"
    )


async def close_service() -> None:
    """Close resources owned by the global service container."""
    global _service_container

    if _service_container is None:
        return

    close_client = getattr(_service_container.embedding_client, "close", None)
    if close_client is not None:
        close_result = close_client()
        if inspect.isawaitable(close_result):
            await close_result

    _service_container = None
