"""Retrieval engine for online search requests."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

import numpy as np

from src.document_store import DocumentStore
from src.errors import (
    EmbeddingDimensionError,
    EmbeddingUpstreamError,
    IndexNotReadyError,
    RetrievalExecutionError,
)
from src.logging import get_logger
from src.metrics import MetricsRegistry
from src.protocols import EmbeddingClient, VectorIndex
from src.types import SearchResponse

logger = get_logger(__name__)


class QueryEmbeddingCache:
    """Bounded async LRU cache for query embedding rows."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(capacity, 0)
        self._rows: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Return whether process-wide query caching is enabled."""
        return self._capacity > 0

    @property
    def size(self) -> int:
        """Return current cached row count."""
        return len(self._rows)

    @property
    def capacity(self) -> int:
        """Return maximum cached row count."""
        return self._capacity

    async def get_many(self, queries: list[str]) -> dict[str, np.ndarray]:
        """Return cached rows for unique query strings."""
        if not self.enabled:
            return {}

        hits: dict[str, np.ndarray] = {}
        async with self._lock:
            for query in queries:
                row = self._rows.get(query)
                if row is not None:
                    self._rows.move_to_end(query)
                    hits[query] = row
        return hits

    async def set_many(self, rows_by_query: dict[str, np.ndarray]) -> None:
        """Store query rows and evict least-recently-used rows."""
        if not self.enabled:
            return

        async with self._lock:
            for query, row in rows_by_query.items():
                self._rows[query] = np.ascontiguousarray(row, dtype=np.float32).copy()
                self._rows.move_to_end(query)

            while len(self._rows) > self._capacity:
                self._rows.popitem(last=False)


class RetrievalEngine:
    """Coordinates embedding, vector search, response assembly, and metrics."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_index: VectorIndex,
        document_store: DocumentStore,
        metrics: MetricsRegistry,
        query_cache_size: int = 0,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        self._document_store = document_store
        self._metrics = metrics
        self._query_cache = QueryEmbeddingCache(query_cache_size)

    @property
    def query_cache_size(self) -> int:
        """Return the active query cache capacity."""
        return self._query_cache.capacity

    async def search(self, queries: list[str], top_k: int) -> SearchResponse:
        """Run retrieval for a batch of query strings."""
        start = time.perf_counter()
        self._metrics.increment(
            "retserve_requests_total", labels={"endpoint": "search"}
        )
        try:
            effective_top_k = self._resolve_top_k(top_k)
            query_embeddings = await self._get_query_embeddings(queries)
            self._validate_query_embeddings(query_embeddings)

            if effective_top_k == 0:
                return SearchResponse(
                    contents=[[] for _ in queries],
                    scores=[[] for _ in queries],
                )

            faiss_start = time.perf_counter()
            distances, indices = await self._vector_index.search(
                query_embeddings,
                effective_top_k,
            )
            self._metrics.observe(
                "retserve_faiss_search_seconds",
                time.perf_counter() - faiss_start,
            )
            return self._build_search_response(indices, distances)
        except (EmbeddingDimensionError, EmbeddingUpstreamError, IndexNotReadyError):
            self._metrics.increment("retserve_errors_total", labels={"type": "known"})
            raise
        except Exception as exc:
            self._metrics.increment(
                "retserve_errors_total",
                labels={"type": "retrieval_execution"},
            )
            raise RetrievalExecutionError("Retrieval execution failed") from exc
        finally:
            self._metrics.observe(
                "retserve_search_seconds",
                time.perf_counter() - start,
            )

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
        """Return query vectors with request-scope dedupe and optional LRU cache."""
        if not queries:
            return np.empty((0, 0), dtype=np.float32)

        unique_queries = list(dict.fromkeys(queries))
        rows_by_query = await self._query_cache.get_many(unique_queries)
        cache_hits = len(rows_by_query)
        missing_queries = [
            query for query in unique_queries if query not in rows_by_query
        ]

        if cache_hits:
            self._metrics.increment("retserve_query_cache_hits_total", cache_hits)
        if missing_queries:
            self._metrics.increment(
                "retserve_query_cache_misses_total",
                len(missing_queries),
            )
            missing_rows = await self._embed_queries(missing_queries)
            if missing_rows.shape[0] != len(missing_queries):
                raise EmbeddingUpstreamError(
                    "Embedding provider returned an unexpected vector count"
                )

            new_rows = {
                query: np.ascontiguousarray(row, dtype=np.float32)
                for query, row in zip(missing_queries, missing_rows)
            }
            rows_by_query.update(new_rows)
            await self._query_cache.set_many(new_rows)

        return np.ascontiguousarray(
            np.vstack([rows_by_query[query] for query in queries]),
            dtype=np.float32,
        )

    async def _embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed queries and normalize the result for FAISS search."""
        try:
            embed_start = time.perf_counter()
            query_embeddings = await self._embedding_client.embed(queries)
            self._metrics.observe(
                "retserve_embedding_seconds",
                time.perf_counter() - embed_start,
            )
        except Exception as exc:
            raise EmbeddingUpstreamError("Embedding provider request failed") from exc

        return np.ascontiguousarray(query_embeddings, dtype=np.float32)

    def _validate_query_embeddings(self, query_embeddings: np.ndarray) -> None:
        """Validate embedding matrix shape and dimension."""
        if query_embeddings.ndim != 2:
            raise EmbeddingUpstreamError(
                "Embedding provider returned a non-matrix result"
            )

        if self._vector_index.size <= 0:
            raise IndexNotReadyError("Vector index is empty or not ready")

        if query_embeddings.shape[1] != self._vector_index.dimension:
            raise EmbeddingDimensionError(
                "Embedding dimension mismatch: "
                f"got {query_embeddings.shape[1]}, expected {self._vector_index.dimension}"
            )

    def _build_search_response(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
    ) -> SearchResponse:
        """Build the public search response from vector search outputs."""
        contents_batch: list[list[dict[str, object]]] = []
        scores_batch: list[list[float]] = []

        for query_indices, query_distances in zip(indices, distances):
            current_contents: list[dict[str, object]] = []
            current_scores: list[float] = []

            for document_index, score in zip(query_indices, query_distances):
                document_index = int(document_index)
                if document_index == -1:
                    continue

                payload = self._document_store.payload_copy(document_index)
                if payload is not None:
                    current_contents.append(payload)
                    current_scores.append(float(score))

            contents_batch.append(current_contents)
            scores_batch.append(current_scores)

        return SearchResponse(contents=contents_batch, scores=scores_batch)
