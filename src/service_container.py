"""Compatibility container and global runtime helpers."""

from __future__ import annotations

from src.document_store import DocumentStore
from src.logging import get_logger
from src.metrics import MetricsRegistry
from src.protocols import EmbeddingClient, VectorIndex
from src.retrieval import RetrievalEngine
from src.runtime import RetServeRuntime
from src.settings import ServiceSettings
from src.types import Document, SearchResponse

logger = get_logger(__name__)


class ServiceContainer:
    """Compatibility wrapper around the retrieval runtime."""

    def __init__(
        self,
        settings: ServiceSettings,
        embedding_client: EmbeddingClient,
        vector_index: VectorIndex,
        corpus: list[Document],
    ) -> None:
        self._runtime: RetServeRuntime | None = None
        self._settings = settings
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        self._document_store = DocumentStore(corpus)
        self._metrics = MetricsRegistry(enabled=settings.metrics.enabled)
        self._engine = RetrievalEngine(
            embedding_client=embedding_client,
            vector_index=vector_index,
            document_store=self._document_store,
            metrics=self._metrics,
            query_cache_size=settings.embedding.effective_query_cache_size,
        )

    @classmethod
    def from_runtime(cls, runtime: RetServeRuntime) -> "ServiceContainer":
        """Create a compatibility wrapper for an initialized runtime."""
        container = cls.__new__(cls)
        container._runtime = runtime
        return container

    @property
    def settings(self) -> ServiceSettings:
        """Get service settings."""
        if self._runtime is not None:
            return self._runtime.settings
        return self._settings

    @property
    def embedding_client(self) -> EmbeddingClient:
        """Get the embedding client."""
        if self._runtime is not None:
            return self._runtime.embedding_client
        return self._embedding_client

    @property
    def vector_index(self) -> VectorIndex:
        """Get the vector index."""
        if self._runtime is not None:
            return self._runtime.vector_index
        return self._vector_index

    @property
    def corpus(self) -> list[Document]:
        """Get the corpus documents."""
        if self._runtime is not None:
            return self._runtime.corpus
        return self._document_store.documents

    @property
    def corpus_size(self) -> int:
        """Get the number of documents in the corpus."""
        if self._runtime is not None:
            return self._runtime.corpus_size
        return self._document_store.size

    async def search(
        self,
        queries: list[str],
        top_k: int,
    ) -> SearchResponse:
        """Perform similarity search for the given queries."""
        if self._runtime is not None:
            return await self._runtime.search(queries=queries, top_k=top_k)
        return await self._engine.search(queries=queries, top_k=top_k)


_runtime: RetServeRuntime | None = None
_service_container: ServiceContainer | None = None


def get_runtime() -> RetServeRuntime:
    """Get the initialized global runtime."""
    if _runtime is None:
        raise RuntimeError("RetServe runtime not initialized")
    return _runtime


def get_service_container() -> ServiceContainer:
    """Get the initialized global compatibility container."""
    if _service_container is None:
        raise RuntimeError("Service container not initialized")
    return _service_container


def initialize_service(settings: ServiceSettings) -> None:
    """Initialize the global retrieval runtime and compatibility container."""
    global _runtime, _service_container

    _runtime = RetServeRuntime.from_settings(settings)
    _service_container = ServiceContainer.from_runtime(_runtime)


async def close_service() -> None:
    """Close resources owned by the global runtime."""
    global _runtime, _service_container

    if _runtime is not None:
        await _runtime.close()

    _runtime = None
    _service_container = None
    logger.info("Retrieval runtime closed")
