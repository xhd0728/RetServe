"""Runtime lifecycle for the online retrieval service."""

from __future__ import annotations

import inspect
import os

from src.corpus import JSONLCorpusLoader
from src.document_store import DocumentStore
from src.embedding_client import OpenAIEmbeddingClient
from src.logging import get_logger
from src.metrics import MetricsRegistry
from src.retrieval import RetrievalEngine
from src.settings import ServiceSettings
from src.types import Document, SearchResponse
from src.vector_index import FAISSVectorIndex

logger = get_logger(__name__)


class RetServeRuntime:
    """Owns retrieval resources and exposes service operations."""

    def __init__(
        self,
        settings: ServiceSettings,
        embedding_client: OpenAIEmbeddingClient,
        vector_index: FAISSVectorIndex,
        document_store: DocumentStore,
        metrics: MetricsRegistry,
        engine: RetrievalEngine,
    ) -> None:
        self._settings = settings
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        self._document_store = document_store
        self._metrics = metrics
        self._engine = engine
        self._ready = True
        self._sync_metrics_gauges()

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> "RetServeRuntime":
        """Create and fully initialize a runtime from service settings."""
        logger.info("Initializing retrieval runtime...")
        metrics = MetricsRegistry(enabled=settings.metrics.enabled)
        metrics.set_gauge("retserve_ready", 0)

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
            search_concurrency_limit=settings.index.search_workers,
            search_workers=settings.index.search_workers,
            omp_threads=settings.index.omp_threads,
        )
        vector_index.load()

        document_store = DocumentStore(
            JSONLCorpusLoader(settings.data.corpus_path).load()
        )
        engine = RetrievalEngine(
            embedding_client=embedding_client,
            vector_index=vector_index,
            document_store=document_store,
            metrics=metrics,
            query_cache_size=settings.embedding.effective_query_cache_size,
        )

        runtime = cls(
            settings=settings,
            embedding_client=embedding_client,
            vector_index=vector_index,
            document_store=document_store,
            metrics=metrics,
            engine=engine,
        )
        logger.info(
            f"Runtime initialized: index_path={settings.index.path}, "
            f"corpus_size={document_store.size}, gpu_enabled={settings.index.use_gpu}"
        )
        return runtime

    @property
    def settings(self) -> ServiceSettings:
        """Return service settings."""
        return self._settings

    @property
    def embedding_client(self) -> OpenAIEmbeddingClient:
        """Return the embedding client."""
        return self._embedding_client

    @property
    def vector_index(self) -> FAISSVectorIndex:
        """Return the vector index backend."""
        return self._vector_index

    @property
    def document_store(self) -> DocumentStore:
        """Return the document store."""
        return self._document_store

    @property
    def metrics(self) -> MetricsRegistry:
        """Return the metrics registry."""
        return self._metrics

    @property
    def corpus(self) -> list[Document]:
        """Return loaded corpus documents."""
        return self._document_store.documents

    @property
    def corpus_size(self) -> int:
        """Return loaded corpus size."""
        return self._document_store.size

    @property
    def ready(self) -> bool:
        """Return whether the runtime is ready for search traffic."""
        return self._ready

    async def search(self, queries: list[str], top_k: int) -> SearchResponse:
        """Run a retrieval search."""
        return await self._engine.search(queries=queries, top_k=top_k)

    async def close(self) -> None:
        """Close runtime-owned resources."""
        self._ready = False
        self._metrics.set_gauge("retserve_ready", 0)

        close_client = getattr(self._embedding_client, "close", None)
        if close_client is not None:
            close_result = close_client()
            if inspect.isawaitable(close_result):
                await close_result

        close_index = getattr(self._vector_index, "close", None)
        if close_index is not None:
            close_index()

    def render_metrics(self) -> str:
        """Render metrics endpoint output."""
        self._sync_metrics_gauges()
        return self._metrics.render_prometheus()

    def _sync_metrics_gauges(self) -> None:
        """Refresh gauges derived from current runtime state."""
        self._metrics.set_gauge("retserve_ready", 1 if self._ready else 0)
        self._metrics.set_gauge("retserve_index_size", float(self._vector_index.size))
        self._metrics.set_gauge(
            "retserve_index_dimension",
            float(self._vector_index.dimension),
        )
        self._metrics.set_gauge(
            "retserve_corpus_size", float(self._document_store.size)
        )
