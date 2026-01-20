"""
Retrieval service implementation.

This module provides a high-performance retrieval service built on FastAPI,
using FAISS for vector similarity search and external embedding models.
The service supports GPU acceleration and concurrent request handling.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import faiss
import numpy as np
import orjson
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from tqdm import tqdm

from src.config_loader import config_loader
from src.decorators import log_execution, measure_time, retry
from src.logging import get_logger
from src.protocols import EmbeddingClient, VectorIndex
from src.settings import ServiceSettings
from src.types import (
    Document,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Embedding Client Implementation
# =============================================================================

class OpenAIEmbeddingClient:
    """
    Embedding client using OpenAI-compatible API.
    
    This client generates embeddings using an external API endpoint
    that follows the OpenAI embeddings API specification.
    
    Attributes:
        model_name: Name of the embedding model to use.
        batch_size: Maximum batch size for embedding requests.
    """
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "None",
        batch_size: int = 16,
        concurrency_limit: int = 32,
    ) -> None:
        """
        Initialize the embedding client.
        
        Args:
            base_url: Base URL of the embedding API.
            model_name: Name of the embedding model.
            api_key: API key for authentication.
            batch_size: Maximum batch size for embedding requests.
            concurrency_limit: Maximum concurrent embedding requests.
        """
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model_name
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        
        logger.info(
            f"Initialized OpenAIEmbeddingClient with model={model_name}, "
            f"batch_size={batch_size}, concurrency_limit={concurrency_limit}"
        )
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch_size
    
    @measure_time(threshold_ms=100)
    @retry(max_attempts=3, delay_seconds=1.0)
    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            Embedding vectors with shape (len(texts), dimension).
        """
        async with self._semaphore:
            if not texts:
                return np.array([], dtype=np.float32)
            
            total_texts = len(texts)
            all_embeddings: list[np.ndarray] = []
            
            logger.debug(f"Generating embeddings for {total_texts} texts")
            
            # Process in batches for memory efficiency
            for batch_start in range(0, total_texts, self._batch_size):
                batch_end = min(batch_start + self._batch_size, total_texts)
                batch_texts = texts[batch_start:batch_end]
                
                response = await self._client.embeddings.create(
                    model=self._model_name,
                    input=batch_texts,
                )
                
                # Extract embeddings from response
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in response.data
                ]
                all_embeddings.extend(batch_embeddings)
            
            return np.vstack(all_embeddings)


# =============================================================================
# Vector Index Implementation
# =============================================================================

class FAISSVectorIndex:
    """
    Vector index implementation using FAISS.
    
    This class wraps a FAISS index and provides async search operations
    with support for GPU acceleration.
    
    Attributes:
        dimension: Vector dimension of the index.
        size: Number of vectors in the index.
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
        
        # GPU requires strict concurrency control
        if use_gpu:
            search_concurrency_limit = 1
        
        self._search_semaphore = asyncio.Semaphore(search_concurrency_limit)
        self._index: faiss.Index | None = None
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
            
            # Move to GPU if configured
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
            
            # Initialize GPU resources
            gpu_resources = faiss.StandardGpuResources()
            logger.debug("GPU resources initialized")
            
            # Configure GPU options for performance
            gpu_options = faiss.GpuClonerOptions()
            gpu_options.useFloat16 = True  # Use FP16 for better performance
            
            # Move index to GPU (device 0 relative to CUDA_VISIBLE_DEVICES)
            self._index = faiss.index_cpu_to_gpu(
                gpu_resources, 0, self._index, gpu_options
            )
            
            logger.info("Index successfully moved to GPU")
            
        except Exception as exc:
            logger.warning(f"Failed to move index to GPU: {exc}")
            logger.info("Falling back to CPU index")
            self._use_gpu = False
            
            # Update semaphore for CPU mode (allow more concurrency)
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
        
        async with self._search_semaphore:
            logger.debug(
                f"Searching index: num_queries={query_vectors.shape[0]}, top_k={top_k}"
            )
            
            # Run search in thread pool to avoid blocking
            distances, indices = await asyncio.to_thread(
                self._index.search, query_vectors, top_k
            )
            
            return distances, indices


# =============================================================================
# Corpus Loader Implementation
# =============================================================================

class JSONLCorpusLoader:
    """
    Corpus loader for JSONL format files.
    
    This loader reads documents from a JSONL file and parses them
    into Document objects with progress feedback.
    """
    
    def __init__(self, corpus_path: str) -> None:
        """
        Initialize the corpus loader.
        
        Args:
            corpus_path: Path to the JSONL corpus file.
        """
        self._corpus_path = corpus_path
        
        logger.info(f"Initialized JSONLCorpusLoader with path={corpus_path}")
    
    @log_execution()
    def load(self) -> list[Document]:
        """
        Load documents from the corpus file.
        
        Returns:
            List of Document objects.
        """
        logger.info(f"Loading corpus from {self._corpus_path}")
        
        file_size = os.path.getsize(self._corpus_path)
        documents: list[Document] = []
        
        with open(self._corpus_path, "rb") as file:
            with tqdm(
                total=file_size,
                desc="Loading corpus",
                unit="B",
                unit_scale=True,
                ncols=100,
            ) as progress_bar:
                for raw_line in file:
                    progress_bar.update(len(raw_line))
                    
                    # Parse JSON line
                    line_data = orjson.loads(raw_line)
                    
                    # Extract document fields
                    document_id = str(line_data.get("id", ""))
                    contents = line_data.get("contents", "")
                    
                    if not isinstance(contents, str):
                        contents = str(contents)
                    
                    # Split content into title and text
                    parts = contents.split("\n", 1)
                    title = parts[0].strip()
                    text = parts[1] if len(parts) > 1 else ""
                    
                    documents.append(Document(
                        id=document_id,
                        title=title,
                        text=text,
                        contents=contents,
                    ))
        
        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents


# =============================================================================
# Service Container
# =============================================================================

class ServiceContainer:
    """
    Dependency injection container for the retrieval service.
    
    This container manages all service dependencies including the
    embedding client, vector index, and corpus data.
    
    Attributes:
        settings: Service configuration settings.
    """
    
    def __init__(
        self,
        settings: ServiceSettings,
        embedding_client: OpenAIEmbeddingClient,
        vector_index: FAISSVectorIndex,
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
        
        logger.info(
            f"ServiceContainer initialized with {len(corpus)} documents, "
            f"index_dim={vector_index.dimension}"
        )
    
    @property
    def settings(self) -> ServiceSettings:
        """Get service settings."""
        return self._settings
    
    @property
    def embedding_client(self) -> OpenAIEmbeddingClient:
        """Get the embedding client."""
        return self._embedding_client
    
    @property
    def vector_index(self) -> FAISSVectorIndex:
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
        # Apply max_top_k limit
        max_top_k = self._settings.server.max_top_k
        if top_k > max_top_k:
            logger.warning(
                f"Requested top_k={top_k} exceeds max_top_k={max_top_k}, "
                f"using max_top_k instead"
            )
            top_k = max_top_k
        
        # Generate query embeddings
        query_embeddings = await self._embedding_client.embed(queries)
        
        # Validate embedding dimension
        if query_embeddings.shape[1] != self._vector_index.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: got {query_embeddings.shape[1]}, "
                f"expected {self._vector_index.dimension}"
            )
        
        # Perform vector search
        distances, indices = await self._vector_index.search(query_embeddings, top_k)
        
        # Build response
        return self._build_search_response(indices, distances)
    
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
        
        corpus_size = len(self._corpus)
        
        for query_indices, query_distances in zip(indices, distances):
            current_contents: list[dict[str, Any]] = []
            current_scores: list[float] = []
            
            for document_index, score in zip(query_indices, query_distances):
                # Skip invalid indices
                if document_index == -1:
                    continue
                
                if 0 <= document_index < corpus_size:
                    document = self._corpus[document_index]
                    current_contents.append(document.model_dump())
                    current_scores.append(float(score))
            
            contents_batch.append(current_contents)
            scores_batch.append(current_scores)
        
        return SearchResponse(contents=contents_batch, scores=scores_batch)


# =============================================================================
# Global Service Container
# =============================================================================

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
    
    logger.info("Initializing retrieval service...")
    
    # Set CUDA visible devices if using GPU
    if settings.index.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.index.gpu_device_ids
        logger.info(f"Set CUDA_VISIBLE_DEVICES={settings.index.gpu_device_ids}")
    
    # Create embedding client
    embedding_client = OpenAIEmbeddingClient(
        base_url=settings.embedding.base_url,
        model_name=settings.embedding.model_name,
        api_key=settings.embedding.api_key,
    )
    
    # Create and load vector index
    vector_index = FAISSVectorIndex(
        index_path=settings.index.path,
        use_gpu=settings.index.use_gpu,
        gpu_device_ids=settings.index.gpu_device_ids,
    )
    vector_index.load()
    
    # Load corpus
    corpus_loader = JSONLCorpusLoader(settings.data.corpus_path)
    corpus = corpus_loader.load()
    
    # Create service container
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


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_application(settings: ServiceSettings) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Service configuration settings.
        
    Returns:
        Configured FastAPI application.
    """
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        initialize_service(settings)
        yield
        # Shutdown (cleanup if needed)
        logger.info("Shutting down retrieval service")
    
    app = FastAPI(
        title="FAISS Retrieval Service",
        version="2.0.0",
        description=(
            "High-performance vector similarity search service powered by "
            "FAISS and embedding models. Supports GPU acceleration and "
            "concurrent request handling."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Mount static files
    static_directory = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_directory), name="static")
    
    # ==========================================================================
    # Route Definitions
    # ==========================================================================
    
    @app.get("/", response_class=RedirectResponse)
    async def redirect_to_ui() -> RedirectResponse:
        """Redirect root path to the web UI."""
        return RedirectResponse(url="/static/index.html")
    
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        description="Check service health and get system information.",
    )
    def check_health(
        container: ServiceContainer = Depends(get_service_container),
    ) -> HealthResponse:
        """
        Health check endpoint.
        
        Returns service status and configuration information.
        """
        return HealthResponse(
            status="ok",
            index_dimension=container.vector_index.dimension,
            corpus_size=container.corpus_size,
            embedding_url=container.settings.embedding.base_url,
            embedding_model=container.settings.embedding.model_name,
            gpu_enabled=container.settings.index.use_gpu,
        )
    
    @app.post(
        "/search",
        response_model=SearchResponse,
        summary="Vector Search",
        description="Search for similar documents using vector similarity.",
    )
    async def perform_search(
        request: SearchRequest,
        container: ServiceContainer = Depends(get_service_container),
    ) -> SearchResponse:
        """
        Perform vector similarity search.
        
        Args:
            request: Search request with queries and top_k.
            
        Returns:
            Search results with matched documents and scores.
        """
        try:
            return await container.search(
                queries=request.queries,
                top_k=request.top_k,
            )
        except ValueError as exc:
            logger.error(f"Search validation error: {exc}")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(f"Search error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return app


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
        description="FAISS Retrieval Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="serve",
        help="Configuration file name (without .yaml extension)",
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the retrieval service.
    """
    import uvicorn
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    settings = config_loader.load_service_settings(args.config)
    
    # Create application
    app = create_application(settings)
    
    # Run server with optimized settings
    uvicorn.run(
        app=app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
        reload=False,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=30,
        backlog=1000,
    )


if __name__ == "__main__":
    main()
