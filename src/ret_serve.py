"""FastAPI application and CLI entry point for the retrieval service."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from importlib.util import find_spec

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse

from src.config_loader import config_loader
from src.corpus import JSONLCorpusLoader
from src.errors import (
    EmbeddingDimensionError,
    EmbeddingUpstreamError,
    IndexNotReadyError,
    RetrievalExecutionError,
)
from src.logging import get_logger
from src.runtime import RetServeRuntime
from src.service_container import (
    ServiceContainer,
    close_service,
    get_runtime,
    get_service_container,
    initialize_service,
)
from src.settings import ServiceSettings
from src.types import HealthResponse, SearchRequest, SearchResponse
from src.vector_index import FAISSVectorIndex

logger = get_logger(__name__)

__all__ = [
    "FAISSVectorIndex",
    "JSONLCorpusLoader",
    "ServiceContainer",
    "RetServeRuntime",
    "close_service",
    "create_application",
    "get_service_container",
    "get_runtime",
    "initialize_service",
    "main",
    "parse_arguments",
]


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
        initialize_service(settings)
        app.state.runtime = get_runtime()
        try:
            yield
        finally:
            await close_service()
            app.state.runtime = None
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

    @app.get("/", response_class=RedirectResponse)
    async def redirect_to_docs() -> RedirectResponse:
        """Redirect root path to the interactive API docs."""
        return RedirectResponse(url="/docs")

    def get_app_runtime(request: Request) -> RetServeRuntime:
        """Return the runtime stored on application state."""
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="service not ready")
        return runtime

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        description="Check service health and get system information.",
    )
    def check_health(
        runtime: RetServeRuntime = Depends(get_app_runtime),
    ) -> HealthResponse:
        """
        Health check endpoint.

        Returns service status and configuration information.
        """
        return HealthResponse(
            status="ok",
            index_dimension=runtime.vector_index.dimension,
            corpus_size=runtime.corpus_size,
            embedding_url=runtime.settings.embedding.base_url,
            embedding_model=runtime.settings.embedding.model_name,
            gpu_enabled=runtime.settings.index.use_gpu,
        )

    @app.get("/livez", summary="Liveness Check")
    def check_liveness() -> dict[str, str]:
        """Return process liveness."""
        return {"status": "ok"}

    @app.get("/readyz", summary="Readiness Check")
    def check_readiness(
        runtime: RetServeRuntime = Depends(get_app_runtime),
    ) -> dict[str, object]:
        """Return readiness for search traffic."""
        if not runtime.ready:
            raise HTTPException(status_code=503, detail="service not ready")
        return {
            "status": "ready",
            "index_dim": runtime.vector_index.dimension,
            "index_size": runtime.vector_index.size,
            "corpus_size": runtime.corpus_size,
        }

    @app.get("/metrics", response_class=PlainTextResponse, summary="Metrics")
    def get_metrics(
        runtime: RetServeRuntime = Depends(get_app_runtime),
    ) -> PlainTextResponse:
        """Return Prometheus text metrics."""
        return PlainTextResponse(
            runtime.render_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
        summary="Vector Search",
        description="Search for similar documents using vector similarity.",
    )
    async def perform_search(
        request: SearchRequest,
        runtime: RetServeRuntime = Depends(get_app_runtime),
    ) -> SearchResponse:
        """
        Perform vector similarity search.

        Args:
            request: Search request with queries and top_k.

        Returns:
            Search results with matched documents and scores.
        """
        return await _search_or_raise(request, runtime)

    return app


async def _search_or_raise(
    request: SearchRequest,
    runtime: RetServeRuntime,
) -> SearchResponse:
    """Run search and map typed errors to stable HTTP responses."""
    try:
        return await runtime.search(
            queries=request.queries,
            top_k=request.top_k,
        )
    except EmbeddingDimensionError as exc:
        logger.error(f"Search validation error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmbeddingUpstreamError as exc:
        logger.error("Embedding provider error during search")
        raise HTTPException(status_code=502, detail="embedding provider error") from exc
    except IndexNotReadyError as exc:
        logger.error("Index not ready during search")
        raise HTTPException(status_code=503, detail="index not ready") from exc
    except RetrievalExecutionError as exc:
        logger.error("Retrieval execution error")
        raise HTTPException(
            status_code=500, detail="retrieval execution error"
        ) from exc


def get_uvicorn_runtime_options() -> dict[str, str]:
    """Choose optional uvicorn runtimes only when they are installed."""
    loop = "uvloop" if find_spec("uvloop") is not None else "asyncio"
    http = "httptools" if find_spec("httptools") is not None else "h11"
    return {"loop": loop, "http": http}


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

    args = parse_arguments()
    settings = config_loader.load_service_settings(args.config)
    app = create_application(settings)
    runtime_options = get_uvicorn_runtime_options()

    uvicorn.run(
        app=app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
        reload=False,
        loop=runtime_options["loop"],
        http=runtime_options["http"],
        timeout_keep_alive=30,
        backlog=1000,
    )


if __name__ == "__main__":
    main()
