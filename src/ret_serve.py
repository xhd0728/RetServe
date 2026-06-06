"""FastAPI application and CLI entry point for the retrieval service."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.corpus import JSONLCorpusLoader
from src.config_loader import config_loader
from src.logging import get_logger
from src.service_container import (
    ServiceContainer,
    close_service,
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
    "close_service",
    "create_application",
    "get_service_container",
    "initialize_service",
    "main",
    "parse_arguments",
]


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
        initialize_service(settings)
        try:
            yield
        finally:
            await close_service()
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
    
    static_directory = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=str(static_directory)), name="static")
    
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
