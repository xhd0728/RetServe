import os
import faiss
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import orjson
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
import argparse
from src.config_loader import config_loader
from src.logging import logger


# -------------------- schemas --------------------
class SearchRequest(BaseModel):
    """Search request model"""
    queries: List[str] = Field(..., description="Query list")
    topk: int = Field(5, gt=0, description="Top-k results per query")

    @validator("queries")
    def queries_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("queries cannot be empty")
        return v


class SearchResponse(BaseModel):
    """Search response model"""
    contents: List[List[Dict[str, Any]]]
    scores: List[List[float]]


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    index_dim: int
    corpus_size: int
    emb_url: str
    emb_model: str
    use_gpu: bool


# -------------------- dependency management --------------------
class ServiceDependencies:
    """Service dependencies manager"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize service dependencies
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._index: Optional[faiss.Index] = None
        self._corpus: List[Dict[str, Any]] = []
        self._client: Optional[AsyncOpenAI] = None
        self._index_dim: int = -1
        
        # Semaphores for concurrency control
        # For GPU, we use stricter limits (1) to ensure thread-safe access
        if self.config['index']['use_gpu']:
            self.emb_semaphore = asyncio.Semaphore(32)  # Moderate limit for embeddings
            self.search_semaphore = asyncio.Semaphore(1)   # Strict limit for GPU search (only 1 at a time)
            logger.info("Using GPU-safe semaphore settings")
        else:
            self.emb_semaphore = asyncio.Semaphore(64)  # Higher limit for CPU embeddings
            self.search_semaphore = asyncio.Semaphore(128)  # Higher limit for CPU search
        
        # Embedding batch size for efficient processing
        self.embedding_batch_size = 16
    
    async def __aenter__(self):
        """Enter async context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager"""
        pass
    
    def load_corpus(self) -> None:
        """Load corpus from JSONL file"""
        logger.info(f"Loading corpus from {self.config['data']['corpus_path']}")
        file_size = os.path.getsize(self.config['data']['corpus_path'])
        
        with open(self.config['data']['corpus_path'], "rb") as f:
            with tqdm(
                total=file_size,
                desc="Loading corpus",
                unit="B",
                unit_scale=True,
                ncols=100,
            ) as pbar:
                for raw_line in f:
                    pbar.update(len(raw_line))
                    line_data = orjson.loads(raw_line)

                    _id = str(line_data.get("id", ""))
                    _contents = line_data.get("contents", "")
                    if not isinstance(_contents, str):
                        _contents = str(_contents)

                    parts = _contents.split("\n", 1)
                    title = parts[0].strip()
                    text = parts[1] if len(parts) > 1 else ""

                    self._corpus.append({
                        "id": _id,
                        "title": title,
                        "text": text,
                        "contents": _contents,
                    })
        
        logger.info(f"Loaded {len(self._corpus)} documents")
    
    def load_index(self) -> None:
        """Load FAISS index"""
        logger.info(f"Loading FAISS index from {self.config['index']['path']}")
        self._index = faiss.read_index(self.config['index']['path'])
        self._index_dim = self._index.d
        logger.info(f"Index loaded. ntotal={self._index.ntotal}, dim={self._index.d}")
        
        # Move to GPU if configured
        if self.config['index']['use_gpu']:
            try:
                logger.info(f"Moving index to GPU using device {os.environ['CUDA_VISIBLE_DEVICES']}")
                
                # Initialize GPU resources
                res = faiss.StandardGpuResources()
                logger.info("Initialized GPU resources")
                
                # Configure GPU options
                gpu_options = faiss.GpuClonerOptions()
                gpu_options.useFloat16 = True  # Use float16 for better performance
                
                # Use device ID 0 since CUDA_VISIBLE_DEVICES is already set
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index, gpu_options)
                logger.info("Index moved to GPU successfully")
            except Exception as e:
                logger.error(f"Failed to move index to GPU: {str(e)}")
                logger.info("Falling back to CPU index")
                # Stay with CPU index if GPU fails
                self.config['index']['use_gpu'] = False
                # Update semaphores for CPU use
                self.emb_semaphore = asyncio.Semaphore(64)
                self.search_semaphore = asyncio.Semaphore(128)
    
    def init_client(self) -> None:
        """Initialize OpenAI client"""
        logger.info(f"Initializing OpenAI client with base_url: {self.config['embedding']['url']}")
        self._client = AsyncOpenAI(
            base_url=self.config['embedding']['url'],
            api_key=self.config['embedding']['api_key']
        )
    
    async def get_embeddings(self, queries: List[str]) -> np.ndarray:
        """
        Get embeddings for queries with efficient batching
        
        Args:
            queries: List of query strings
            
        Returns:
            Embeddings as numpy array
        """
        async with self.emb_semaphore:
            total_queries = len(queries)
            logger.info(f"Getting embeddings for {total_queries} queries with batch size {self.embedding_batch_size}")
            
            all_embeddings = []
            
            # Process in batches for better memory management and throughput
            for i in range(0, total_queries, self.embedding_batch_size):
                batch_queries = queries[i:i + self.embedding_batch_size]
                resp = await self._client.embeddings.create(
                    model=self.config['embedding']['model'], 
                    input=batch_queries
                )
                
                # Convert to numpy arrays
                batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
                all_embeddings.extend(batch_embeddings)
            
            if not all_embeddings:
                return np.array([], dtype=np.float32)
            
            return np.vstack(all_embeddings)
    
    async def search(self, embs: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS search
        
        Args:
            embs: Query embeddings
            topk: Number of results to return
            
        Returns:
            Distances and indices
        """
        async with self.search_semaphore:
            logger.info(f"Searching with {embs.shape[0]} queries, topk={topk}")
            D, I = await asyncio.to_thread(self._index.search, embs, topk)
            return D, I
    
    def build_results(self, I: np.ndarray, D: np.ndarray) -> Dict[str, List[List[Any]]]:
        """
        Build search results
        
        Args:
            I: Indices of matched documents
            D: Distances of matched documents
            
        Returns:
            Formatted search results
        """
        contents_batch, scores_batch = [], []
        N = len(self._corpus)
        
        for idxs, dists in zip(I, D):
            cur_c, cur_s = [], []
            for idx, dist in zip(idxs, dists):
                if idx == -1:
                    continue
                if 0 <= idx < N:
                    cur_c.append(self._corpus[idx])
                    cur_s.append(float(dist))
            contents_batch.append(cur_c)
            scores_batch.append(cur_s)
        
        return {"contents": contents_batch, "scores": scores_batch}
    
    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index"""
        return self._index
    
    @property
    def corpus(self) -> List[Dict[str, Any]]:
        """Get the corpus"""
        return self._corpus
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client"""
        return self._client
    
    @property
    def index_dim(self) -> int:
        """Get the index dimension"""
        return self._index_dim


# -------------------- singleton dependency --------------------
deps: Optional[ServiceDependencies] = None


def get_deps() -> ServiceDependencies:
    """Get service dependencies"""
    global deps
    if deps is None:
        raise RuntimeError("Service dependencies not initialized")
    return deps


# -------------------- startup and shutdown --------------------
def startup_event(config: Dict[str, Any]) -> None:
    """
    Service startup event
    
    Args:
        config: Configuration dictionary
    """
    global deps
    
    logger.info("Initializing service dependencies")
    
    # Set CUDA visible devices if using GPU
    if config['index']['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['index']['gpu_ids']
    
    # Initialize dependencies
    deps = ServiceDependencies(config)
    
    # Load resources
    deps.load_corpus()
    deps.load_index()
    deps.init_client()
    
    logger.info(f"Service initialized successfully on {config['server']['host']}:{config['server']['port']}")
    logger.info(f"Index: {config['index']['path']}, dim={deps.index_dim}, use_gpu={config['index']['use_gpu']}")
    logger.info(f"Corpus: {config['data']['corpus_path']}, size={len(deps.corpus)}")
    logger.info(f"Embedding: {config['embedding']['url']}, model={config['embedding']['model']}")


# -------------------- API routes --------------------
def create_app(config: Dict[str, Any]) -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="FAISS Retrieval Service",
        version="1.0.0",
        description="High-performance retrieval service based on FAISS and embedding models",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add static file serving
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Redirect root to static UI
    @app.get("/", response_class=RedirectResponse)
    async def root():
        return RedirectResponse(url="/static/index.html")
    
    # Startup event
    @app.on_event("startup")
    def _startup():
        startup_event(config)
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, description="Health check endpoint")
    def health(dependencies: ServiceDependencies = Depends(get_deps)) -> HealthResponse:
        """Health check endpoint"""
        return HealthResponse(
            status="ok",
            index_dim=dependencies.index_dim,
            corpus_size=len(dependencies.corpus),
            emb_url=config['embedding']['url'],
            emb_model=config['embedding']['model'],
            use_gpu=config['index']['use_gpu']
        )
    
    # Search endpoint
    @app.post("/search", response_model=SearchResponse, description="Search endpoint")
    async def search(
        request: SearchRequest,
        dependencies: ServiceDependencies = Depends(get_deps)
    ) -> SearchResponse:
        """Search endpoint"""
        try:
            # Validate topk - use soft limit instead of hard restriction
            max_topk = config['server']['max_topk']
            if request.topk > max_topk:
                logger.warning(f"topk={request.topk} exceeds server max_topk={max_topk}, using max_topk instead")
                actual_topk = max_topk
            else:
                actual_topk = request.topk
            
            # Get query embeddings
            embs = await dependencies.get_embeddings(request.queries)
            
            # Validate embedding dimension
            if embs.shape[1] != dependencies.index_dim:
                logger.error(f"Embedding dim {embs.shape[1]} != index dim {dependencies.index_dim}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding dim {embs.shape[1]} != index dim {dependencies.index_dim}. "
                    f"Check embedding model and FAISS index."
                )
            
            # Perform search
            D, I = await dependencies.search(embs, actual_topk)
            
            # Build results
            result = dependencies.build_results(I, D)
            
            logger.info(f"Search completed for {len(request.queries)} queries")
            return SearchResponse(**result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# -------------------- argument parsing --------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Retrieval Service")
    parser.add_argument(
        "--config", 
        type=str, 
        default="serve", 
        help="Configuration file name (without extension)"
    )
    return parser.parse_args()


# -------------------- main function --------------------
def main() -> None:
    """Main function"""
    args = parse_args()
    
    # Load configuration
    config = config_loader.load_config(args.config)
    
    # Create application
    app = create_app(config)
    
    # Create and run application
    import uvicorn
    
    # Configure uvicorn with optimized settings without workers
    uvicorn.run(
        app=app,
        host=config['server']['host'],
        port=config['server']['port'],
        log_level="info",
        reload=False,
        loop="uvloop",  # Use faster uvloop if available
        http="httptools",  # Use faster httptools implementation
        timeout_keep_alive=30,  # Keep-alive timeout
        backlog=1000,  # Maximum number of pending connections
    )


if __name__ == "__main__":
    main()
