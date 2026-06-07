from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class EmbeddingConfig(BaseSettings):
    """Embedding pipeline configuration."""

    url: str = "http://localhost:8000/v1"
    model: str = "qwen3-embedding-0.6b"
    api_key: str = "None"
    api_key_env: str = "RET_SERVE_EMBED_API_KEY"
    batch_size: int = 128
    concurrency_limit: int = 16
    encode_batch_size: int = 4096
    query_cache_enabled: bool = False
    query_cache_size: int = 4096
    request_timeout: float = 120.0
    max_retries: int = 2
    normalize: bool = True

    class Config:
        env_prefix = "EMB_"


class IndexConfig(BaseSettings):
    """Index building configuration."""

    index_chunk_size: int = 50000
    faiss_use_gpu: bool = True

    class Config:
        env_prefix = "INDEX_"


class ServeConfig(BaseSettings):
    """Retrieval service configuration."""

    faiss_index_path: str = "./data/example_faiss.index"
    corpus_jsonl_path: str = "./data/example_corpus.jsonl"
    emb_url: str = "http://localhost:8000/v1"
    emb_model: str = "qwen3-embedding-0.6b"
    api_key: str = "None"
    api_key_env: str = "RET_SERVE_EMBED_API_KEY"
    gpu_ids: str = "0"
    use_gpu: bool = False
    search_workers: int = 128
    omp_threads: int | None = None
    port: int = 8000
    host: str = "0.0.0.0"

    class Config:
        env_prefix = "SERVE_"


class DataConfig(BaseSettings):
    """Data file configuration."""

    corpus_path: str = "./data/example_corpus.jsonl"
    embedding_path: str = "./data/example_embeddings.npy"

    class Config:
        env_prefix = "DATA_"


class LogConfig(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "logs/ret_serve.log"

    class Config:
        env_prefix = "LOG_"


class MetricsConfig(BaseSettings):
    """Metrics endpoint configuration."""

    enabled: bool = True

    class Config:
        env_prefix = "METRICS_"


emb_config = EmbeddingConfig()
index_config = IndexConfig()
serve_config = ServeConfig()
data_config = DataConfig()
log_config = LogConfig()
metrics_config = MetricsConfig()
