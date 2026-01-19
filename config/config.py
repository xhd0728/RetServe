import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class EmbeddingConfig(BaseSettings):
    """嵌入配置"""
    model_path: str = "./models/Qwen3-Embedding-0.6B/"
    batch_size: int = 4
    gpu_ids: str = "5"
    pooling_method: str = "auto"
    bettertransformer: bool = False
    model_warmup: bool = False
    trust_remote_code: bool = True
    device: str = "cuda"
    
    class Config:
        env_prefix = "EMB_"


class IndexConfig(BaseSettings):
    """索引配置"""
    index_chunk_size: int = 50000
    faiss_use_gpu: bool = True
    
    class Config:
        env_prefix = "INDEX_"


class ServeConfig(BaseSettings):
    """服务配置"""
    faiss_index_path: str = "./data/san_guo_yan_yi.index"
    corpus_jsonl_path: str = "./data/san_guo_yan_yi.jsonl"
    emb_url: str = "http://localhost:65504/v1"
    emb_model: str = "qwen-embedding"
    max_topk: int = 999
    api_key: str = "None"
    gpu_ids: str = "5"
    use_gpu: bool = True
    port: int = 8000
    host: str = "0.0.0.0"
    
    class Config:
        env_prefix = "SERVE_"


class DataConfig(BaseSettings):
    """数据配置"""
    corpus_path: str = "./data/san_guo_yan_yi.jsonl"
    embedding_path: str = "./data/san_guo_yan_yi.npy"
    
    class Config:
        env_prefix = "DATA_"


class LogConfig(BaseSettings):
    """日志配置"""
    level: str = "INFO"
    file: str = "logs/ret_serve.log"
    
    class Config:
        env_prefix = "LOG_"


# 创建全局配置实例
emb_config = EmbeddingConfig()
index_config = IndexConfig()
serve_config = ServeConfig()
data_config = DataConfig()
log_config = LogConfig()
