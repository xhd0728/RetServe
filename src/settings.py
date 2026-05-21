"""
Type-safe configuration settings using Pydantic.

This module provides structured configuration classes that replace
dictionary-based configs with type-safe, validated settings objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Server Settings
# =============================================================================

class ServerSettings(BaseModel):
    """
    HTTP server configuration settings.
    
    Attributes:
        host: Server bind address.
        port: Server listen port.
        max_top_k: Maximum allowed top_k value for search requests.
    """
    
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8088, ge=1, le=65535, description="Server port")
    max_top_k: int = Field(default=999, ge=1, alias="max_topk", description="Maximum top_k")
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# =============================================================================
# Index Settings
# =============================================================================

class IndexSettings(BaseModel):
    """
    Vector index configuration settings.
    
    Attributes:
        path: Path to the FAISS index file (required for serving, optional for building).
        use_gpu: Whether to use GPU acceleration.
        gpu_device_ids: Comma-separated GPU device IDs.
        chunk_size: Batch size for index building.
    """
    
    path: str = Field(default="", description="Path to index file")
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    gpu_device_ids: str = Field(default="0", alias="gpu_ids", description="GPU device IDs")
    chunk_size: int = Field(default=50000, ge=1, description="Index building batch size")
    mmap_embeddings: bool = Field(
        default=True,
        alias="mmap",
        description="Memory-map embeddings while building the index",
    )
    normalize: bool = Field(
        default=False,
        description="L2-normalize vectors before adding them to an inner-product index",
    )
    search_concurrency_limit: int = Field(
        default=128,
        ge=1,
        description="Maximum concurrent FAISS search calls for CPU indexes",
    )
    
    @property
    def index_path(self) -> Path:
        """Get index path as Path object."""
        return Path(self.path)
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# =============================================================================
# Data Settings
# =============================================================================

class DataSettings(BaseModel):
    """
    Data file configuration settings.
    
    Attributes:
        corpus_path: Path to the corpus JSONL file.
        embedding_path: Path to the embeddings numpy file.
        index_path: Path to the FAISS index file.
    """
    
    corpus_path: str = Field(default="", description="Path to corpus file")
    embedding_path: str = Field(default="", description="Path to embeddings file")
    index_path: str = Field(default="", description="Path to index file")
    
    @property
    def corpus_file(self) -> Path:
        """Get corpus path as Path object."""
        return Path(self.corpus_path)
    
    @property
    def embedding_file(self) -> Path:
        """Get embedding path as Path object."""
        return Path(self.embedding_path)
    
    @property
    def index_file(self) -> Path:
        """Get index path as Path object."""
        return Path(self.index_path)
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# =============================================================================
# Embedding Settings
# =============================================================================

class EmbeddingSettings(BaseModel):
    """
    Embedding service configuration settings.
    
    Attributes:
        base_url: Base URL of the embedding API.
        model_name: Name of the embedding model.
        api_key: API key for authentication.
    """
    
    base_url: str = Field(..., alias="url", description="Embedding API base URL")
    model_name: str = Field(..., alias="model", description="Embedding model name")
    api_key: str = Field(default="None", description="API key")
    api_key_env: Optional[str] = Field(
        default=None,
        description="Environment variable that contains the API key",
    )
    batch_size: int = Field(default=16, ge=1, description="Texts per API request")
    concurrency_limit: int = Field(
        default=32,
        ge=1,
        description="Maximum concurrent embedding API requests",
    )
    request_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Embedding API request timeout in seconds",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Embedding API retry count",
    )
    normalize: bool = Field(
        default=False,
        description="L2-normalize returned embeddings",
    )
    dimensions: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional embedding dimensions request parameter",
    )
    encode_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Texts per offline streaming write batch",
    )
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }

    @property
    def resolved_api_key(self) -> str:
        """Resolve the API key from api_key_env first, then api_key."""
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                return value

        if self.api_key.startswith("$"):
            value = os.environ.get(self.api_key[1:])
            if value:
                return value

        return self.api_key


# =============================================================================
# Logging Settings
# =============================================================================

class LoggingSettings(BaseModel):
    """
    Logging configuration settings.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        file: Path to the log file.
        max_bytes: Maximum log file size in bytes.
        backup_count: Number of backup log files to keep.
    """
    
    level: str = Field(default="INFO", description="Logging level")
    file: str = Field(default="logs/app.log", description="Log file path")
    max_bytes: int = Field(default=10 * 1024 * 1024, description="Max log file size")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    @field_validator("level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_value = value.upper()
        if upper_value not in valid_levels:
            raise ValueError(f"Invalid log level: {value}. Must be one of {valid_levels}")
        return upper_value
    
    @property
    def log_file(self) -> Path:
        """Get log file path as Path object."""
        return Path(self.file)
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# =============================================================================
# Composite Settings
# =============================================================================

class ServiceSettings(BaseModel):
    """
    Complete service configuration combining all settings.
    
    Attributes:
        server: Server configuration.
        index: Index configuration.
        data: Data file configuration.
        embedding: Embedding service configuration.
    """
    
    server: ServerSettings = Field(default_factory=ServerSettings)
    index: IndexSettings
    data: DataSettings = Field(default_factory=DataSettings)
    embedding: EmbeddingSettings
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ServiceSettings":
        """
        Create ServiceSettings from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with nested sections.
            
        Returns:
            ServiceSettings instance.
        """
        return cls(
            server=ServerSettings(**config.get("server", {})),
            index=IndexSettings(**config.get("index", {})),
            data=DataSettings(**config.get("data", {})),
            embedding=EmbeddingSettings(**config.get("embedding", {})),
        )
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class EmbedSettings(BaseModel):
    """
    Configuration for the embedding processor.
    
    Attributes:
        data: Data file configuration.
        embedding: OpenAI-compatible embedding API configuration.
    """
    
    data: DataSettings
    embedding: EmbeddingSettings
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "EmbedSettings":
        """
        Create EmbedSettings from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with nested sections.
            
        Returns:
            EmbedSettings instance.
        """
        return cls(
            data=DataSettings(**config.get("data", {})),
            embedding=EmbeddingSettings(**config.get("embedding", {})),
        )
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class IndexBuildSettings(BaseModel):
    """
    Configuration for the index builder.
    
    Attributes:
        index: Index building configuration.
        data: Data file configuration.
    """
    
    index: IndexSettings
    data: DataSettings
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "IndexBuildSettings":
        """
        Create IndexBuildSettings from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with nested sections.
            
        Returns:
            IndexBuildSettings instance.
        """
        return cls(
            index=IndexSettings(**config.get("index", {})),
            data=DataSettings(**config.get("data", {})),
        )
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
