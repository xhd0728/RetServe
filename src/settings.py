"""
Type-safe configuration settings using Pydantic.

This module provides structured configuration classes that replace
dictionary-based configs with type-safe, validated settings objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# =============================================================================
# Model Settings (for embedding processor)
# =============================================================================

class ModelSettings(BaseModel):
    """
    Embedding model configuration for local processing.
    
    Attributes:
        path: Path to the model directory.
        batch_size: Batch size for embedding generation.
        gpu_device_ids: Comma-separated GPU device IDs.
        pooling_method: Pooling method for embeddings.
        better_transformer: Whether to use BetterTransformer.
        model_warmup: Whether to warm up the model.
        trust_remote_code: Whether to trust remote code.
        device: Device to run the model on.
    """
    
    path: str = Field(..., description="Model directory path")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    gpu_device_ids: str = Field(default="0", alias="gpu_ids", description="GPU device IDs")
    pooling_method: str = Field(default="auto", description="Pooling method")
    better_transformer: bool = Field(default=False, alias="bettertransformer")
    model_warmup: bool = Field(default=False, description="Warm up model")
    trust_remote_code: bool = Field(default=True, description="Trust remote code")
    device: str = Field(default="cuda", description="Compute device")
    
    @property
    def model_path(self) -> Path:
        """Get model path as Path object."""
        return Path(self.path)
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


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
        model: Model configuration.
        data: Data file configuration.
    """
    
    model: ModelSettings
    data: DataSettings
    
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
            model=ModelSettings(**config.get("model", {})),
            data=DataSettings(**config.get("data", {})),
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
