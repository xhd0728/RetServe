"""
Type definitions and data models for the retrieval service.

This module contains type aliases and Pydantic models used throughout
the application for type safety and data validation.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Type Aliases
# =============================================================================

EmbeddingVector: TypeAlias = np.ndarray
"""Type alias for embedding vectors represented as numpy arrays."""

DocumentId: TypeAlias = str
"""Type alias for document identifiers."""

DistanceScore: TypeAlias = float
"""Type alias for similarity/distance scores."""


# =============================================================================
# Data Models
# =============================================================================

class Document(BaseModel):
    """
    Represents a document in the corpus.
    
    Attributes:
        id: Unique identifier for the document.
        title: Document title extracted from the first line of contents.
        text: Document body text (everything after the title).
        contents: Full original content including title and text.
    """
    
    id: DocumentId = Field(..., description="Unique document identifier")
    title: str = Field(default="", description="Document title")
    text: str = Field(default="", description="Document body text")
    contents: str = Field(default="", description="Full document content")
    
    model_config = {
        "frozen": False,
        "extra": "allow",
    }


class SearchResultItem(BaseModel):
    """
    Represents a single search result with document and score.
    
    Attributes:
        document: The matched document.
        score: Similarity score for this match.
    """
    
    document: Document = Field(..., description="Matched document")
    score: DistanceScore = Field(..., description="Similarity score")


class SearchResult(BaseModel):
    """
    Represents search results for a single query.
    
    Attributes:
        items: List of search result items ordered by relevance.
    """
    
    items: list[SearchResultItem] = Field(
        default_factory=list,
        description="Search results ordered by relevance"
    )
    
    @property
    def documents(self) -> list[Document]:
        """Extract documents from search results."""
        return [item.document for item in self.items]
    
    @property
    def scores(self) -> list[float]:
        """Extract scores from search results."""
        return [item.score for item in self.items]


# =============================================================================
# API Request/Response Models
# =============================================================================

class SearchRequest(BaseModel):
    """
    Search request model for the API.
    
    Attributes:
        queries: List of query strings to search for.
        top_k: Number of top results to return per query.
    """
    
    queries: list[str] = Field(..., description="List of query strings")
    top_k: int = Field(
        default=5, 
        gt=0, 
        alias="topk",
        description="Number of top results per query"
    )
    
    @field_validator("queries")
    @classmethod
    def validate_queries_not_empty(cls, value: list[str]) -> list[str]:
        """Validate that queries list is not empty."""
        if not value:
            raise ValueError("queries cannot be empty")
        return value
    
    model_config = {
        "populate_by_name": True,
    }


class SearchResponse(BaseModel):
    """
    Search response model for the API.
    
    Attributes:
        contents: Nested list of document dictionaries for each query.
        scores: Nested list of similarity scores for each query.
    """
    
    contents: list[list[dict[str, Any]]] = Field(
        ..., 
        description="Documents for each query"
    )
    scores: list[list[float]] = Field(
        ..., 
        description="Scores for each query"
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status: Service status string.
        index_dimension: Dimension of the vector index.
        corpus_size: Number of documents in the corpus.
        embedding_url: URL of the embedding service.
        embedding_model: Name of the embedding model.
        gpu_enabled: Whether GPU acceleration is enabled.
    """
    
    status: str = Field(..., description="Service status")
    index_dimension: int = Field(..., alias="index_dim", description="Vector dimension")
    corpus_size: int = Field(..., description="Number of documents")
    embedding_url: str = Field(..., alias="emb_url", description="Embedding service URL")
    embedding_model: str = Field(..., alias="emb_model", description="Embedding model name")
    gpu_enabled: bool = Field(..., alias="use_gpu", description="GPU acceleration status")
    
    model_config = {
        "populate_by_name": True,
    }
