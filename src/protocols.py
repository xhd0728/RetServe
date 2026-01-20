"""
Protocol definitions for the retrieval service.

This module defines abstract interfaces (protocols) that establish contracts
for the core components of the retrieval system. Using protocols enables
loose coupling and dependency injection patterns.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from src.types import Document


# =============================================================================
# Embedding Client Protocol
# =============================================================================

@runtime_checkable
class EmbeddingClient(Protocol):
    """
    Protocol for embedding clients that convert text to vectors.
    
    Implementations should handle batching and async operations
    for efficient embedding generation.
    """
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            numpy.ndarray: Embedding vectors with shape (len(texts), dimension).
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...


# =============================================================================
# Vector Index Protocol
# =============================================================================

@runtime_checkable
class VectorIndex(Protocol):
    """
    Protocol for vector indices that support similarity search.
    
    Implementations should handle vector storage and efficient
    nearest neighbor search operations.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the dimension of vectors in this index.
        
        Returns:
            int: Vector dimension.
        """
        ...
    
    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the number of vectors in this index.
        
        Returns:
            int: Number of indexed vectors.
        """
        ...
    
    @abstractmethod
    async def search(
        self, 
        query_vectors: np.ndarray, 
        top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors for each query vector.
        
        Args:
            query_vectors: Query vectors with shape (num_queries, dimension).
            top_k: Number of nearest neighbors to return.
            
        Returns:
            Tuple of (distances, indices) where:
                - distances: Shape (num_queries, top_k) with similarity scores.
                - indices: Shape (num_queries, top_k) with document indices.
                
        Raises:
            SearchError: If search operation fails.
        """
        ...


# =============================================================================
# Corpus Loader Protocol
# =============================================================================

@runtime_checkable
class CorpusLoader(Protocol):
    """
    Protocol for corpus loaders that read documents from storage.
    
    Implementations should handle various file formats and
    provide progress feedback for large corpora.
    """
    
    @abstractmethod
    def load(self) -> list["Document"]:
        """
        Load documents from the corpus source.
        
        Returns:
            List of Document objects.
            
        Raises:
            CorpusLoadError: If loading fails.
        """
        ...


# =============================================================================
# Index Builder Protocol
# =============================================================================

@runtime_checkable
class IndexBuilder(Protocol):
    """
    Protocol for building vector indices from embeddings.
    
    Implementations should handle efficient batch indexing
    with progress feedback.
    """
    
    @abstractmethod
    def build(self, embeddings: np.ndarray) -> VectorIndex:
        """
        Build a vector index from embeddings.
        
        Args:
            embeddings: Embedding vectors with shape (num_docs, dimension).
            
        Returns:
            VectorIndex: The built vector index.
            
        Raises:
            IndexBuildError: If index building fails.
        """
        ...
    
    @abstractmethod
    def save(self, index: VectorIndex, path: str) -> None:
        """
        Save the index to persistent storage.
        
        Args:
            index: The vector index to save.
            path: File path for saving.
            
        Raises:
            IndexSaveError: If saving fails.
        """
        ...
