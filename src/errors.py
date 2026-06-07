"""Typed errors for retrieval service failure handling."""

from __future__ import annotations


class RetServeError(Exception):
    """Base class for service errors that can be mapped to HTTP responses."""


class ConfigurationError(RetServeError):
    """Configuration or startup resource error."""


class EmbeddingUpstreamError(RetServeError):
    """Embedding provider failed or returned an invalid response."""


class EmbeddingDimensionError(ValueError, RetServeError):
    """Embedding dimensions do not match the loaded index."""


class IndexNotReadyError(RetServeError):
    """Search was requested before the vector index was ready."""


class RetrievalExecutionError(RetServeError):
    """Unexpected retrieval execution failure."""
