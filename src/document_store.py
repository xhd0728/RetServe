"""Document storage helpers for online retrieval responses."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from src.types import Document


class DocumentStore:
    """Read-only document payload cache keyed by corpus row position."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._payloads: tuple[Mapping[str, Any], ...] = tuple(
            MappingProxyType(document.model_dump()) for document in documents
        )

    @property
    def documents(self) -> list[Document]:
        """Return the loaded document models."""
        return self._documents

    @property
    def size(self) -> int:
        """Return the number of loaded documents."""
        return len(self._payloads)

    def payload_copy(self, index: int) -> dict[str, Any] | None:
        """Return a mutable response copy for a valid corpus index."""
        if 0 <= index < len(self._payloads):
            return dict(self._payloads[index])
        return None
