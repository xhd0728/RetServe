import unittest
from unittest.mock import patch

import numpy as np

from src.service_container import ServiceContainer
from src.settings import (
    DataSettings,
    EmbeddingSettings,
    IndexSettings,
    ServerSettings,
    ServiceSettings,
)
from src.types import Document


class FakeEmbeddingClient:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
        self.requests: list[list[str]] = []

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.requests.append(list(texts))
        return self._embeddings


class FakeVectorIndex:
    def __init__(
        self,
        dimension: int,
        size: int,
        indices: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        self._dimension = dimension
        self._size = size
        self._indices = indices
        self._distances = distances
        self.search_called = False
        self.last_top_k: int | None = None
        self.last_query_vectors: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return self._size

    async def search(
        self,
        query_vectors: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.search_called = True
        self.last_top_k = top_k
        self.last_query_vectors = query_vectors
        return self._distances[:, :top_k], self._indices[:, :top_k]


def make_settings(max_top_k: int = 999) -> ServiceSettings:
    return ServiceSettings(
        server=ServerSettings(max_top_k=max_top_k),
        index=IndexSettings(path="unused.index"),
        data=DataSettings(corpus_path="unused.jsonl"),
        embedding=EmbeddingSettings(url="http://example.test/v1", model="test-model"),
    )


def make_documents() -> list[Document]:
    return [
        Document(
            id="doc-1",
            title="Title 1",
            text="Text 1",
            contents="Title 1\nText 1",
        ),
        Document(
            id="doc-2",
            title="Title 2",
            text="Text 2",
            contents="Title 2\nText 2",
        ),
    ]


class ServiceContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_response_skips_invalid_indices(self) -> None:
        container = ServiceContainer(
            settings=make_settings(),
            embedding_client=FakeEmbeddingClient(
                np.array([[1.0, 2.0]], dtype=np.float32)
            ),
            vector_index=FakeVectorIndex(
                dimension=2,
                size=4,
                indices=np.array([[0, -1, 99, 1]], dtype=np.int64),
                distances=np.array([[0.9, 0.8, 0.7, 0.6]]),
            ),
            corpus=make_documents(),
        )

        response = await container.search(["query"], top_k=4)

        self.assertEqual(
            response.contents,
            [
                [
                    {
                        "id": "doc-1",
                        "title": "Title 1",
                        "text": "Text 1",
                        "contents": "Title 1\nText 1",
                    },
                    {
                        "id": "doc-2",
                        "title": "Title 2",
                        "text": "Text 2",
                        "contents": "Title 2\nText 2",
                    },
                ]
            ],
        )
        self.assertEqual(response.scores, [[0.9, 0.6]])

    async def test_top_k_limited_and_vectors_cast_for_search(self) -> None:
        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)[:, ::2]
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0, 1]], dtype=np.int64),
            distances=np.array([[1.0, 0.5]]),
        )
        container = ServiceContainer(
            settings=make_settings(max_top_k=3),
            embedding_client=FakeEmbeddingClient(embeddings),
            vector_index=vector_index,
            corpus=make_documents(),
        )

        response = await container.search(["query"], top_k=99)

        self.assertEqual(vector_index.last_top_k, 2)
        self.assertIsNotNone(vector_index.last_query_vectors)
        self.assertEqual(vector_index.last_query_vectors.dtype, np.float32)
        self.assertTrue(vector_index.last_query_vectors.flags.c_contiguous)
        self.assertEqual(len(response.contents[0]), 2)
        self.assertEqual(response.scores, [[1.0, 0.5]])

    async def test_dimension_mismatch_raises_validation_error(self) -> None:
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0, 1]], dtype=np.int64),
            distances=np.array([[1.0, 0.5]]),
        )
        container = ServiceContainer(
            settings=make_settings(),
            embedding_client=FakeEmbeddingClient(
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            ),
            vector_index=vector_index,
            corpus=make_documents(),
        )

        with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch"):
            await container.search(["query"], top_k=2)

        self.assertFalse(vector_index.search_called)

    async def test_cached_document_payloads_avoid_repeated_model_dump(self) -> None:
        container = ServiceContainer(
            settings=make_settings(),
            embedding_client=FakeEmbeddingClient(
                np.array([[1.0, 2.0]], dtype=np.float32)
            ),
            vector_index=FakeVectorIndex(
                dimension=2,
                size=2,
                indices=np.array([[1]], dtype=np.int64),
                distances=np.array([[0.5]]),
            ),
            corpus=make_documents(),
        )

        with patch.object(Document, "model_dump", side_effect=AssertionError):
            response = await container.search(["query"], top_k=1)

        self.assertEqual(response.contents[0][0]["id"], "doc-2")
        self.assertEqual(response.scores, [[0.5]])


if __name__ == "__main__":
    unittest.main()
