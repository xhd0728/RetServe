import os
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
    def __init__(self, embeddings: np.ndarray | dict[str, np.ndarray]) -> None:
        self._embeddings = embeddings
        self.requests: list[list[str]] = []

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.requests.append(list(texts))

        if isinstance(self._embeddings, dict):
            return np.vstack([self._embeddings[text] for text in texts]).astype(
                np.float32
            )

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


def make_settings(
    query_cache_enabled: bool = False,
    query_cache_size: int = 4096,
    server: ServerSettings | None = None,
) -> ServiceSettings:
    return ServiceSettings(
        server=server or ServerSettings(),
        index=IndexSettings(path="unused.index"),
        data=DataSettings(corpus_path="unused.jsonl"),
        embedding=EmbeddingSettings(
            url="http://example.test/v1",
            model="test-model",
            query_cache_enabled=query_cache_enabled,
            query_cache_size=query_cache_size,
        ),
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

    async def test_top_k_limited_by_index_size_and_vectors_cast_for_search(
        self,
    ) -> None:
        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)[:, ::2]
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0, 1]], dtype=np.int64),
            distances=np.array([[1.0, 0.5]]),
        )
        container = ServiceContainer(
            settings=make_settings(),
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

    async def test_legacy_top_k_cap_setting_does_not_limit_search(self) -> None:
        vector_index = FakeVectorIndex(
            dimension=2,
            size=4,
            indices=np.array([[0, 1, -1]], dtype=np.int64),
            distances=np.array([[1.0, 0.5, 0.0]]),
        )
        legacy_server_settings = ServerSettings(**{"max" + "_topk": 1})
        container = ServiceContainer(
            settings=make_settings(server=legacy_server_settings),
            embedding_client=FakeEmbeddingClient(
                np.array([[1.0, 2.0]], dtype=np.float32)
            ),
            vector_index=vector_index,
            corpus=make_documents(),
        )

        response = await container.search(["query"], top_k=3)

        self.assertEqual(vector_index.last_top_k, 3)
        self.assertEqual(len(response.contents[0]), 2)

    async def test_query_embedding_cache_reuses_repeated_query(self) -> None:
        embedding_client = FakeEmbeddingClient(
            {"query": np.array([1.0, 2.0], dtype=np.float32)}
        )
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0], [1]], dtype=np.int64),
            distances=np.array([[1.0], [0.5]]),
        )
        container = ServiceContainer(
            settings=make_settings(query_cache_enabled=True, query_cache_size=4096),
            embedding_client=embedding_client,
            vector_index=vector_index,
            corpus=make_documents(),
        )

        await container.search(["query"], top_k=1)
        await container.search(["query"], top_k=1)

        self.assertEqual(embedding_client.requests, [["query"]])

    async def test_query_embedding_cache_preserves_batch_order(self) -> None:
        embedding_client = FakeEmbeddingClient(
            {
                "b": np.array([2.0, 20.0], dtype=np.float32),
                "a": np.array([1.0, 10.0], dtype=np.float32),
            }
        )
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0], [1], [0]], dtype=np.int64),
            distances=np.array([[1.0], [0.5], [0.25]]),
        )
        container = ServiceContainer(
            settings=make_settings(query_cache_enabled=True, query_cache_size=4096),
            embedding_client=embedding_client,
            vector_index=vector_index,
            corpus=make_documents(),
        )

        await container.search(["b", "a", "b"], top_k=1)

        self.assertEqual(embedding_client.requests, [["b", "a"]])
        np.testing.assert_array_equal(
            vector_index.last_query_vectors,
            np.array([[2.0, 20.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        )

    async def test_query_embedding_cache_is_disabled_by_default(self) -> None:
        embedding_client = FakeEmbeddingClient(
            {"query": np.array([1.0, 2.0], dtype=np.float32)}
        )
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0]], dtype=np.int64),
            distances=np.array([[1.0]]),
        )
        container = ServiceContainer(
            settings=make_settings(),
            embedding_client=embedding_client,
            vector_index=vector_index,
            corpus=make_documents(),
        )

        await container.search(["query"], top_k=1)
        await container.search(["query"], top_k=1)

        self.assertEqual(embedding_client.requests, [["query"], ["query"]])

    async def test_query_embedding_cache_size_zero_disables_cache(self) -> None:
        embedding_client = FakeEmbeddingClient(
            {"query": np.array([1.0, 2.0], dtype=np.float32)}
        )
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0]], dtype=np.int64),
            distances=np.array([[1.0]]),
        )
        container = ServiceContainer(
            settings=make_settings(query_cache_enabled=True, query_cache_size=0),
            embedding_client=embedding_client,
            vector_index=vector_index,
            corpus=make_documents(),
        )

        await container.search(["query"], top_k=1)
        await container.search(["query"], top_k=1)

        self.assertEqual(embedding_client.requests, [["query"], ["query"]])

    async def test_query_embedding_cache_size_can_be_set_by_environment(
        self,
    ) -> None:
        embedding_client = FakeEmbeddingClient(
            {
                "a": np.array([1.0, 2.0], dtype=np.float32),
                "b": np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        vector_index = FakeVectorIndex(
            dimension=2,
            size=2,
            indices=np.array([[0]], dtype=np.int64),
            distances=np.array([[1.0]]),
        )

        with patch.dict(
            os.environ,
            {
                "RET_SERVE_QUERY_CACHE_ENABLED": "true",
                "RET_SERVE_QUERY_CACHE_SIZE": "1",
            },
        ):
            container = ServiceContainer(
                settings=make_settings(),
                embedding_client=embedding_client,
                vector_index=vector_index,
                corpus=make_documents(),
            )

            await container.search(["a"], top_k=1)
            await container.search(["b"], top_k=1)
            await container.search(["a"], top_k=1)

        self.assertEqual(embedding_client.requests, [["a"], ["b"], ["a"]])

    async def test_invalid_query_cache_size_environment_raises(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RET_SERVE_QUERY_CACHE_ENABLED": "true",
                "RET_SERVE_QUERY_CACHE_SIZE": "invalid",
            },
        ):
            with self.assertRaisesRegex(ValueError, "RET_SERVE_QUERY_CACHE_SIZE"):
                ServiceContainer(
                    settings=make_settings(),
                    embedding_client=FakeEmbeddingClient(
                        np.array([[1.0, 2.0]], dtype=np.float32)
                    ),
                    vector_index=FakeVectorIndex(
                        dimension=2,
                        size=2,
                        indices=np.array([[0]], dtype=np.int64),
                        distances=np.array([[1.0]]),
                    ),
                    corpus=make_documents(),
                )

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

        response.contents[0][0]["title"] = "Changed"
        next_response = await container.search(["query"], top_k=1)

        self.assertEqual(next_response.contents[0][0]["title"], "Title 2")


if __name__ == "__main__":
    unittest.main()
