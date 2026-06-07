import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.document_store import DocumentStore
from src.index import FAISSIndexBuilder, IndexSaver
from src.metrics import MetricsRegistry
from src.retrieval import RetrievalEngine
from src.runtime import RetServeRuntime
from src.settings import DataSettings, EmbeddingSettings, IndexSettings, ServiceSettings
from src.types import Document
from src.vector_index import FAISSVectorIndex


class FakeEmbeddingClient:
    async def embed(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "api query": np.array([1.0, 0.0], dtype=np.float32),
            "index query": np.array([0.0, 1.0], dtype=np.float32),
        }
        return np.vstack([vectors[text] for text in texts]).astype(np.float32)

    async def close(self) -> None:
        return None


def make_documents() -> list[Document]:
    return [
        Document(id="doc-1", title="API", text="HTTP API", contents="API\nHTTP API"),
        Document(
            id="doc-2",
            title="Index",
            text="FAISS index",
            contents="Index\nFAISS index",
        ),
    ]


class RuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_search_with_temp_faiss_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "test.faiss"
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            index = FAISSIndexBuilder(chunk_size=1).build(embeddings)
            IndexSaver(index_path).save(index)

            vector_index = FAISSVectorIndex(
                index_path=str(index_path),
                search_concurrency_limit=2,
                search_workers=2,
            )
            vector_index.load()
            document_store = DocumentStore(make_documents())
            metrics = MetricsRegistry(enabled=True)
            settings = ServiceSettings(
                index=IndexSettings(path=str(index_path), search_workers=2),
                data=DataSettings(corpus_path="unused.jsonl"),
                embedding=EmbeddingSettings(
                    url="http://example.test/v1",
                    model="test-model",
                ),
            )
            engine = RetrievalEngine(
                embedding_client=FakeEmbeddingClient(),
                vector_index=vector_index,
                document_store=document_store,
                metrics=metrics,
            )
            runtime = RetServeRuntime(
                settings=settings,
                embedding_client=FakeEmbeddingClient(),
                vector_index=vector_index,
                document_store=document_store,
                metrics=metrics,
                engine=engine,
            )

            response = await runtime.search(["api query", "index query"], top_k=1)
            await runtime.close()

        self.assertEqual(response.contents[0][0]["id"], "doc-1")
        self.assertEqual(response.contents[1][0]["id"], "doc-2")
        metrics_text = metrics.render_prometheus()
        self.assertIn("retserve_requests_total", metrics_text)
        self.assertIn("retserve_faiss_search_seconds", metrics_text)


if __name__ == "__main__":
    unittest.main()
