import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.errors import (
    EmbeddingDimensionError,
    EmbeddingUpstreamError,
    IndexNotReadyError,
    RetrievalExecutionError,
)
from src.ret_serve import create_application
from src.settings import DataSettings, EmbeddingSettings, IndexSettings, ServiceSettings
from src.types import SearchResponse


def make_settings() -> ServiceSettings:
    return ServiceSettings(
        index=IndexSettings(path="unused.index"),
        data=DataSettings(corpus_path="unused.jsonl"),
        embedding=EmbeddingSettings(url="http://example.test/v1", model="test-model"),
    )


class FakeRuntime:
    def __init__(self, error: Exception | None = None) -> None:
        self.settings = make_settings()
        self.vector_index = SimpleNamespace(dimension=2, size=2)
        self.corpus_size = 2
        self.ready = True
        self.error = error
        self.requests: list[tuple[list[str], int]] = []

    async def search(self, queries: list[str], top_k: int) -> SearchResponse:
        self.requests.append((queries, top_k))
        if self.error is not None:
            raise self.error
        return SearchResponse(
            contents=[[{"id": "doc-1", "contents": "Title\nText"}]],
            scores=[[0.9]],
        )

    def render_metrics(self) -> str:
        return "retserve_ready 1\nretserve_requests_total 1\n"


async def noop_close_service() -> None:
    return None


class RuntimeApiTests(unittest.TestCase):
    def _client(self, runtime: FakeRuntime) -> TestClient:
        app = create_application(make_settings())
        patches = [
            patch("src.ret_serve.initialize_service"),
            patch("src.ret_serve.get_runtime", return_value=runtime),
            patch("src.ret_serve.close_service", new=noop_close_service),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return TestClient(app)

    def test_operational_endpoints(self) -> None:
        runtime = FakeRuntime()
        with self._client(runtime) as client:
            self.assertEqual(client.get("/livez").json(), {"status": "ok"})
            self.assertEqual(client.get("/readyz").json()["status"], "ready")
            self.assertEqual(client.get("/health").json()["index_dim"], 2)

            metrics_response = client.get("/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            self.assertIn("retserve_ready", metrics_response.text)

    def test_search_uses_runtime_response_shape(self) -> None:
        runtime = FakeRuntime()
        with self._client(runtime) as client:
            response = client.post(
                "/search",
                json={"queries": ["query"], "topk": 1},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json().keys()), {"contents", "scores"})
        self.assertEqual(runtime.requests, [(["query"], 1)])

    def test_search_error_status_mapping(self) -> None:
        cases = [
            (EmbeddingDimensionError("dimension mismatch"), 400),
            (EmbeddingUpstreamError("provider failed"), 502),
            (IndexNotReadyError("not ready"), 503),
            (RetrievalExecutionError("failed"), 500),
        ]

        for error, expected_status in cases:
            with self.subTest(error=type(error).__name__):
                runtime = FakeRuntime(error=error)
                with self._client(runtime) as client:
                    response = client.post(
                        "/search",
                        json={"queries": ["query"], "topk": 1},
                    )
                self.assertEqual(response.status_code, expected_status)


if __name__ == "__main__":
    unittest.main()
