import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config_loader import ConfigLoader


class ConfigLoaderTests(unittest.TestCase):
    def test_service_settings_support_nested_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir)
            (config_dir / "serve.yaml").write_text(
                """
server:
  host: "0.0.0.0"
  port: 8088
index:
  path: "index.faiss"
  search_workers: 8
data:
  corpus_path: "corpus.jsonl"
embedding:
  url: "http://example.test/v1"
  model: "test-model"
metrics:
  enabled: true
""",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "RET_SERVE__SERVER__PORT": "9000",
                    "RET_SERVE__INDEX__SEARCH_WORKERS": "16",
                    "RET_SERVE__INDEX__OMP_THREADS": "4",
                    "RET_SERVE__METRICS__ENABLED": "false",
                },
            ):
                settings = ConfigLoader(config_dir).load_service_settings("serve")

        self.assertEqual(settings.server.port, 9000)
        self.assertEqual(settings.index.search_workers, 16)
        self.assertEqual(settings.index.omp_threads, 4)
        self.assertFalse(settings.metrics.enabled)

    def test_legacy_search_concurrency_limit_still_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir)
            (config_dir / "serve.yaml").write_text(
                """
index:
  path: "index.faiss"
  search_concurrency_limit: 12
data:
  corpus_path: "corpus.jsonl"
embedding:
  url: "http://example.test/v1"
  model: "test-model"
""",
                encoding="utf-8",
            )

            settings = ConfigLoader(config_dir).load_service_settings("serve")

        self.assertEqual(settings.index.search_workers, 12)


if __name__ == "__main__":
    unittest.main()
