import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.vector_index import FAISSVectorIndex


class FakeGpuClonerOptions:
    def __init__(self) -> None:
        self.useFloat16 = False


class FakeGpuMultipleClonerOptions(FakeGpuClonerOptions):
    def __init__(self) -> None:
        super().__init__()
        self.shard = False


class FAISSVectorIndexGpuTests(unittest.TestCase):
    def _make_index(self, gpu_ids: str) -> FAISSVectorIndex:
        vector_index = FAISSVectorIndex(
            index_path="unused.index",
            use_gpu=True,
            gpu_device_ids=gpu_ids,
            search_workers=4,
        )
        self.addCleanup(vector_index.close)
        return vector_index

    def test_multiple_configured_gpus_use_sharded_clone(self) -> None:
        vector_index = self._make_index("1,5")
        cpu_index = SimpleNamespace(ntotal=10, d=4)
        gpu_index = SimpleNamespace(ntotal=10, d=4)
        vector_index._index = cpu_index
        clone = Mock(return_value=gpu_index)
        resources = [object(), object()]

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,5"}),
            patch(
                "src.vector_index.faiss.get_num_gpus",
                return_value=2,
            ),
            patch(
                "src.vector_index.faiss.StandardGpuResources",
                side_effect=resources,
            ),
            patch(
                "src.vector_index.faiss.GpuMultipleClonerOptions",
                FakeGpuMultipleClonerOptions,
            ),
            patch(
                "src.vector_index.faiss.index_cpu_to_gpu_multiple_py",
                clone,
            ),
        ):
            vector_index._move_to_gpu()

        clone.assert_called_once()
        call = clone.call_args
        self.assertEqual(call.args, (resources, cpu_index))
        self.assertEqual(call.kwargs["gpus"], [0, 1])
        self.assertTrue(call.kwargs["co"].shard)
        self.assertTrue(call.kwargs["co"].useFloat16)
        self.assertIs(vector_index._index, gpu_index)
        self.assertEqual(vector_index._gpu_resources, resources)

    def test_single_gpu_keeps_single_gpu_clone_path(self) -> None:
        vector_index = self._make_index("3")
        cpu_index = SimpleNamespace(ntotal=10, d=4)
        gpu_index = SimpleNamespace(ntotal=10, d=4)
        vector_index._index = cpu_index
        resource = object()
        clone = Mock(return_value=gpu_index)

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}),
            patch(
                "src.vector_index.faiss.get_num_gpus",
                return_value=1,
            ),
            patch(
                "src.vector_index.faiss.StandardGpuResources",
                return_value=resource,
            ),
            patch(
                "src.vector_index.faiss.GpuClonerOptions",
                FakeGpuClonerOptions,
            ),
            patch("src.vector_index.faiss.index_cpu_to_gpu", clone),
        ):
            vector_index._move_to_gpu()

        clone.assert_called_once()
        call = clone.call_args
        self.assertEqual(call.args[:3], (resource, 0, cpu_index))
        self.assertTrue(call.args[3].useFloat16)
        self.assertIs(vector_index._index, gpu_index)
        self.assertEqual(vector_index._gpu_resources, [resource])

    def test_unavailable_gpu_configuration_falls_back_to_cpu(self) -> None:
        vector_index = self._make_index("1,5")
        cpu_index = SimpleNamespace(ntotal=10, d=4)
        vector_index._index = cpu_index

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,5"}),
            patch(
                "src.vector_index.faiss.get_num_gpus",
                return_value=1,
            ),
            patch.object(vector_index, "_replace_search_executor") as replace,
        ):
            vector_index._move_to_gpu()

        self.assertIs(vector_index._index, cpu_index)
        self.assertFalse(vector_index._use_gpu)
        self.assertIsNone(vector_index._gpu_resources)
        replace.assert_called_once_with(4)

    def test_gpu_device_ids_reject_duplicates(self) -> None:
        vector_index = self._make_index("1,1")

        with self.assertRaisesRegex(ValueError, "Duplicate GPU device IDs"):
            vector_index._parse_gpu_device_ids()


if __name__ == "__main__":
    unittest.main()
