import sys
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src import faiss_utils
from src.eval import compute_recalls_from_features
from src.targets import build_attack_targets
from src import rank_eval


class ExactFlatL2Index:
    def __init__(self, dimension):
        self.dimension = dimension
        self.database = np.empty((0, dimension), dtype=np.float32)

    def add(self, features):
        self.database = np.asarray(features, dtype=np.float32).copy()

    def search(self, features, k):
        queries = np.asarray(features, dtype=np.float32)
        distances = np.sum((queries[:, None, :] - self.database[None, :, :]) ** 2, axis=2)
        indexes = np.argsort(distances, axis=1)[:, :k]
        return np.take_along_axis(distances, indexes, axis=1), indexes.astype(np.int64)


class FakeGpuResources:
    def __init__(self):
        self.temp_memory_bytes = None
        self.events = []

    def setTempMemory(self, size_bytes):
        self.temp_memory_bytes = size_bytes
        self.events.append("setTempMemory")


class FaissUtilsTests(unittest.TestCase):
    def test_rank_eval_fails_on_faiss_before_checking_model_paths(self):
        args = rank_eval.build_parser().parse_args(
            [
                "--datasets",
                "msls",
                "--model_type",
                "supervlad",
                "--model_paths",
                "missing.pth",
                "--epsilons",
                "0.01",
            ]
        )

        with (
            mock.patch.object(rank_eval, "validate_cuda_runtime"),
            mock.patch.object(
                rank_eval,
                "validate_faiss_runtime",
                side_effect=RuntimeError("FAISS GPU unavailable"),
            ),
            mock.patch.object(rank_eval, "require_file") as require_file,
        ):
            with self.assertRaisesRegex(RuntimeError, "FAISS GPU unavailable"):
                rank_eval.validate_arguments(args)

        require_file.assert_not_called()

    def test_cpu_index_does_not_query_gpu_apis(self):
        cpu_faiss = SimpleNamespace(IndexFlatL2=ExactFlatL2Index)

        with (
            mock.patch.object(faiss_utils, "_import_faiss", return_value=cpu_faiss),
            mock.patch.object(faiss_utils.torch.cuda, "current_device") as current_device,
        ):
            index = faiss_utils.create_flat_l2_index(2, "cpu")

        self.assertFalse(index.uses_gpu)
        current_device.assert_not_called()
        index.add(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
        _, neighbors = index.search(np.array([[1.9, 0.0]], dtype=np.float32), 1)
        np.testing.assert_array_equal(neighbors, np.array([[1]]))

    def test_cuda_rejects_cpu_only_faiss_module(self):
        cpu_faiss = SimpleNamespace(IndexFlatL2=ExactFlatL2Index, get_num_gpus=lambda: 1)

        with mock.patch.object(faiss_utils, "_import_faiss", return_value=cpu_faiss):
            with self.assertRaisesRegex(RuntimeError, "missing StandardGpuResources, index_cpu_to_gpu"):
                faiss_utils.create_flat_l2_index(2, "cuda")

    def test_cuda_rejects_zero_visible_faiss_gpus(self):
        fake_faiss = SimpleNamespace(
            IndexFlatL2=ExactFlatL2Index,
            StandardGpuResources=object,
            index_cpu_to_gpu=lambda *_args: None,
            get_num_gpus=lambda: 0,
        )

        with mock.patch.object(faiss_utils, "_import_faiss", return_value=fake_faiss):
            with self.assertRaisesRegex(RuntimeError, "reported zero CUDA devices"):
                faiss_utils.create_flat_l2_index(2, "cuda")

    def test_cuda_initialization_failure_is_actionable(self):
        def fail_initialization(*_args):
            raise RuntimeError("CUDA allocation failed")

        fake_faiss = SimpleNamespace(
            IndexFlatL2=ExactFlatL2Index,
            StandardGpuResources=FakeGpuResources,
            index_cpu_to_gpu=fail_initialization,
            get_num_gpus=lambda: 1,
        )

        with (
            mock.patch.object(faiss_utils, "_import_faiss", return_value=fake_faiss),
            mock.patch.object(faiss_utils.torch.cuda, "current_device", return_value=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU index initialization failed"):
                faiss_utils.create_flat_l2_index(2, "cuda")

    def test_cuda_index_uses_current_torch_device_and_retains_resources(self):
        resources = FakeGpuResources()
        conversion = {}

        def to_gpu(received_resources, device_index, cpu_index):
            conversion["resources"] = received_resources
            conversion["device_index"] = device_index
            conversion["cpu_index"] = cpu_index
            return ExactFlatL2Index(cpu_index.dimension)

        fake_faiss = SimpleNamespace(
            IndexFlatL2=ExactFlatL2Index,
            StandardGpuResources=lambda: resources,
            index_cpu_to_gpu=to_gpu,
            get_num_gpus=lambda: 2,
        )

        with (
            mock.patch.object(faiss_utils, "_import_faiss", return_value=fake_faiss),
            mock.patch.object(faiss_utils.torch.cuda, "current_device", return_value=1),
        ):
            index = faiss_utils.create_flat_l2_index(3, "cuda")

        self.assertTrue(index.uses_gpu)
        self.assertIs(index.gpu_resources, resources)
        self.assertIs(conversion["resources"], resources)
        self.assertEqual(conversion["device_index"], 1)
        self.assertEqual(conversion["cpu_index"].dimension, 3)

    def test_cuda_index_caps_temporary_memory_before_gpu_conversion(self):
        resources = FakeGpuResources()
        events = []

        def to_gpu(_resources, _device_index, cpu_index):
            events.append("index_cpu_to_gpu")
            return ExactFlatL2Index(cpu_index.dimension)

        fake_faiss = SimpleNamespace(
            IndexFlatL2=ExactFlatL2Index,
            StandardGpuResources=lambda: resources,
            index_cpu_to_gpu=to_gpu,
            get_num_gpus=lambda: 1,
        )
        resources.events = events

        with (
            mock.patch.object(faiss_utils, "_import_faiss", return_value=fake_faiss),
            mock.patch.object(faiss_utils.torch.cuda, "current_device", return_value=0),
        ):
            faiss_utils.create_flat_l2_index(3, "cuda")

        self.assertEqual(resources.temp_memory_bytes, faiss_utils.FAISS_GPU_TEMP_MEMORY_BYTES)
        self.assertEqual(faiss_utils.FAISS_GPU_TEMP_MEMORY_BYTES, 256 * 1024 * 1024)
        self.assertEqual(events, ["setTempMemory", "index_cpu_to_gpu"])

    def test_cuda_startup_probe_reports_search_failure(self):
        class FailingSearchIndex(ExactFlatL2Index):
            def search(self, features, k):
                raise RuntimeError("kernel launch failed")

        fake_faiss = SimpleNamespace(
            IndexFlatL2=ExactFlatL2Index,
            StandardGpuResources=FakeGpuResources,
            index_cpu_to_gpu=lambda _resources, _device, cpu_index: FailingSearchIndex(cpu_index.dimension),
            get_num_gpus=lambda: 1,
        )

        with (
            mock.patch.object(faiss_utils, "_import_faiss", return_value=fake_faiss),
            mock.patch.object(faiss_utils.torch.cuda, "current_device", return_value=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "startup GPU add/search probe failed"):
                faiss_utils.validate_faiss_runtime("cuda")

    def test_cpu_recall_and_target_searches_preserve_expected_neighbors(self):
        database = np.array([[0.0], [1.0], [10.0]], dtype=np.float32)
        queries = np.array([[0.1]], dtype=np.float32)
        positives = [np.array([0], dtype=np.int64)]
        recall_args = Namespace(device="cpu", features_dim=1, recall_values=[1, 2])

        recalls = compute_recalls_from_features(recall_args, database, queries, positives)

        self.assertEqual(recalls["recalls"], {"R@1": 100.0, "R@2": 100.0})

        eval_ds = SimpleNamespace(database_num=3, get_positives=lambda: positives)
        target_args = Namespace(device="cpu", adv_negatives=1)
        targets, valid_queries = build_attack_targets(target_args, eval_ds, database, queries)

        np.testing.assert_array_equal(valid_queries, np.array([0]))
        self.assertEqual(targets[0]["positive_index"], 0)
        np.testing.assert_array_equal(targets[0]["negative_indexes"], np.array([1]))


if __name__ == "__main__":
    unittest.main()
