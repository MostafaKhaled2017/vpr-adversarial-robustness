import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src import supervlad_compat
from src.cli import parse_arguments
from src.data import PipeDataLoader, loader_class


class LoaderClassTests(unittest.TestCase):
    def test_default_flag_is_shm(self):
        args = parse_arguments(["--eval_datasets_folder", "/tmp", "--device", "cpu"])
        self.assertFalse(args.pipe_loader)
        self.assertIs(loader_class(args), DataLoader)

    def test_flag_selects_pipe(self):
        args = parse_arguments(["--eval_datasets_folder", "/tmp", "--device", "cpu", "--pipe_loader"])
        self.assertIs(loader_class(args), PipeDataLoader)

    def test_missing_attribute_defaults_to_shm(self):
        self.assertIs(loader_class(SimpleNamespace()), DataLoader)

    def test_pipe_and_shm_loaders_yield_identical_batches(self):
        dataset = TensorDataset(torch.rand(12, 3, 8, 8), torch.arange(12))
        shm = list(DataLoader(dataset, batch_size=4, num_workers=2))
        pipe = list(PipeDataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True))
        self.assertEqual(len(shm), len(pipe))
        for (shm_x, shm_y), (pipe_x, pipe_y) in zip(shm, pipe):
            self.assertEqual(shm_x.dtype, pipe_x.dtype)
            self.assertEqual(shm_y.dtype, pipe_y.dtype)
            self.assertTrue(torch.equal(shm_x, pipe_x))
            self.assertTrue(torch.equal(shm_y, pipe_y))


class SupervladTestRoutingTests(unittest.TestCase):
    def loader_seen_by_upstream(self, args):
        import test as test_module

        seen = {}

        def fake_test(*_args, **_kwargs):
            seen["DataLoader"] = test_module.DataLoader
            return [0.0], ""

        with mock.patch.object(test_module, "test", side_effect=fake_test):
            supervlad_compat.test(args, None, None, test_method="hard_resize")
        return seen["DataLoader"]

    def test_default_leaves_upstream_loader_alone(self):
        self.assertIs(self.loader_seen_by_upstream(SimpleNamespace(pipe_loader=False)), DataLoader)

    def test_flag_patches_upstream_loader(self):
        self.assertIs(self.loader_seen_by_upstream(SimpleNamespace(pipe_loader=True)), PipeDataLoader)


if __name__ == "__main__":
    unittest.main()
