import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.data import make_train_sampler


class TrainSamplerTests(unittest.TestCase):
    def make_dataset(self, size):
        return TensorDataset(torch.arange(size))

    def test_returns_none_without_batches_per_epoch(self):
        self.assertIsNone(make_train_sampler(self.make_dataset(100), None, 16))

    def test_limits_epoch_to_requested_batches(self):
        dataset = self.make_dataset(1000)
        sampler = make_train_sampler(dataset, 10, 16)
        loader = DataLoader(dataset, batch_size=16, sampler=sampler)
        self.assertEqual(len(loader), 10)
        self.assertEqual(sum(batch[0].shape[0] for batch in loader), 160)

    def test_caps_samples_at_dataset_size(self):
        dataset = self.make_dataset(50)
        sampler = make_train_sampler(dataset, 10, 16)
        self.assertEqual(len(sampler), 50)

    def test_first_epoch_yields_sequential_prefix(self):
        sampler = make_train_sampler(self.make_dataset(1000), 10, 16)
        self.assertEqual(list(iter(sampler)), list(range(160)))

    def test_each_epoch_advances_to_the_next_chunk(self):
        sampler = make_train_sampler(self.make_dataset(1000), 10, 16)
        sampler.set_epoch(1)
        self.assertEqual(list(iter(sampler)), list(range(160, 320)))

    def test_chunks_wrap_around_the_dataset_end(self):
        sampler = make_train_sampler(self.make_dataset(100), 4, 10)  # 40-sample chunks
        sampler.set_epoch(2)  # starts at index 80, wraps into 0..19
        self.assertEqual(list(iter(sampler)), list(range(80, 100)) + list(range(0, 20)))

    def test_epochs_partition_the_dataset_before_wrapping(self):
        dataset = self.make_dataset(1000)
        sampler = make_train_sampler(dataset, 10, 16)
        seen = []
        for epoch in range(6):  # 6 epochs x 160 samples = 960 <= 1000, no wrap yet
            sampler.set_epoch(epoch)
            seen.extend(iter(sampler))
        self.assertEqual(seen, list(range(960)))


if __name__ == "__main__":
    unittest.main()
