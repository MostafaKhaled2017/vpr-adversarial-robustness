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


class ShuffledChunkSamplerTests(unittest.TestCase):
    def make_dataset(self, size):
        return TensorDataset(torch.arange(size))

    def collect_epochs(self, sampler, epochs):
        indices = []
        for epoch in epochs:
            sampler.set_epoch(epoch)
            indices.extend(iter(sampler))
        return indices

    def test_no_seed_keeps_sequential_behavior(self):
        sampler = make_train_sampler(self.make_dataset(1000), 10, 16, shuffle_seed=None)
        self.assertEqual(list(iter(sampler)), list(range(160)))

    def test_shuffled_epoch_is_not_sequential(self):
        sampler = make_train_sampler(self.make_dataset(1000), 10, 16, shuffle_seed=0)
        self.assertNotEqual(list(iter(sampler)), list(range(160)))

    def test_shuffled_pass_covers_every_index_once(self):
        # 5 epochs x 20 samples = one full pass over the 100-sample dataset.
        sampler = make_train_sampler(self.make_dataset(100), 2, 10, shuffle_seed=0)
        seen = self.collect_epochs(sampler, range(5))
        self.assertEqual(sorted(seen), list(range(100)))

    def test_same_seed_reproduces_ordering(self):
        first = make_train_sampler(self.make_dataset(100), 2, 10, shuffle_seed=3)
        second = make_train_sampler(self.make_dataset(100), 2, 10, shuffle_seed=3)
        self.assertEqual(self.collect_epochs(first, range(7)), self.collect_epochs(second, range(7)))

    def test_different_seeds_give_different_orderings(self):
        first = make_train_sampler(self.make_dataset(1000), 10, 16, shuffle_seed=0)
        second = make_train_sampler(self.make_dataset(1000), 10, 16, shuffle_seed=1)
        self.assertNotEqual(list(iter(first)), list(iter(second)))

    def test_each_pass_uses_a_fresh_permutation(self):
        sampler = make_train_sampler(self.make_dataset(100), 2, 10, shuffle_seed=0)
        first_pass = self.collect_epochs(sampler, range(5))
        second_pass = self.collect_epochs(sampler, range(5, 10))
        self.assertEqual(sorted(second_pass), list(range(100)))
        self.assertNotEqual(first_pass, second_pass)

    def test_epoch_straddling_a_pass_boundary_completes_both_passes(self):
        # 40-sample epochs over a 100-sample dataset: epoch 2 spans positions
        # 80..119, i.e. the tail of pass 0 and the head of pass 1.
        sampler = make_train_sampler(self.make_dataset(100), 4, 10, shuffle_seed=0)
        epochs = []
        for epoch in range(5):
            sampler.set_epoch(epoch)
            epochs.append(list(iter(sampler)))
        pass_zero = epochs[0] + epochs[1] + epochs[2][:20]
        pass_one = epochs[2][20:] + epochs[3] + epochs[4]
        self.assertEqual(sorted(pass_zero), list(range(100)))
        self.assertEqual(sorted(pass_one), list(range(100)))

    def test_shuffle_without_batches_per_epoch_covers_full_dataset(self):
        sampler = make_train_sampler(self.make_dataset(100), None, 16, shuffle_seed=0)
        self.assertIsNotNone(sampler)
        self.assertEqual(len(sampler), 100)
        self.assertEqual(sorted(iter(sampler)), list(range(100)))
        self.assertNotEqual(list(iter(sampler)), list(range(100)))


if __name__ == "__main__":
    unittest.main()
