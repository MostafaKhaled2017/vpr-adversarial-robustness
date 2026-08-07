import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.negative_pool import NegativePool


def descriptors(*values):
    return torch.tensor([[value, 0.0] for value in values])


class NegativePoolStorageTests(unittest.TestCase):
    def test_push_accumulates_until_capacity(self):
        pool = NegativePool(capacity=4, descriptor_dim=2)

        pool.push(descriptors(1.0, 2.0), torch.tensor([10, 11]))

        self.assertEqual(len(pool), 2)

    def test_fifo_eviction_drops_the_oldest_entries(self):
        pool = NegativePool(capacity=3, descriptor_dim=2)
        pool.push(descriptors(1.0, 2.0, 3.0), torch.tensor([10, 11, 12]))

        pool.push(descriptors(4.0, 5.0), torch.tensor([13, 14]))

        self.assertEqual(len(pool), 3)
        stored_place_ids = sorted(int(value) for value in pool.place_ids().tolist())
        self.assertEqual(stored_place_ids, [12, 13, 14])

    def test_pushed_descriptors_are_detached_from_the_graph(self):
        pool = NegativePool(capacity=2, descriptor_dim=2)
        tracked = descriptors(1.0, 2.0).requires_grad_(True)

        pool.push(tracked, torch.tensor([10, 11]))

        self.assertFalse(pool.descriptors().requires_grad)

    def test_pushing_more_than_capacity_at_once_keeps_the_newest(self):
        pool = NegativePool(capacity=2, descriptor_dim=2)

        pool.push(descriptors(1.0, 2.0, 3.0, 4.0), torch.tensor([10, 11, 12, 13]))

        self.assertEqual(len(pool), 2)
        self.assertEqual(sorted(int(v) for v in pool.place_ids().tolist()), [12, 13])

    def test_zero_capacity_pool_stays_empty(self):
        pool = NegativePool(capacity=0, descriptor_dim=2)

        pool.push(descriptors(1.0, 2.0), torch.tensor([10, 11]))

        self.assertEqual(len(pool), 0)
        self.assertFalse(pool.enabled)


class NegativePoolMiningTests(unittest.TestCase):
    def test_mine_returns_the_k_nearest_cross_place_negatives(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        pool.push(descriptors(1.0, 2.0, 3.0, 10.0), torch.tensor([11, 12, 13, 14]))

        mined = pool.mine(descriptors(0.0), torch.tensor([99]), k=2)

        self.assertEqual(mined.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(mined[0], descriptors(1.0, 2.0)))

    def test_mine_excludes_entries_from_the_query_place(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        # The two nearest entries share the query's place and must never be mined.
        pool.push(descriptors(1.0, 2.0, 5.0, 6.0), torch.tensor([99, 99, 13, 14]))

        mined = pool.mine(descriptors(0.0), torch.tensor([99]), k=2)

        self.assertTrue(torch.allclose(mined[0], descriptors(5.0, 6.0)))

    def test_mine_draws_from_the_pool_and_the_current_batch(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        pool.push(descriptors(9.0), torch.tensor([13]))

        mined = pool.mine(
            descriptors(0.0),
            torch.tensor([99]),
            k=2,
            batch_descriptors=descriptors(1.0, 20.0),
            batch_place_ids=torch.tensor([21, 22]),
        )

        # Nearest overall is the batch entry at 1.0, then the pooled entry at 9.0.
        self.assertTrue(torch.allclose(mined[0], descriptors(1.0, 9.0)))

    def test_mine_caps_k_at_the_number_of_available_candidates(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        pool.push(descriptors(1.0), torch.tensor([13]))

        mined = pool.mine(descriptors(0.0), torch.tensor([99]), k=5)

        self.assertEqual(mined.shape, (1, 1, 2))

    def test_mine_uses_one_shared_k_across_the_batch(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        pool.push(descriptors(1.0, 2.0), torch.tensor([11, 12]))

        # Query 0 sees both entries; query 1 shares a place with one of them, so only one
        # candidate is available to it. A stacked tensor needs a single k.
        mined = pool.mine(descriptors(0.0, 0.0), torch.tensor([99, 11]), k=2)

        self.assertEqual(mined.shape, (2, 1, 2))

    def test_mine_returns_none_when_every_candidate_shares_the_query_place(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        pool.push(descriptors(1.0, 2.0), torch.tensor([99, 99]))

        self.assertIsNone(pool.mine(descriptors(0.0), torch.tensor([99]), k=2))

    def test_mine_returns_none_when_nothing_has_been_pushed(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)

        self.assertIsNone(pool.mine(descriptors(0.0), torch.tensor([99]), k=2))

    def test_mined_negatives_carry_no_gradient(self):
        pool = NegativePool(capacity=8, descriptor_dim=2)
        pool.push(descriptors(1.0, 2.0).requires_grad_(True), torch.tensor([11, 12]))

        mined = pool.mine(descriptors(0.0), torch.tensor([99]), k=1)

        self.assertFalse(mined.requires_grad)


if __name__ == "__main__":
    unittest.main()
