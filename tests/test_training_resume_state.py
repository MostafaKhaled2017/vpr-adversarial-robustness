import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.negative_pool import NegativePool
from src.train_loop import build_checkpoint_state


class FakeScaler:
    def state_dict(self):
        return {"scale": 256.0}


class TrainingResumeStateTests(unittest.TestCase):
    def test_checkpoint_contains_every_state_needed_at_the_next_epoch(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        pool = NegativePool(capacity=4, descriptor_dim=2)
        pool.push(torch.ones(2, 2), torch.tensor([3, 4]))
        reference = torch.tensor([[0.1, 0.2]])
        metrics = {
            "NoAttack": {
                "recalls_list": [1.0, 2.0, 3.0, 4.0],
                "recalls": {"R@1": 1.0, "R@5": 2.0, "R@10": 3.0, "R@100": 4.0},
            }
        }

        checkpoint = build_checkpoint_state(
            Namespace(tensorboard_dir="tensorboard"),
            model,
            optimizer,
            epoch_num=4,
            metrics=metrics,
            validation_scores={"clean_score": 1.0, "robust_score": 2.0, "selection_score": 3.0},
            next_best_score=3.0,
            next_not_improved=2,
            scaler=FakeScaler(),
            iteration=321,
            negative_pool=pool,
            collapse_sample_indices=[7],
            reference_descriptors=reference,
        )

        self.assertEqual(checkpoint["next_epoch"], 5)
        self.assertEqual(checkpoint["scaler_state_dict"], {"scale": 256.0})
        self.assertIn("rng_state", checkpoint)
        runtime = checkpoint["runtime_state"]
        self.assertEqual(runtime["iteration"], 321)
        self.assertEqual(runtime["collapse_sample_indices"], [7])
        self.assertTrue(torch.equal(runtime["reference_descriptors"], reference))
        self.assertEqual(runtime["negative_pool"]["size"], 2)


if __name__ == "__main__":
    unittest.main()
