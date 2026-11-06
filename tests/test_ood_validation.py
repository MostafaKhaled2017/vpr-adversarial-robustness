"""Clean recall on out-of-domain sets after each validation."""

import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments  # noqa: E402
from src.rank_validation import evaluate_ood_clean  # noqa: E402
from src.train_loop import build_validation_metrics_record  # noqa: E402

SIZE = 6


class OneHotSet(torch.utils.data.Dataset):
    """SIZE database images and SIZE queries; image i encodes place i % SIZE as a one-hot vector."""

    database_num = SIZE
    queries_num = SIZE

    def __init__(self, positives):
        self.positives = positives

    def __len__(self):
        return 2 * SIZE

    def __getitem__(self, index):
        return torch.eye(SIZE)[index % SIZE], index

    def get_positives(self):
        return self.positives


class Identity(nn.Module):
    def forward(self, x, queryflag=0):
        return x


def metric(r1):
    return {"recalls": {"R@1": r1, "R@5": r1, "R@10": r1, "R@100": r1}}


class OodValidationTest(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(infer_batch_size=4, num_workers=0, device="cpu", recall_values=[1, 5])

    def test_recall_on_each_set(self):
        own_place = OneHotSet([np.array([q]) for q in range(SIZE)])
        other_place = OneHotSet([np.array([(q + 1) % SIZE]) for q in range(SIZE)])
        queries = np.array([0, 2, 5])
        model = Identity().train()
        results = evaluate_ood_clean(self.args, model, {"own": (own_place, queries), "other": (other_place, queries)})
        self.assertEqual(results, {"own": 100.0, "other": 0.0})
        self.assertTrue(model.training)

    def test_record_field_only_when_given(self):
        metrics = {"NoAttack": metric(80.0), "rank_pgd_linf_eps_0.0685": metric(40.0)}
        scores = {"clean_score": 80.0, "robust_score": 40.0, "selection_score": 40.0}
        self.assertNotIn("ood_clean", build_validation_metrics_record(3, metrics, scores))
        record = build_validation_metrics_record(3, metrics, scores, ood_clean={"sped": 70.0, "nordland": 30.0})
        self.assertEqual(record["ood_clean"], {"sped": 70.0, "nordland": 30.0})

    def test_parser_default_is_off(self):
        base = ["--eval_datasets_folder", "/tmp", "--device", "cpu", "--model=supervlad"]
        args = parse_arguments(base)
        self.assertEqual(args.ood_val_datasets, [])
        self.assertEqual(args.ood_val_queries, 500)
        self.assertEqual(parse_arguments(base + ["--ood_val_datasets", "sped", "nordland"]).ood_val_datasets,
                         ["sped", "nordland"])
        with self.assertRaisesRegex(ValueError, "--ood_val_queries"):
            parse_arguments(base + ["--ood_val_queries=0"])


if __name__ == "__main__":
    unittest.main()
