import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments
from src.rank_validation import evaluate_rank_validation, sample_validation_queries, select_checkpoint
from src.retrieval_metrics import compute_recalls_from_features
from src.train_loop import build_checkpoint_state, build_validation_metrics_record


def metrics(clean, *attacked):
    result = {"NoAttack": {"recalls": {"R@1": clean, "R@5": clean}}}
    for index, value in enumerate(attacked):
        result[f"rank_pgd_linf_eps_{index}"] = {"recalls": {"R@1": value, "R@5": value}}
    return result


class SelectCheckpointTests(unittest.TestCase):
    def test_eligible_epoch_scores_mean_attacked_r1(self):
        scores = select_checkpoint(metrics(90.0, 20.0, 10.0), initial_clean_r1=90.5, max_clean_drop=1.0, is_clean_only=False)
        self.assertTrue(scores["eligible"])
        self.assertEqual(scores["robust_score"], 15.0)
        self.assertEqual(scores["selection_score"], 15.0)

    def test_clean_drop_beyond_limit_is_ineligible(self):
        scores = select_checkpoint(metrics(88.0, 40.0, 30.0), initial_clean_r1=90.0, max_clean_drop=1.0, is_clean_only=False)
        self.assertFalse(scores["eligible"])
        self.assertEqual(scores["selection_score"], -math.inf)

    def test_drop_exactly_at_limit_is_eligible(self):
        self.assertTrue(select_checkpoint(metrics(89.0, 1.0), 90.0, 1.0, False)["eligible"])

    def test_initial_validation_is_always_eligible(self):
        self.assertTrue(select_checkpoint(metrics(50.0, 1.0), None, 1.0, False)["eligible"])

    def test_clean_only_selects_on_clean_r1(self):
        scores = select_checkpoint(metrics(91.0), 90.0, 1.0, True)
        self.assertEqual(scores["selection_score"], 91.0)
        self.assertEqual(scores["robust_score"], 91.0)


class SampleQueriesTests(unittest.TestCase):
    def test_sample_is_deterministic_sorted_and_has_positives(self):
        positives = [np.array([1]) if index % 3 else np.array([], dtype=int) for index in range(100)]
        first = sample_validation_queries(positives, 10, seed=0)
        self.assertTrue(np.array_equal(first, sample_validation_queries(positives, 10, seed=0)))
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertTrue(all(len(positives[index]) > 0 for index in first))
        self.assertEqual(len(first), 10)

    def test_small_pool_returns_every_valid_query(self):
        positives = [np.array([1]), np.array([], dtype=int), np.array([2])]
        self.assertEqual(sample_validation_queries(positives, 10, seed=0).tolist(), [0, 2])


BASE = [
    "--eval_datasets_folder", "/tmp", "--device", "cpu",
    "--model=supervlad", "--attack", "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
]


class RankValidationCliTests(unittest.TestCase):
    def test_defaults_keep_legacy_protocol(self):
        args = parse_arguments(BASE)
        self.assertEqual(args.validation_protocol, "legacy")
        self.assertEqual(args.checkpoint_selection_rule, "robust_weighted")
        self.assertEqual(args.val_queries, 2000)
        self.assertEqual(args.val_query_seed, 0)
        self.assertEqual(args.val_rank_steps, 10)
        self.assertEqual(args.selection_max_clean_drop, 1.0)

    def test_rank_pgd_sets_selection_rule(self):
        args = parse_arguments(BASE + ["--validation_protocol", "rank_pgd"])
        self.assertEqual(args.checkpoint_selection_rule, "clean_constrained_rank_pgd")
        self.assertEqual(args.val_rank_epsilons, [0.01, 0.1])

    def test_rank_pgd_clean_only_selects_on_clean_recall(self):
        args = parse_arguments(BASE[:5] + ["--validation_protocol", "rank_pgd"])
        self.assertEqual(args.checkpoint_selection_rule, "clean_recall")

    def test_rank_pgd_rejects_skipping_initial_validation(self):
        with self.assertRaisesRegex(ValueError, "requires the initial validation"):
            parse_arguments(BASE + ["--validation_protocol", "rank_pgd", "--skip_initial_validation"])

    def test_rejects_invalid_values(self):
        for extra in (
            ["--val_queries", "0"],
            ["--val_rank_steps", "0"],
            ["--val_rank_epsilons", "0.01", "0"],
            ["--selection_max_clean_drop", "-0.5"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                parse_arguments(BASE + ["--validation_protocol", "rank_pgd", *extra])


class FakeValDataset:
    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.database_num = 6
        self.queries_num = 4
        self.images = torch.rand(10, 3, 4, 4, generator=generator)
        self.test_method = None

    def get_positives(self):
        return [np.array([0, 1]), np.array([2]), np.array([], dtype=np.int64), np.array([3, 4])]

    def __getitem__(self, index):
        return self.images[index], index

    def __len__(self):
        return self.database_num + self.queries_num


class LinearDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(48, 2)

    def forward(self, inputs, queryflag=0):
        return self.linear(inputs.flatten(1))


def rank_args(**overrides):
    values = dict(
        device="cpu", infer_batch_size=2, num_workers=0, features_dim=2, adv_negatives=2, adv_margin=0.1,
        val_rank_steps=2, val_rank_epsilons=[0.01, 0.1], val_query_seed=0, recall_values=[1, 5, 10, 100],
        is_clean_only=False, test_method="hard_resize", mixed_precision=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class EvaluateRankValidationTests(unittest.TestCase):
    def test_adversarial_run_reports_clean_and_every_epsilon(self):
        model = LinearDescriptor()
        result = evaluate_rank_validation(rank_args(), model, FakeValDataset(), np.array([0, 1, 3]))
        self.assertEqual(list(result), ["NoAttack", "rank_pgd_linf_eps_0.01", "rank_pgd_linf_eps_0.1"])
        for value in result.values():
            self.assertTrue(0.0 <= value["recalls"]["R@1"] <= 100.0)
        self.assertTrue(model.training)

    def test_clean_recall_is_computed_on_the_sampled_queries_only(self):
        dataset = FakeValDataset()
        model = LinearDescriptor()
        sampled = np.array([0, 3])
        with torch.no_grad():
            database = model(dataset.images[:6]).numpy()
            queries = model(dataset.images[6 + sampled]).numpy()
        expected = compute_recalls_from_features(
            database, queries, [dataset.get_positives()[index] for index in sampled], [1, 5, 10, 100]
        )
        result = evaluate_rank_validation(rank_args(is_clean_only=True), model, dataset, sampled)
        self.assertEqual(result["NoAttack"]["recalls"], expected["recalls"])

    def test_clean_only_run_skips_attacks(self):
        result = evaluate_rank_validation(rank_args(is_clean_only=True), LinearDescriptor(), FakeValDataset(), np.array([0, 1, 3]))
        self.assertEqual(list(result), ["NoAttack"])


class CheckpointRuntimeStateTests(unittest.TestCase):
    def test_initial_clean_r1_is_stored_in_runtime_state(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        clean = {"recalls_list": [1.0, 2.0, 3.0, 4.0], "recalls": {"R@1": 1.0, "R@5": 2.0, "R@10": 3.0, "R@100": 4.0}}
        checkpoint = build_checkpoint_state(
            Namespace(tensorboard_dir="tensorboard"), model, optimizer, epoch_num=0, metrics={"NoAttack": clean},
            validation_scores={"clean_score": 1.0, "robust_score": 1.0, "selection_score": 1.0, "eligible": True},
            next_best_score=1.0, next_not_improved=0, initial_clean_r1=90.0,
        )
        self.assertEqual(checkpoint["runtime_state"]["initial_clean_r1"], 90.0)

    def test_record_adds_eligibility_only_when_present(self):
        clean = {"recalls": {"R@1": 1.0, "R@5": 2.0, "R@10": 3.0, "R@100": 4.0}}
        legacy = build_validation_metrics_record(0, {"NoAttack": clean}, {"clean_score": 1.0, "robust_score": 1.0, "selection_score": 1.0})
        self.assertNotIn("eligible", legacy)
        self.assertNotIn("initial_clean_r1", legacy)
        rank = build_validation_metrics_record(
            1, {"NoAttack": clean}, {"clean_score": 1.0, "robust_score": 1.0, "selection_score": -math.inf, "eligible": False},
            initial_clean_r1=90.0,
        )
        self.assertFalse(rank["eligible"])
        self.assertEqual(rank["initial_clean_r1"], 90.0)


def scripted_metrics(clean, attacked):
    def recalls(value):
        return {"recalls": {f"R@{k}": value for k in (1, 5, 10, 100)}, "recalls_list": [value] * 4}

    return {"NoAttack": recalls(clean), "rank_pgd_linf_eps_0.01": recalls(attacked)}


class RunTrainingRankProtocolTests(unittest.TestCase):
    def test_ineligible_epoch_is_never_best_and_initial_clean_r1_is_kept(self):
        import tempfile
        from unittest import mock

        from src import train_loop

        # Initial C0=90; epoch 1 is more robust but drops clean R@1 by 3 (ineligible);
        # epoch 2 stays within the limit and becomes best.
        results = iter([scripted_metrics(90.0, 10.0), scripted_metrics(87.0, 50.0), scripted_metrics(89.5, 20.0)])
        saved = []
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            train_loop, "evaluate_rank_validation", side_effect=lambda *a, **k: next(results)
        ), mock.patch.object(
            train_loop, "save_checkpoint", side_effect=lambda args, state, is_best, filename: saved.append((filename, is_best, state))
        ), mock.patch.object(train_loop, "maybe_remove_old_checkpoint"), mock.patch.object(
            train_loop.util, "load_trusted_checkpoint", return_value={"model_state_dict": model.state_dict()}
        ), mock.patch.object(train_loop.test, "test", return_value=([0.0] * 4, "")):
            args = SimpleNamespace(
                validation_protocol="rank_pgd", val_queries=10, val_query_seed=0, selection_max_clean_drop=1.0,
                is_clean_only=False, skip_initial_validation=False, lr_schedule="", lr_plateau_patience=None, lr=1e-4,
                epochs_num=2, adv_warmup_epochs=0, randomize_attack=False, early_stop_min_delta=0.0, patience=5,
                save_dir=save_dir, tensorboard_dir=save_dir, device="cpu", test_method="hard_resize",
                recall_values=[1, 5, 10, 100], mixed_precision=False,
            )
            outcome = train_loop.run_training(
                args, model, torch.optim.Adam(model.parameters(), lr=1e-4), None, [], FakeValDataset(), None,
                -math.inf, 0, 0, mock.MagicMock(), [], [],
            )
        self.assertEqual(outcome["state"], "completed")
        last = [(is_best, state) for filename, is_best, state in saved if filename == "last_model.pth"]
        self.assertEqual([is_best for is_best, _ in last], [False, True])
        self.assertEqual(last[0][1]["not_improved_num"], 1)
        self.assertEqual(last[1][1]["best_r5"], 20.0)
        for _, _, state in saved:
            self.assertEqual(state["runtime_state"]["initial_clean_r1"], 90.0)


if __name__ == "__main__":
    unittest.main()
