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
from src.train_loop import build_checkpoint_state, build_validation_metrics_record, is_validation_epoch


def metrics(clean, *attacked):
    result = {"NoAttack": {"recalls": {"R@1": clean, "R@5": clean}}}
    for index, value in enumerate(attacked):
        result[f"rank_pgd_linf_eps_{index}"] = {"recalls": {"R@1": value, "R@5": value}}
    return result


class SelectCheckpointTests(unittest.TestCase):
    def test_adversarial_run_selects_on_mean_attacked_r1(self):
        scores = select_checkpoint(metrics(80.0, 20.0, 10.0), initial_clean_r1=90.0, clean_budgets=[1.0], is_clean_only=False)
        self.assertEqual(scores["robust_score"], 15.0)
        self.assertEqual(scores["selection_score"], 15.0)

    def test_eligible_budgets_follow_clean_drop(self):
        scores = select_checkpoint(metrics(87.0, 40.0), 90.0, [1.0, 3.0, 5.0], False)
        self.assertEqual(scores["eligible_budgets"], [3.0, 5.0])

    def test_drop_exactly_at_budget_is_eligible(self):
        self.assertEqual(select_checkpoint(metrics(89.0, 1.0), 90.0, [1.0], False)["eligible_budgets"], [1.0])

    def test_initial_validation_is_eligible_for_every_budget(self):
        self.assertEqual(select_checkpoint(metrics(50.0, 1.0), None, [1.0, 3.0], False)["eligible_budgets"], [1.0, 3.0])

    def test_clean_only_selects_on_clean_r1(self):
        scores = select_checkpoint(metrics(91.0), 90.0, [1.0], True)
        self.assertEqual(scores["selection_score"], 91.0)
        self.assertEqual(scores["robust_score"], 91.0)
        self.assertEqual(scores["eligible_budgets"], [1.0])

    def test_clean_only_budgets_follow_clean_drop(self):
        self.assertEqual(select_checkpoint(metrics(88.0), 90.0, [1.0, 3.0], True)["eligible_budgets"], [3.0])
        self.assertEqual(select_checkpoint(metrics(50.0), None, [1.0, 3.0], True)["eligible_budgets"], [1.0, 3.0])


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


class ImprovedBudgetsTests(unittest.TestCase):
    def test_only_eligible_strict_improvements_are_returned(self):
        from src import train_loop

        best = {"1": 10.0, "3": 50.0}
        scores = {"robust_score": 20.0, "eligible_budgets": [1.0, 3.0, 5.0]}
        self.assertEqual(train_loop.improved_budgets(scores, best), ["1", "5"])

    def test_legacy_scores_without_budgets_improve_nothing(self):
        from src import train_loop

        self.assertEqual(train_loop.improved_budgets({"robust_score": 99.0}, {}), [])


BASE = [
    "--eval_datasets_folder", "/tmp", "--device", "cpu",
    "--model=supervlad", "--attack", "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
]


class RankValidationCliTests(unittest.TestCase):
    def test_defaults_use_rank_pgd_protocol(self):
        args = parse_arguments(BASE)
        self.assertEqual(args.validation_protocol, "rank_pgd")
        self.assertEqual(args.checkpoint_selection_rule, "robust_rank_pgd")
        self.assertEqual(args.val_queries, 2000)
        self.assertEqual(args.val_query_seed, 0)
        self.assertEqual(args.val_rank_steps, 10)
        self.assertEqual(args.selection_clean_budgets, [1.0, 3.0, 5.0])

    def test_legacy_protocol_keeps_weighted_selection(self):
        args = parse_arguments(BASE + ["--validation_protocol", "legacy"])
        self.assertEqual(args.checkpoint_selection_rule, "robust_weighted")

    def test_rank_pgd_sets_selection_rule(self):
        args = parse_arguments(BASE + ["--validation_protocol", "rank_pgd"])
        self.assertEqual(args.checkpoint_selection_rule, "robust_rank_pgd")
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
            ["--val_every", "0"],
            ["--val_rank_epsilons", "0.01", "0"],
            ["--selection_clean_budgets", "1", "-0.5"],
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
    def test_database_features_use_queryflag_one(self):
        """Database extraction in evaluate_rank_validation must use queryflag=1."""
        import src.rank_validation as rank_val_module

        recorded_calls = []

        def fake_extract(args, dataset, model, indices, queryflag):
            recorded_calls.append({"queryflag": queryflag, "indices_len": len(list(indices))})
            return np.zeros((len(list(indices)), 2), dtype=np.float32)

        original_extract = rank_val_module._extract
        rank_val_module._extract = fake_extract
        try:
            dataset = FakeValDataset()
            model = LinearDescriptor()
            args = rank_args()
            evaluate_rank_validation(args, model, dataset, np.array([0, 1, 3]))

            # Should have called _extract for database with queryflag=1
            database_calls = [c for c in recorded_calls if c["indices_len"] == dataset.database_num]
            self.assertTrue(len(database_calls) > 0, "No database extraction calls found")
            for call in database_calls:
                self.assertEqual(call["queryflag"], 1, f"Database extraction should use queryflag=1, got {call['queryflag']}")
        finally:
            rank_val_module._extract = original_extract

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
    def test_budget_scores_are_stored_in_runtime_state(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        clean = {"recalls_list": [1.0, 2.0, 3.0, 4.0], "recalls": {"R@1": 1.0, "R@5": 2.0, "R@10": 3.0, "R@100": 4.0}}
        checkpoint = build_checkpoint_state(
            Namespace(tensorboard_dir="tensorboard"), model, optimizer, epoch_num=0, metrics={"NoAttack": clean},
            validation_scores={"clean_score": 1.0, "robust_score": 1.0, "selection_score": 1.0, "eligible_budgets": [1.0]},
            next_best_score=1.0, next_not_improved=0, initial_clean_r1=90.0, best_budget_scores={"1": 12.0},
            best_checkpoint_epochs={"best": -1, "1": 0},
        )
        self.assertEqual(checkpoint["runtime_state"]["initial_clean_r1"], 90.0)
        self.assertEqual(checkpoint["runtime_state"]["best_budget_scores"], {"1": 12.0})
        self.assertEqual(checkpoint["runtime_state"]["best_checkpoint_epochs"], {"best": -1, "1": 0})

    def test_unvalidated_epoch_checkpoint_keeps_resume_state(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = build_checkpoint_state(
            Namespace(tensorboard_dir="tensorboard"), model, optimizer, epoch_num=2, metrics=None,
            validation_scores=None, next_best_score=7.0, next_not_improved=1, initial_clean_r1=90.0,
            best_budget_scores={"5": 12.0},
        )
        self.assertEqual((checkpoint["next_epoch"], checkpoint["best_r5"], checkpoint["not_improved_num"]), (3, 7.0, 1))
        self.assertIsNone(checkpoint["validation_metrics"])
        self.assertEqual(checkpoint["runtime_state"]["best_budget_scores"], {"5": 12.0})

    def test_val_every_validates_every_nth_and_the_last_epoch(self):
        self.assertEqual([e for e in range(1, 11) if is_validation_epoch(e, 2, 10)], [2, 4, 6, 8, 10])
        self.assertEqual([e for e in range(1, 10) if is_validation_epoch(e, 2, 9)], [2, 4, 6, 8, 9])
        self.assertEqual([e for e in range(1, 4) if is_validation_epoch(e, 1, 3)], [1, 2, 3])

    def test_record_adds_budgets_only_when_present(self):
        clean = {"recalls": {"R@1": 1.0, "R@5": 2.0, "R@10": 3.0, "R@100": 4.0}}
        legacy = build_validation_metrics_record(0, {"NoAttack": clean}, {"clean_score": 1.0, "robust_score": 1.0, "selection_score": 1.0})
        self.assertNotIn("eligible_budgets", legacy)
        self.assertNotIn("initial_clean_r1", legacy)
        rank = build_validation_metrics_record(
            1, {"NoAttack": clean}, {"clean_score": 1.0, "robust_score": 1.0, "selection_score": 1.0, "eligible_budgets": [3.0]},
            initial_clean_r1=90.0,
        )
        self.assertEqual(rank["eligible_budgets"], [3.0])
        self.assertEqual(rank["initial_clean_r1"], 90.0)


def scripted_metrics(clean, attacked):
    def recalls(value):
        return {"recalls": {f"R@{k}": value for k in (1, 5, 10, 100)}, "recalls_list": [value] * 4}

    return {"NoAttack": recalls(clean), "rank_pgd_linf_eps_0.01": recalls(attacked)}


class RunTrainingRankProtocolTests(unittest.TestCase):
    def run_scripted(
        self, scripted, is_clean_only=False, start_epoch=0, resume_runtime_state=None, keep_every=0, val_every=1, epochs_num=None,
        attack_ramp_epochs=0,
    ):
        import tempfile
        from unittest import mock

        from src import train_loop

        results = iter(scripted)
        saved, copied = [], []
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            train_loop, "evaluate_rank_validation", side_effect=lambda *a, **k: next(results)
        ), mock.patch.object(
            train_loop, "save_checkpoint", side_effect=lambda args, state, is_best, filename: saved.append((filename, is_best, state))
        ), mock.patch.object(
            train_loop, "copy_budget_checkpoints", side_effect=lambda args, source, keys: copied.append((source, list(keys)))
        ), mock.patch.object(train_loop, "maybe_remove_old_checkpoint"), mock.patch.object(
            train_loop.util, "load_trusted_checkpoint", return_value={"model_state_dict": model.state_dict()}
        ), mock.patch.object(train_loop.test, "test", return_value=([0.0] * 4, "")):
            args = SimpleNamespace(
                validation_protocol="rank_pgd", val_queries=10, val_query_seed=0, selection_clean_budgets=[1.0, 3.0, 5.0],
                is_clean_only=is_clean_only, skip_initial_validation=False, lr_schedule="", lr_plateau_patience=None, lr=1e-4,
                epochs_num=epochs_num or start_epoch + len(scripted) - (0 if resume_runtime_state else 1), adv_warmup_epochs=0, randomize_attack=False, early_stop_min_delta=0.0, patience=5,
                save_dir=save_dir, tensorboard_dir=save_dir, device="cpu", test_method="hard_resize",
                recall_values=[1, 5, 10, 100], mixed_precision=False, keep_every=keep_every, val_every=val_every,
                attack_ramp_epochs=attack_ramp_epochs,
            )
            outcome = train_loop.run_training(
                args, model, torch.optim.Adam(model.parameters(), lr=1e-4), None, [], FakeValDataset(), None,
                -math.inf, start_epoch, 0, mock.MagicMock(), [], [], resume_runtime_state=resume_runtime_state,
            )
        return outcome, saved, copied

    def test_no_periodic_checkpoints_by_default(self):
        _, saved, _ = self.run_scripted([scripted_metrics(90.0, 10.0), scripted_metrics(87.0, 50.0), scripted_metrics(89.5, 20.0)])
        self.assertFalse([filename for filename, _, _ in saved if filename.startswith("checkpoint_epoch_")])

    def test_keep_every_saves_periodic_checkpoints(self):
        _, saved, _ = self.run_scripted(
            [scripted_metrics(90.0, 10.0), scripted_metrics(87.0, 50.0), scripted_metrics(89.5, 20.0)], keep_every=2
        )
        periodic = [filename for filename, _, _ in saved if filename.startswith("checkpoint_epoch_")]
        self.assertEqual(periodic, ["checkpoint_epoch_0001.pth", "checkpoint_epoch_0002.pth"])

    def test_val_every_skips_validation_but_still_saves_last_model(self):
        # Three epochs, validated after epochs 2 and 3 (the last); epoch 1 only saves last_model.pth.
        outcome, saved, _ = self.run_scripted(
            [scripted_metrics(90.0, 10.0), scripted_metrics(87.0, 50.0), scripted_metrics(89.5, 20.0)], val_every=2, epochs_num=3
        )
        self.assertEqual(outcome["state"], "completed")
        last = [state for filename, _, state in saved if filename == "last_model.pth"]
        self.assertEqual([state["next_epoch"] for state in last], [1, 2, 3])
        self.assertIsNone(last[0]["validation_metrics"])
        self.assertEqual(last[0]["not_improved_num"], 0)
        self.assertEqual(last[1]["runtime_state"]["best_checkpoint_epochs"]["best"], 2)
        self.assertEqual(last[2]["not_improved_num"], 1)

    def test_non_improving_validations_during_the_ramp_do_not_count_toward_patience(self):
        # Ramp of 2 epochs (no warm-up): only epoch 3 trains at full strength, so only it counts.
        worse = [scripted_metrics(90.0, 50.0)] + [scripted_metrics(89.0, 10.0)] * 3
        _, saved, _ = self.run_scripted(worse, attack_ramp_epochs=2)
        last = [state for filename, _, state in saved if filename == "last_model.pth"]
        self.assertEqual([state["not_improved_num"] for state in last], [0, 0, 1])

        _, saved, _ = self.run_scripted(worse, attack_ramp_epochs=2, is_clean_only=True)
        last = [state for filename, _, state in saved if filename == "last_model.pth"]
        self.assertEqual([state["not_improved_num"] for state in last], [1, 2, 3])

    def test_most_robust_epoch_is_best_and_budget_checkpoints_follow_clean_drop(self):
        # C0=90, initial robust 10. Epoch 1: clean 87 (drop 3), robust 50 -> best overall and
        # best within budgets 3 and 5. Epoch 2: clean 89.5, robust 20 -> best within budget 1 only.
        outcome, saved, copied = self.run_scripted(
            [scripted_metrics(90.0, 10.0), scripted_metrics(87.0, 50.0), scripted_metrics(89.5, 20.0)]
        )
        self.assertEqual(outcome["state"], "completed")
        last = [(is_best, state) for filename, is_best, state in saved if filename == "last_model.pth"]
        self.assertEqual([is_best for is_best, _ in last], [True, False])
        self.assertEqual(last[0][1]["best_r5"], 50.0)
        self.assertEqual(last[1][1]["not_improved_num"], 1)
        self.assertEqual(
            copied,
            [("initial_validation_model.pth", ["1", "3", "5"]), ("last_model.pth", ["3", "5"]), ("last_model.pth", ["1"])],
        )
        self.assertEqual(last[1][1]["runtime_state"]["best_budget_scores"], {"1": 20.0, "3": 50.0, "5": 50.0})
        initial_state = [state for filename, _, state in saved if filename == "initial_validation_model.pth"][0]
        self.assertEqual(initial_state["runtime_state"]["best_checkpoint_epochs"], {"best": -1, "1": -1, "3": -1, "5": -1})
        self.assertEqual(last[1][1]["runtime_state"]["best_checkpoint_epochs"], {"best": 1, "1": 2, "3": 1, "5": 1})
        for _, _, state in saved:
            self.assertEqual(state["runtime_state"]["initial_clean_r1"], 90.0)

    def test_clean_only_run_writes_budget_checkpoints(self):
        # C0=90. Epoch 1: clean 92 -> best and best within every budget. Epoch 2: clean 91 -> nothing.
        clean = lambda value: {"NoAttack": {"recalls": {f"R@{k}": value for k in (1, 5, 10, 100)}, "recalls_list": [value] * 4}}
        outcome, saved, copied = self.run_scripted([clean(90.0), clean(92.0), clean(91.0)], is_clean_only=True)
        self.assertEqual(outcome["state"], "completed")
        self.assertEqual(
            copied, [("initial_validation_model.pth", ["1", "3", "5"]), ("last_model.pth", ["1", "3", "5"])]
        )
        last_state = [state for filename, _, state in saved if filename == "last_model.pth"][-1]
        self.assertEqual(last_state["runtime_state"]["best_budget_scores"], {"1": 92.0, "3": 92.0, "5": 92.0})

    def test_resume_keeps_recorded_checkpoint_epochs(self):
        resume = {
            "initial_clean_r1": 90.0,
            "best_budget_scores": {"1": 20.0, "3": 50.0, "5": 50.0},
            "best_checkpoint_epochs": {"best": 1, "1": 2, "3": 1, "5": 1},
        }
        # Epoch 3: clean 87 (drop 3), robust 60 -> new best and best within budgets 3 and 5.
        _, saved, _ = self.run_scripted([scripted_metrics(87.0, 60.0)], start_epoch=2, resume_runtime_state=resume)
        last_state = [state for filename, _, state in saved if filename == "last_model.pth"][-1]
        self.assertEqual(last_state["runtime_state"]["best_checkpoint_epochs"], {"best": 3, "1": 2, "3": 3, "5": 3})

    def test_warns_when_best_model_is_the_initial_model(self):
        clean = lambda value: {"NoAttack": {"recalls": {f"R@{k}": value for k in (1, 5, 10, 100)}, "recalls_list": [value] * 4}}
        with self.assertLogs(level="WARNING") as logs:
            self.run_scripted([clean(90.0), clean(89.0)], is_clean_only=True)
        self.assertTrue(any("best_model.pth is the initial" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
