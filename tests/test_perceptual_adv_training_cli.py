import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments


class PerceptualTrainingCliTests(unittest.TestCase):
    def parse(self, *extra):
        return parse_arguments(["--eval_datasets_folder", "/tmp", "--device", "cpu", *extra])

    def test_supervlad_defaults_remain_available(self):
        args = self.parse()
        self.assertEqual(args.model, "supervlad")
        self.assertEqual(args.train_resize, args.resize)
        self.assertIsNone(args.weight_decay)

    def test_boq_defaults_to_published_dinov2_dimension(self):
        args = self.parse("--model", "boq")
        self.assertEqual(args.boq_backbone, "Dinov2")
        self.assertEqual(args.boq_descriptors_dimension, 12288)

    def test_boq_rejects_incompatible_dimension(self):
        with self.assertRaisesRegex(ValueError, "supports descriptor dimension"):
            self.parse("--model", "boq", "--boq_descriptors_dimension", "16384")

    def test_mixvpr_defaults_to_4096_dimensions_and_trains_backbone(self):
        args = self.parse("--model", "mixvpr")
        self.assertEqual(args.mixvpr_descriptors_dimension, 4096)
        self.assertFalse(args.mixvpr_freeze_backbone)

    def test_mixvpr_accepts_all_published_dimensions(self):
        for dimension in (128, 512, 4096):
            with self.subTest(dimension=dimension):
                args = self.parse(
                    "--model",
                    "mixvpr",
                    "--mixvpr_descriptors_dimension",
                    str(dimension),
                )
                self.assertEqual(args.mixvpr_descriptors_dimension, dimension)

    def test_mixvpr_rejects_unknown_dimension(self):
        with self.assertRaises(SystemExit):
            self.parse("--model", "mixvpr", "--mixvpr_descriptors_dimension", "1024")

    def test_mixvpr_can_freeze_backbone(self):
        args = self.parse("--model", "mixvpr", "--mixvpr_freeze_backbone")
        self.assertTrue(args.mixvpr_freeze_backbone)

    def test_train_resize_can_differ_from_evaluation_resize(self):
        args = self.parse("--train_resize", "224", "224", "--resize", "322", "322")
        self.assertEqual(args.train_resize, [224, 224])
        self.assertEqual(args.resize, [322, 322])

    def test_launcher_resize_arguments_parse(self):
        args = self.parse(
            "--model=boq",
            "--boq_backbone=Dinov2",
            "--boq_descriptors_dimension=12288",
            "--train_resize",
            "224",
            "224",
            "--resize",
            "322",
            "322",
        )
        self.assertEqual(args.train_resize, [224, 224])
        self.assertEqual(args.resize, [322, 322])

    def test_mixvpr_launcher_arguments_parse(self):
        args = self.parse(
            "--model=mixvpr",
            "--mixvpr_descriptors_dimension=4096",
            "--train_resize",
            "320",
            "320",
            "--resize",
            "320",
            "320",
            "--optim=adamw",
            "--weight_decay=0.0001",
        )
        self.assertEqual(args.model, "mixvpr")
        self.assertEqual(args.mixvpr_descriptors_dimension, 4096)
        self.assertEqual(args.train_resize, [320, 320])
        self.assertEqual(args.resize, [320, 320])

    def test_negative_weight_decay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.parse("--weight_decay", "-0.1")

    def test_batches_per_epoch_defaults_to_full_epoch(self):
        self.assertIsNone(self.parse().batches_per_epoch)

    def test_batches_per_epoch_accepts_positive_value(self):
        args = self.parse("--batches_per_epoch", "400")
        self.assertEqual(args.batches_per_epoch, 400)

    def test_batches_per_epoch_rejects_zero(self):
        with self.assertRaisesRegex(ValueError, "batches_per_epoch"):
            self.parse("--batches_per_epoch", "0")

    def test_shuffle_defaults_off(self):
        self.assertFalse(self.parse().shuffle)

    def test_shuffle_flag_parses(self):
        self.assertTrue(self.parse("--shuffle").shuffle)

    def test_lr_plateau_defaults(self):
        args = self.parse()
        self.assertIsNone(args.lr_plateau_patience)
        self.assertEqual(args.lr_plateau_factor, 0.1)

    def test_lr_plateau_patience_rejects_zero(self):
        with self.assertRaisesRegex(ValueError, "lr_plateau_patience"):
            self.parse("--lr_plateau_patience", "0")

    def test_lr_plateau_factor_must_be_a_fraction(self):
        with self.assertRaisesRegex(ValueError, "lr_plateau_factor"):
            self.parse("--lr_plateau_factor", "1.5")

    def test_selection_robust_weight_defaults(self):
        self.assertEqual(self.parse().selection_robust_weight, 0.75)

    def test_selection_robust_weight_parses_custom_value(self):
        self.assertEqual(self.parse("--selection_robust_weight", "1.0").selection_robust_weight, 1.0)

    def test_selection_robust_weight_rejects_values_above_one(self):
        with self.assertRaisesRegex(ValueError, "selection_robust_weight"):
            self.parse("--selection_robust_weight", "1.5")

    def test_selection_robust_weight_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "selection_robust_weight"):
            self.parse("--selection_robust_weight", "-0.1")

    def test_supervlad_adv_launcher_arguments_parse(self):
        args = self.parse(
            "--model=supervlad",
            "--backbone=dino",
            "--supervlad_clusters=4",
            "--crossimage_encoder",
            "--freeze_te=7",
            "--lr=0.000005",
            "--lr_plateau_patience=5",
            "--num_epochs=150",
            "--patience=15",
            "--batch_size=16",
            "--batches_per_epoch=400",
            "--mixed_precision",
            "--attack", "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)",
            "--attack", "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
            "--attack", "LinfAttack(model, num_iterations=10)",
            "--attack", "StAdvAttack(model, num_iterations=10)",
            "--attack", "ReColorAdvAttack(model, num_iterations=10)",
            "--randomize_attack",
            "--adv_loss_weight=0.25",
            "--adv_align_weight=0.2",
            "--adv_negatives=5",
            "--adv_warmup_epochs=5",
            "--keep_every=3",
            "--val_batches=40",
        )
        self.assertEqual(len(args.attack), 5)
        self.assertTrue(args.randomize_attack)
        self.assertTrue(args.mixed_precision)
        self.assertEqual(args.freeze_te, 7)
        self.assertEqual(args.batches_per_epoch, 400)
        self.assertEqual(args.lr_plateau_patience, 5)
        self.assertEqual(args.patience, 15)
        self.assertEqual(args.keep_every, 3)
        self.assertEqual(args.val_batches, 40)


class MatchedCleanOnlyFineTuneTests(unittest.TestCase):
    """Task 1.1 — the clean-only control must be matched to the adversarial run."""

    SHARED_FLAGS = (
        "--model=supervlad",
        "--backbone=dino",
        "--supervlad_clusters=4",
        "--crossimage_encoder",
        "--lr=1e-5",
        "--freeze_te=8",
        "--num_epochs=100",
        "--patience=12",
        "--batch_size=16",
        "--batches_per_epoch=400",
        "--lr_plateau_patience=5",
        "--keep_every=6",
        "--val_batches=200",
        "--adv_negatives=5",
        "--adv_warmup_epochs=1",
        "--adv_margin=0.1",
        "--mixed_precision",
        "--randomize_attack",
    )

    ADVERSARIAL_FLAGS = (
        "--attack",
        "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)",
        "--attack",
        "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
    )

    # Fields the two runs are *allowed* to differ in: the attack list itself and the
    # selection-rule fields derived from it. Everything else must match exactly.
    ALLOWED_DIFFERENCES = {
        "attack",
        "adversarial_attack_names",
        "is_clean_only",
        "checkpoint_selection_rule",
        "effective_selection_robust_weight",
    }

    def parse(self, *extra):
        return parse_arguments(["--eval_datasets_folder", "/tmp", "--device", "cpu", *extra])

    def test_clean_only_config_matches_adversarial_config_except_attack_fields(self):
        adversarial = self.parse(*self.SHARED_FLAGS, *self.ADVERSARIAL_FLAGS)
        clean_only = self.parse(*self.SHARED_FLAGS)

        adversarial_config = vars(adversarial)
        clean_config = vars(clean_only)
        self.assertEqual(set(adversarial_config), set(clean_config))

        differing = {
            key
            for key in adversarial_config
            if adversarial_config[key] != clean_config[key]
        }
        self.assertEqual(differing, self.ALLOWED_DIFFERENCES)

    def test_clean_only_keeps_optimizer_and_schedule_fields_identical(self):
        adversarial = self.parse(*self.SHARED_FLAGS, *self.ADVERSARIAL_FLAGS)
        clean_only = self.parse(*self.SHARED_FLAGS)

        for field in (
            "lr",
            "optim",
            "lr_schedule",
            "lr_plateau_patience",
            "lr_plateau_factor",
            "weight_decay",
            "clip_grad",
            "epochs_num",
            "patience",
            "train_batch_size",
            "batches_per_epoch",
            "freeze_te",
            "early_stop_min_delta",
            "keep_every",
            "seed",
            "adv_loss_weight",
            "adv_align_weight",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(adversarial, field), getattr(clean_only, field))

    def test_no_attack_is_an_accepted_attack_expression(self):
        args = self.parse("--attack", "NoAttack(model)")
        self.assertEqual(args.attack, ["NoAttack(model)"])
        self.assertTrue(args.is_clean_only)
        self.assertEqual(args.adversarial_attack_names, [])

    def test_clean_only_resolves_checkpoint_selection_to_clean_recall(self):
        args = self.parse("--selection_robust_weight=0.75")
        self.assertTrue(args.is_clean_only)
        self.assertEqual(args.checkpoint_selection_rule, "clean_recall")
        self.assertEqual(args.effective_selection_robust_weight, 0.0)

    def test_adversarial_run_keeps_robust_weighted_selection(self):
        args = self.parse(
            "--selection_robust_weight=0.75",
            "--attack",
            "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
        )
        self.assertFalse(args.is_clean_only)
        self.assertEqual(args.checkpoint_selection_rule, "robust_weighted")
        self.assertEqual(args.effective_selection_robust_weight, 0.75)
        self.assertEqual(args.adversarial_attack_names, ["PerceptualPGDAttack"])


class Phase2FlagDefaultTests(unittest.TestCase):
    """Every Phase 2 component is separately switchable and OFF by default.

    Phase 1's PAT re-run must reproduce the pre-Phase-2 method exactly, and Phase 4's
    ablations need each component to be independently toggleable, so no new flag may
    change behaviour unless it is asked for.
    """

    def parse(self, *extra):
        return parse_arguments(["--eval_datasets_folder", "/tmp", "--device", "cpu", *extra])

    def test_phase2_components_default_to_the_pre_phase2_method(self):
        args = self.parse()

        self.assertFalse(args.multi_positive)
        self.assertEqual(args.negative_pool_size, 0)
        self.assertEqual(args.defense_loss, "hinge")
        self.assertEqual(args.attack_ramp_epochs, 0)
        self.assertIsNone(args.collapse_abort_knn)

    def test_each_component_can_be_enabled_independently(self):
        args = self.parse(
            "--multi_positive",
            "--negative_pool_size=4096",
            "--defense_loss=listwise",
            "--listwise_tau=0.01",
            "--listwise_k=5",
            "--attack_ramp_epochs=5",
            "--attack_ramp_min_scale=0.25",
            "--collapse_abort_knn=0.2",
        )

        self.assertTrue(args.multi_positive)
        self.assertEqual(args.negative_pool_size, 4096)
        self.assertEqual(args.defense_loss, "listwise")
        self.assertAlmostEqual(args.listwise_tau, 0.01)
        self.assertEqual(args.listwise_k, 5)
        self.assertEqual(args.attack_ramp_epochs, 5)
        self.assertAlmostEqual(args.attack_ramp_min_scale, 0.25)
        self.assertAlmostEqual(args.collapse_abort_knn, 0.2)

    def test_negative_pool_size_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "negative_pool_size"):
            self.parse("--negative_pool_size=-1")

    def test_listwise_tau_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "listwise_tau"):
            self.parse("--listwise_tau=0")

    def test_listwise_k_must_be_at_least_one(self):
        with self.assertRaisesRegex(ValueError, "listwise_k"):
            self.parse("--listwise_k=0")

    def test_attack_ramp_min_scale_must_be_a_fraction(self):
        with self.assertRaisesRegex(ValueError, "attack_ramp_min_scale"):
            self.parse("--attack_ramp_min_scale=1.5")

    def test_unknown_defense_loss_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--defense_loss=triplet")


if __name__ == "__main__":
    unittest.main()
