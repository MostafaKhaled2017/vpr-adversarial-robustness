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


if __name__ == "__main__":
    unittest.main()
