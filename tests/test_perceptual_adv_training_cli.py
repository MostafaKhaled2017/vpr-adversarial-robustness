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

    def test_negative_weight_decay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.parse("--weight_decay", "-0.1")


if __name__ == "__main__":
    unittest.main()
