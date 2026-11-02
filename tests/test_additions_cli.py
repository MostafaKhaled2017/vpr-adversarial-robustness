import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments

BASE = [
    "--eval_datasets_folder", "/tmp", "--device", "cpu",
    "--model=supervlad", "--attack", "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
]
INITIAL = BASE + ["--align_target", "initial", "--resume", "x.pth"]


class AdditionsCliTests(unittest.TestCase):
    def test_defaults(self):
        args = parse_arguments(BASE)
        self.assertEqual(args.clean_drift_weight, 0.0)
        self.assertIsNone(args.negative_bank)
        self.assertEqual(args.bank_negatives, 5)
        self.assertEqual(args.bank_device, "cuda")
        self.assertEqual(args.anchor_kind, "isotropic")
        self.assertEqual(args.relational_neighbours, 32)
        self.assertEqual(args.relational_temperature, 0.1)

    def test_drift_requires_initial_anchor(self):
        with self.assertRaisesRegex(ValueError, "--clean_drift_weight"):
            parse_arguments(BASE + ["--clean_drift_weight", "1"])
        self.assertEqual(parse_arguments(INITIAL + ["--clean_drift_weight", "1"]).clean_drift_weight, 1.0)

    def test_negative_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "--clean_drift_weight"):
            parse_arguments(INITIAL + ["--clean_drift_weight", "-1"])

    def test_relational_requires_bank_and_initial_anchor(self):
        with self.assertRaisesRegex(ValueError, "--negative_bank"):
            parse_arguments(INITIAL + ["--anchor_kind", "relational"])
        with self.assertRaisesRegex(ValueError, "--align_target initial"):
            parse_arguments(BASE + ["--anchor_kind", "relational", "--negative_bank", "bank.pt"])

    def test_counts_and_temperature_bounds(self):
        for flag in ("--bank_negatives", "--relational_neighbours", "--relational_temperature"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, flag):
                parse_arguments(BASE + [flag, "0"])

    def test_full_combination_parses(self):
        args = parse_arguments(INITIAL + [
            "--clean_drift_weight", "0.5", "--negative_bank", "bank.pt", "--bank_negatives", "8",
            "--bank_device", "cpu", "--anchor_kind", "both", "--relational_neighbours", "16",
            "--relational_temperature", "0.5",
        ])
        self.assertEqual(args.clean_drift_weight, 0.5)
        self.assertEqual(args.negative_bank, "bank.pt")
        self.assertEqual(args.bank_negatives, 8)
        self.assertEqual(args.bank_device, "cpu")
        self.assertEqual(args.anchor_kind, "both")
        self.assertEqual(args.relational_neighbours, 16)
        self.assertEqual(args.relational_temperature, 0.5)


if __name__ == "__main__":
    unittest.main()
