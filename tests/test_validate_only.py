"""Validation-only screens produce the same table row as training screens."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.rank_validation import select_checkpoint  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_only = load_script("validate_only")
screen_summary = load_script("screen_summary")

SMALL, LARGE = "rank_pgd_linf_eps_0.0342", "rank_pgd_linf_eps_0.0685"


def metric(r1):
    return {"recalls": {"R@1": r1, "R@5": r1 + 5, "R@10": r1 + 7, "R@100": r1 + 10}}


def fake_metrics(clean, small, large):
    return {"NoAttack": metric(clean), SMALL: metric(small), LARGE: metric(large)}


class ValidateOnlyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, key, clean, small, large, initial=88.10):
        metrics = fake_metrics(clean, small, large)
        scores = select_checkpoint(metrics, initial, [1, 3, 5, 7, 10], False)
        return validate_only.write_screen_record(self.root / key, metrics, scores, initial)

    def test_record_has_the_trainer_fields(self):
        self.write("screen", 83.5, 61.2, 33.34)
        lines = (self.root / "screen" / "validation_recalls.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["epoch"], 12)
        self.assertEqual(record["initial_clean_r1"], 88.10)
        self.assertEqual(record["clean"]["R@1"], 83.5)
        self.assertEqual(set(record["attacks"]), {SMALL, LARGE})
        self.assertAlmostEqual(record["attacked_mean"]["R@1"], (61.2 + 33.34) / 2)
        self.assertAlmostEqual(record["robust_score"], (61.2 + 33.34) / 2)
        self.assertEqual(record["eligible_budgets"], [5.0, 7.0, 10.0])
        status = json.loads((self.root / "screen" / "run_status.json").read_text())
        self.assertEqual(status["state"], "completed")

    def test_rerun_replaces_the_record(self):
        self.write("screen", 80.0, 50.0, 30.0)
        self.write("screen", 83.5, 61.2, 33.34)
        lines = (self.root / "screen" / "validation_recalls.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["clean"]["R@1"], 83.5)

    def test_screen_summary_row_matches_a_training_screen(self):
        self.write("araft_a0", 83.5, 61.2, 33.34)
        self.write("soup_a0.5", 82.0, 70.0, 40.0)
        rows = {row["run"]: row for row in screen_summary.summarise(self.root, 12, budgets=(5.0, 7.0))}
        within = rows["araft_a0"]
        self.assertEqual(within["state"], "completed")
        self.assertEqual(within["last_epoch"], 12)
        self.assertAlmostEqual(within["clean_drop"], 88.10 - 83.5)
        self.assertAlmostEqual(within["R@1 0.0342"], 61.2)
        self.assertAlmostEqual(within["R@1 0.0685"], 33.34)
        self.assertEqual(within["b5"], "12 (47.27)")
        self.assertEqual(within["b7"], "12 (47.27)")
        outside = rows["soup_a0.5"]
        self.assertAlmostEqual(outside["clean_drop"], 88.10 - 82.0)
        self.assertEqual(outside["b5"], "-")
        self.assertEqual(outside["b7"], "12 (55.00)")

    def test_arguments_are_split(self):
        key, initial, rest = validate_only.split_arguments(
            ["k", "--resize", "322", "322", "--initial_clean_r1", "88.1", "--seed=0"]
        )
        self.assertEqual((key, initial, rest), ("k", 88.1, ["--resize", "322", "322", "--seed=0"]))
        self.assertEqual(validate_only.split_arguments(["k", "--initial_clean_r1=91.35"])[1], 91.35)
        for argv in ([], ["--seed=0"], ["k", "--seed=0"], ["k", "--initial_clean_r1"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                validate_only.split_arguments(argv)


if __name__ == "__main__":
    unittest.main()
