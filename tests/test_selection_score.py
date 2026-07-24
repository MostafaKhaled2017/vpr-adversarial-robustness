import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.train_loop import compute_validation_selection_scores


def make_metrics():
    return {
        "NoAttack": {"recalls": {"R@1": 50.0, "R@5": 70.0}},
        "pgd": {"recalls": {"R@1": 10.0, "R@5": 30.0}},
    }


class SelectionScoreTests(unittest.TestCase):
    def test_default_weight_is_three_quarters_robust(self):
        scores = compute_validation_selection_scores(make_metrics())
        self.assertEqual(scores["clean_score"], 120.0)
        self.assertEqual(scores["robust_score"], 40.0)
        self.assertAlmostEqual(scores["selection_score"], 0.25 * 120.0 + 0.75 * 40.0)

    def test_robust_weight_one_selects_on_robust_score_only(self):
        scores = compute_validation_selection_scores(make_metrics(), robust_weight=1.0)
        self.assertAlmostEqual(scores["selection_score"], scores["robust_score"])

    def test_robust_weight_zero_selects_on_clean_score_only(self):
        scores = compute_validation_selection_scores(make_metrics(), robust_weight=0.0)
        self.assertAlmostEqual(scores["selection_score"], scores["clean_score"])

    def test_custom_weight_blends_clean_and_robust(self):
        scores = compute_validation_selection_scores(make_metrics(), robust_weight=0.5)
        self.assertAlmostEqual(scores["selection_score"], 0.5 * 120.0 + 0.5 * 40.0)

    def test_without_attacks_robust_falls_back_to_clean(self):
        metrics = {"NoAttack": {"recalls": {"R@1": 50.0, "R@5": 70.0}}}
        scores = compute_validation_selection_scores(metrics, robust_weight=0.75)
        self.assertEqual(scores["robust_score"], scores["clean_score"])
        self.assertAlmostEqual(scores["selection_score"], scores["clean_score"])


if __name__ == "__main__":
    unittest.main()
