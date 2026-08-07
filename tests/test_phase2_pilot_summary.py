import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.phase2_pilot_summary import (
    best_validation_epoch,
    meets_acceptance,
    rank_pilot_runs,
    read_collapse_metrics,
)

FIELDNAMES = [
    "epoch",
    "split",
    "attack",
    "R@1",
    "R@5",
    "R@10",
    "R@100",
    "clean_score",
    "robust_score",
    "selection_score",
]


def write_validation_csv(run_dir, epochs):
    """epochs: list of (epoch, clean_r1, attacked_r1, selection_score)."""
    path = Path(run_dir) / "validation_recalls.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for epoch, clean_r1, attacked_r1, selection in epochs:
            base = {
                "epoch": epoch,
                "R@5": 0.0,
                "R@10": 0.0,
                "R@100": 0.0,
                "clean_score": clean_r1,
                "robust_score": attacked_r1,
                "selection_score": selection,
            }
            writer.writerow({**base, "split": "clean", "attack": "NoAttack", "R@1": clean_r1})
            writer.writerow({**base, "split": "attacked_mean", "attack": "mean", "R@1": attacked_r1})
    return path


def write_validation_jsonl(run_dir, records):
    path = Path(run_dir) / "validation_recalls.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


class BestEpochTests(unittest.TestCase):
    def test_best_epoch_is_the_one_with_the_highest_selection_score(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_validation_csv(run_dir, [(1, 80.0, 40.0, 50.0), (2, 78.0, 55.0, 61.0), (3, 77.0, 50.0, 58.0)])

            best = best_validation_epoch(Path(run_dir))

        self.assertEqual(best["epoch"], 2)
        self.assertAlmostEqual(best["clean_r1"], 78.0)
        self.assertAlmostEqual(best["attacked_r1"], 55.0)

    def test_a_run_without_attacked_rows_reports_no_attacked_recall(self):
        with tempfile.TemporaryDirectory() as run_dir:
            path = Path(run_dir) / "validation_recalls.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "split": "clean",
                        "attack": "NoAttack",
                        "R@1": 85.0,
                        "R@5": 0.0,
                        "R@10": 0.0,
                        "R@100": 0.0,
                        "clean_score": 85.0,
                        "robust_score": 85.0,
                        "selection_score": 85.0,
                    }
                )

            best = best_validation_epoch(Path(run_dir))

        self.assertAlmostEqual(best["clean_r1"], 85.0)
        self.assertIsNone(best["attacked_r1"])

    def test_a_missing_csv_returns_none(self):
        with tempfile.TemporaryDirectory() as run_dir:
            self.assertIsNone(best_validation_epoch(Path(run_dir)))


class CollapseReadingTests(unittest.TestCase):
    def test_collapse_metrics_come_from_the_matching_epoch(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_validation_jsonl(
                run_dir,
                [
                    {"epoch": 1, "collapse": {"knn_overlap": 0.9, "mean_pairwise_cosine": 0.1}},
                    {"epoch": 2, "collapse": {"knn_overlap": 0.4, "mean_pairwise_cosine": 0.7}},
                ],
            )

            collapse = read_collapse_metrics(Path(run_dir), epoch=2)

        self.assertAlmostEqual(collapse["knn_overlap"], 0.4)

    def test_absent_collapse_block_reads_as_none(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_validation_jsonl(run_dir, [{"epoch": 1, "collapse": None}])

            self.assertIsNone(read_collapse_metrics(Path(run_dir), epoch=1))


class AcceptanceTests(unittest.TestCase):
    def test_config_within_two_points_of_clean_ft_and_beating_pat_is_accepted(self):
        self.assertTrue(
            meets_acceptance(
                clean_r1=84.5,
                attacked_r1=60.0,
                clean_ft_clean_r1=86.0,
                pat_attacked_r1=58.0,
                max_clean_drop=2.0,
            )
        )

    def test_too_large_a_clean_drop_is_rejected(self):
        self.assertFalse(
            meets_acceptance(
                clean_r1=83.0,
                attacked_r1=70.0,
                clean_ft_clean_r1=86.0,
                pat_attacked_r1=58.0,
                max_clean_drop=2.0,
            )
        )

    def test_failing_to_match_pat_robustness_is_rejected(self):
        self.assertFalse(
            meets_acceptance(
                clean_r1=85.5,
                attacked_r1=57.0,
                clean_ft_clean_r1=86.0,
                pat_attacked_r1=58.0,
                max_clean_drop=2.0,
            )
        )

    def test_acceptance_is_unknown_without_baselines(self):
        self.assertIsNone(
            meets_acceptance(
                clean_r1=85.0,
                attacked_r1=60.0,
                clean_ft_clean_r1=None,
                pat_attacked_r1=None,
                max_clean_drop=2.0,
            )
        )


class RankingTests(unittest.TestCase):
    def build_runs(self, directory):
        runs = []
        # (label, clean_r1, attacked_r1)
        for label, clean_r1, attacked_r1 in [
            ("tau0.05_k1_pool0", 84.0, 60.0),
            ("tau0.01_k5_pool4096", 85.5, 62.0),
            ("tau0.01_k1_pool0", 79.0, 68.0),
        ]:
            run_dir = Path(directory) / label
            run_dir.mkdir()
            write_validation_csv(run_dir, [(1, clean_r1, attacked_r1, attacked_r1)])
            runs.append((label, run_dir))
        return runs

    def test_accepted_configs_are_ranked_above_rejected_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = rank_pilot_runs(
                self.build_runs(directory),
                clean_ft_clean_r1=86.0,
                pat_attacked_r1=58.0,
                max_clean_drop=2.0,
            )

        # tau0.01_k5_pool4096 loses only 0.5 clean points and beats PAT by the most.
        self.assertEqual(rows[0]["label"], "tau0.01_k5_pool4096")
        self.assertTrue(rows[0]["accepted"])
        # tau0.01_k1_pool0 is the most robust but drops 7 clean points, so it is rejected.
        self.assertEqual(rows[-1]["label"], "tau0.01_k1_pool0")
        self.assertFalse(rows[-1]["accepted"])

    def test_every_run_reports_its_deltas_against_the_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = rank_pilot_runs(
                self.build_runs(directory),
                clean_ft_clean_r1=86.0,
                pat_attacked_r1=58.0,
                max_clean_drop=2.0,
            )

        best = rows[0]
        self.assertAlmostEqual(best["clean_r1_drop_vs_clean_ft"], 0.5)
        self.assertAlmostEqual(best["attacked_r1_gain_vs_pat"], 4.0)

    def test_runs_without_results_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = self.build_runs(directory)
            empty = Path(directory) / "unfinished"
            empty.mkdir()
            runs.append(("unfinished", empty))

            rows = rank_pilot_runs(runs, clean_ft_clean_r1=86.0, pat_attacked_r1=58.0, max_clean_drop=2.0)

        self.assertNotIn("unfinished", [row["label"] for row in rows])


if __name__ == "__main__":
    unittest.main()
