import csv
import tempfile
import unittest
from pathlib import Path

from src.robustness_curve_summary import main, normalized_auc, summarize, worst_case


def row(model, condition, epsilon, r1, success=""):
    return {"dataset": "msls", "model": model, "condition": condition, "epsilon": epsilon, "R@1": r1,
            "targeted_success_rate": success}


ROWS = [
    row("clean_ft", "clean_all_queries", "", "80"),
    row("clean_ft", "clean_attacked_subset", "", "76"),
    row("clean_ft", "rank_pgd_linf_eps_0.5", "0.5", "40"),
    row("clean_ft", "rank_pgd_linf_eps_1", "1.0", "20"),
    row("mplc", "clean_all_queries", "", "78"),
    row("mplc", "clean_attacked_subset", "", "74"),
    row("mplc", "rank_pgd_linf_eps_0.5", "0.5", "60"),
    row("mplc", "rank_pgd_linf_eps_1", "1.0", "50"),
    row("mplc", "rank_pgd_linf_eps_1_targeted", "1.0", "70", "12.5"),
]


class NormalizedAucTests(unittest.TestCase):
    def test_trapezoid_from_clean_point(self):
        # (80+40)/2*0.5 + (40+20)/2*0.5 = 45, divided by eps_max 1.0
        self.assertAlmostEqual(normalized_auc([0.5, 1.0], [40.0, 20.0], 80.0), 45.0)


class SummarizeTests(unittest.TestCase):
    def test_groups_by_attack_family_and_reports_clean_drop(self):
        summary = summarize(ROWS, reference_model="clean_ft")
        by_key = {(item["model"], item["family"]): item for item in summary}
        self.assertEqual(set(by_key), {("clean_ft", "rank_pgd_linf"), ("mplc", "rank_pgd_linf"), ("mplc", "rank_pgd_linf_targeted")})
        mplc = by_key[("mplc", "rank_pgd_linf")]
        # The eps=0 anchor is clean R@1 over the attacked queries (74), not all queries (78).
        self.assertAlmostEqual(mplc["auc"], ((74 + 60) / 2 * 0.5 + (60 + 50) / 2 * 0.5) / 1.0)
        self.assertAlmostEqual(mplc["clean_r1"], 78.0)
        self.assertAlmostEqual(mplc["clean_drop_vs_reference"], 2.0)
        self.assertEqual(mplc["r1_by_epsilon"], {0.5: 60.0, 1.0: 50.0})
        self.assertEqual(by_key[("mplc", "rank_pgd_linf_targeted")]["mean_targeted_success"], 12.5)

    def test_curve_anchor_falls_back_to_all_queries_without_subset_row(self):
        rows = [r for r in ROWS if r["condition"] != "clean_attacked_subset"]
        by_key = {(item["model"], item["family"]): item for item in summarize(rows)}
        mplc = by_key[("mplc", "rank_pgd_linf")]
        self.assertAlmostEqual(mplc["auc"], ((78 + 60) / 2 * 0.5 + (60 + 50) / 2 * 0.5) / 1.0)
        self.assertAlmostEqual(mplc["clean_r1"], 78.0)

    def test_duplicate_curve_point_is_rejected(self):
        rows = [*ROWS, row("mplc", "rank_pgd_linf_eps_1", "1.0", "45")]
        with self.assertRaisesRegex(ValueError, r"duplicate.*mplc.*rank_pgd_linf.*1\.0"):
            summarize(rows)


def query_row(model, condition, query_id, correct, epsilon="0.0685"):
    return {"dataset": "msls", "model_tag": model, "condition": condition, "epsilon": epsilon,
            "query_id": str(query_id), "attacked_rank": "1" if correct else "7",
            "attacked_correct_at_1": str(correct)}


class WorstCaseTests(unittest.TestCase):
    def test_query_is_robust_only_if_every_attack_fails(self):
        rows = [
            query_row("mplc", "rank_apgd_linf_eps_0.0685", 0, True),
            query_row("mplc", "rank_apgd_linf_eps_0.0685", 1, True),
            query_row("mplc", "embshift_linf_eps_0.0685", 0, True),
            query_row("mplc", "embshift_linf_eps_0.0685", 1, False),
        ]
        (item,) = worst_case(rows)
        self.assertEqual(item["worst_case_r1"], 50.0)
        self.assertEqual(item["attacks"], "embshift_linf rank_apgd_linf")
        self.assertEqual(item["queries"], 2)

    def test_models_and_epsilons_are_kept_apart(self):
        rows = [
            query_row("mplc", "rank_apgd_linf_eps_0.0685", 0, True),
            query_row("twin", "rank_apgd_linf_eps_0.0685", 0, False),
            query_row("mplc", "rank_apgd_linf_eps_0.0342", 0, False, epsilon="0.0342"),
        ]
        results = {(item["model"], item["epsilon"]): item["worst_case_r1"] for item in worst_case(rows)}
        self.assertEqual(results, {("mplc", 0.0342): 0.0, ("mplc", 0.0685): 100.0, ("twin", 0.0685): 0.0})

    def test_attacks_over_different_queries_are_rejected(self):
        rows = [
            query_row("mplc", "rank_apgd_linf_eps_0.0685", 0, True),
            query_row("mplc", "rank_apgd_linf_eps_0.0685", 1, True),
            query_row("mplc", "embshift_linf_eps_0.0685", 0, True),
        ]
        with self.assertRaisesRegex(ValueError, "different queries"):
            worst_case(rows)

    def test_cli_reads_per_query_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, correct in (("apgd", [True, True]), ("targeted", [True, False])):
                path = Path(tmp) / f"{name}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(query_row("m", "c", 0, True)))
                    writer.writeheader()
                    for query_id, ok in enumerate(correct):
                        # bools, as rank_eval writes them
                        writer.writerow({**query_row("mplc", f"{name}_eps_0.0685", query_id, ok), "attacked_correct_at_1": ok})
                paths.append(str(path))
            output = Path(tmp) / "worst.csv"
            main([*paths, "--worst_case", "--output", str(output)])
            with output.open() as handle:
                (record,) = list(csv.DictReader(handle))
        self.assertEqual(float(record["worst_case_r1"]), 50.0)
        self.assertEqual(record["attacks"], "apgd targeted")


if __name__ == "__main__":
    unittest.main()
