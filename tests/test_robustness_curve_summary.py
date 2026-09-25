import unittest

from src.robustness_curve_summary import normalized_auc, summarize


def row(model, condition, epsilon, r1, success=""):
    return {"dataset": "msls", "model": model, "condition": condition, "epsilon": epsilon, "R@1": r1,
            "targeted_success_rate": success}


ROWS = [
    row("clean_ft", "clean_all_queries", "", "80"),
    row("clean_ft", "rank_pgd_linf_eps_0.5", "0.5", "40"),
    row("clean_ft", "rank_pgd_linf_eps_1", "1.0", "20"),
    row("mplc", "clean_all_queries", "", "78"),
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
        self.assertAlmostEqual(mplc["auc"], ((78 + 60) / 2 * 0.5 + (60 + 50) / 2 * 0.5) / 1.0)
        self.assertAlmostEqual(mplc["clean_drop_vs_reference"], 2.0)
        self.assertEqual(mplc["r1_by_epsilon"], {0.5: 60.0, 1.0: 50.0})
        self.assertEqual(by_key[("mplc", "rank_pgd_linf_targeted")]["mean_targeted_success"], 12.5)


if __name__ == "__main__":
    unittest.main()
