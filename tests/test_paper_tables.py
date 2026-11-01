"""src/paper/tables.py reproduces the numbers printed in the paper (docs/reports/paper).

Needs the trained runs in logs/paper and the evaluations in output/paper; skipped without them.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from src.paper import tables

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS, OUTPUT = REPO_ROOT / "logs" / "paper", REPO_ROOT / "output" / "paper"


def read(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


@unittest.skipUnless((OUTPUT / "supervlad").is_dir() and LOGS.is_dir(), "no paper results in logs/paper, output/paper")
class PaperTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        tables.main(["--logs", str(LOGS), "--output", str(OUTPUT), "--out", str(cls.out)])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def means(self, name):
        return {(r["model"], r["method"]): (r["mean_clean"], r["mean_2"], r["mean_4"]) for r in read(self.out / name)}

    def test_budget_table(self):
        self.assertEqual(self.means("tab_budget.csv"), {
            ("supervlad", "Plain AT"): ("84.2", "0.0", "0.0"),
            ("supervlad", "ARAFT"): ("71.9", "46.1", "22.8"),
            ("boq", "Plain AT"): ("87.9", "0.0", "0.0"),
            ("boq", "ARAFT"): ("74.3", "49.8", "26.0"),
        })
        boq_araft = read(self.out / "tab_budget.csv")[3]
        self.assertEqual((boq_araft["sped_clean"], boq_araft["sped_2"], boq_araft["sped_4"]), ("69.0", "47.5", "23.6"))

    def test_main_table(self):
        self.assertEqual(self.means("tab_main.csv"), {
            ("supervlad", "Plain AT"): ("61.7", "46.4", "32.0"),
            ("supervlad", "ARAFT"): ("71.9", "46.1", "22.8"),
            ("boq", "Plain AT"): ("63.4", "47.2", "31.6"),
            ("boq", "ARAFT"): ("74.3", "49.8", "26.0"),
        })

    def test_ablation_table(self):
        rows = {r["variant"]: (r["msls_apgd4"], r["msls_clean"], r["nordland_apgd4"], r["nordland_clean"])
                for r in read(self.out / "tab_ablation.csv")}
        self.assertEqual(rows, {
            "ARAFT": ("14.8", "59.4", "3.7", "68.7"),
            "Closest positive": ("10.3", "60.4", "2.4", "70.4"),
            "Closest positive, hinge": ("10.9", "59.7", "2.6", "71.4"),
            "No anchor": ("19.8", "50.0", "8.0", "58.0"),
            "Stronger anchor": ("11.8", "60.4", "3.4", "71.0"),
        })

    def test_clean_table(self):
        rows = {(r["model"], r["row"]): [r[ds] for ds in tables.DATASETS] for r in read(self.out / "tab_clean.csv")}
        self.assertEqual(rows["supervlad", "Clean twin"], ["66.05", "92.33", "88.14", "79.20"])
        self.assertEqual(rows["supervlad", "ARAFT drop vs. twin"], ["6.63", "1.58", "19.28", "10.51"])
        self.assertEqual(rows["supervlad", "Plain AT drop vs. pretrained"], ["22.67", "5.52", "26.85", "34.90"])
        self.assertEqual(rows["boq", "ARAFT drop vs. pretrained"], ["9.43", "2.48", "23.56", "18.79"])
        # The paper prints 5.76 and 29.33 here: differences of the rounded values.
        self.assertEqual(rows["boq", "Plain AT drop vs. pretrained"], ["21.86", "5.75", "29.32", "40.95"])

    def test_in_text_numbers(self):
        numbers = (self.out / "numbers.md").read_text()
        for line in (
            "| supervlad_araft_s0 | 6.1 | 47.3 | 47.3 | 47.3 | 47.3 (4.6) | 2.2 |",
            "| supervlad_plain_at_s0 | — | — | — | 21.7 | 56.1 (14.5) | 9.5 |",
            "| boq_araft_s0 | 9.9 | 55.3 | 55.3 | 55.3 | 55.3 (4.7) | 1.4 |",
            "| boq_plain_at_s0 | — | — | 28.7 | 28.7 | 58.3 (13.0) | 6.9 |",
            "clean drop of 14.3.",
            "clean drop of 13.0.",
            "| supervlad | msls | 0.03425 | araft | plain_at | 38.0 | 34.9 | 96 | 65 | 0.0178 |",
            "| supervlad | nordland | 0.03425 | araft | plain_at | 24.7 | 28.0 | 72 | 105 | 0.0159 |",
            "| boq | nordland | 0.0342 | araft | plain_at | 27.6 | 23.7 | 108 | 69 | 0.00416 |",
            "| supervlad | sped | 0.0685 | araft | araft_closest | 22.4 | 20.1 | 22 | 8 | 0.0161 |",
            "MSLS, 500 steps vs. APGD-100: +0.00",
            "MSLS, 3 restarts vs. APGD-100: +0.00",
            "eps 0.548: 0.00",
        ):
            self.assertIn(line, numbers)
        self.assertNotIn("| False |", numbers)  # attacked R@1 never grows with eps


if __name__ == "__main__":
    unittest.main()
