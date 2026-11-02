import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "screen_summary.py"
spec = importlib.util.spec_from_file_location("screen_summary", SCRIPT)
screen_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen_summary)

EPS_SMALL = "rank_pgd_linf_eps_0.0342"
EPS_LARGE = "rank_pgd_linf_eps_0.0685"


def record(epoch, clean_r1, small, large, robust, initial=None):
    rec = {
        "epoch": epoch,
        "clean": {"R@1": clean_r1, "R@5": clean_r1 + 2.0},
        "attacks": {EPS_SMALL: {"R@1": small}, EPS_LARGE: {"R@1": large}},
        "robust_score": robust,
        "clean_score": clean_r1,
        "selection_score": robust,
    }
    if initial is not None:
        rec["initial_clean_r1"] = initial
    return rec


def write_run(root, name, records, state):
    run = root / name
    run.mkdir()
    with open(run / "validation_recalls.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    (run / "run_status.json").write_text(json.dumps({"state": state}))


class ScreenSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # screen_a: initial clean 90; drops 2, 4, 6, 8 at epochs 3, 6, 9, 12.
        write_run(
            self.root,
            "screen_a",
            [
                record(-1, 90.0, 10.0, 2.0, 12.0),
                record(3, 88.0, 30.0, 10.0, 40.0, initial=90.0),
                record(6, 86.0, 35.0, 15.0, 50.0, initial=90.0),
                record(9, 84.0, 40.0, 20.0, 60.0, initial=90.0),
                record(12, 82.0, 45.0, 25.0, 70.0, initial=90.0),
            ],
            "completed",
        )
        write_run(
            self.root,
            "screen_b",
            [record(-1, 80.0, 8.0, 1.0, 9.0), record(3, 77.0, 20.0, 5.0, 25.0, initial=80.0)],
            "paused",
        )
        (self.root / "not_a_run").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_summarise_rows(self):
        rows = screen_summary.summarise(self.root, 12)
        self.assertEqual([row["run"] for row in rows], ["screen_a", "screen_b"])
        a, b = rows

        self.assertEqual(a["state"], "completed")
        self.assertEqual(a["last_epoch"], 12)
        self.assertAlmostEqual(a["clean_r1"], 82.0)
        self.assertAlmostEqual(a["clean_drop"], 90.0 - 82.0)
        self.assertAlmostEqual(a["R@1 0.0342"], 45.0)
        self.assertAlmostEqual(a["R@1 0.0685"], 25.0)
        self.assertAlmostEqual(a["robust"], 70.0)
        # drop <= 5: epochs 3 (robust 40) and 6 (robust 50); epoch -1 is never picked.
        self.assertEqual(a["b5"], "6 (50.00)")
        # drop <= 7: epochs 3, 6 and 9 -> 9 is most robust.
        self.assertEqual(a["b7"], "9 (60.00)")

        self.assertEqual(b["state"], "paused")
        self.assertEqual(b["last_epoch"], 3)
        for key in ("clean_r1", "clean_drop", "R@1 0.0342", "R@1 0.0685", "robust"):
            self.assertEqual(b[key], "-", key)
        # Budget columns use the epochs present: epoch 3 has drop 3.
        self.assertEqual(b["b5"], "3 (25.00)")

    def test_budget_without_eligible_epoch_is_dash(self):
        rows = screen_summary.summarise(self.root, 12, budgets=(1.0,))
        self.assertEqual(rows[0]["b1"], "-")
        self.assertEqual(rows[1]["b1"], "-")

    def test_initial_clean_fallback_without_initial_record(self):
        write_run(self.root, "screen_c", [record(12, 85.0, 1.0, 1.0, 5.0, initial=88.0)], "running")
        rows = screen_summary.summarise(self.root, 12)
        self.assertAlmostEqual(rows[2]["clean_drop"], 3.0)

    def test_missing_run_status_prints_dash(self):
        (self.root / "screen_b" / "run_status.json").unlink()
        rows = screen_summary.summarise(self.root, 12)
        self.assertEqual(rows[1]["state"], "-")

    def test_format_table(self):
        table = screen_summary.format_table(screen_summary.summarise(self.root, 12))
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)  # header, separator, two runs
        header = [cell.strip() for cell in lines[0].strip("|").split("|")]
        self.assertEqual(
            header,
            ["run", "state", "last epoch", "clean R@1", "clean drop", "R@1 0.0342", "R@1 0.0685", "robust", "b5", "b7"],
        )
        self.assertTrue(set(lines[1]) <= set("|- "))
        row_a = [cell.strip() for cell in lines[2].strip("|").split("|")]
        self.assertEqual(row_a, ["screen_a", "completed", "12", "82.00", "8.00", "45.00", "25.00", "70.00", "6 (50.00)", "9 (60.00)"])
        self.assertIn("screen_b", lines[3])

    def test_main_prints_table(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            screen_summary.main([str(self.root), "--epoch", "12", "--budgets", "5", "7"])
        expected = screen_summary.format_table(screen_summary.summarise(self.root, 12, budgets=(5.0, 7.0)))
        self.assertEqual(out.getvalue().strip(), expected.strip())


if __name__ == "__main__":
    unittest.main()
