import dataclasses
import re
import tempfile
import unittest
from pathlib import Path

from src import step1_sweep as sweep
from src import step2_plan as step2
from tests.test_step1_sweep import EPOCHS, REPO_ROOT, TRAIN_SCRIPT, finished, run_script, write_run

STEP2_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_step2.sh"
BATCH = int(re.search(r"^SUPERVLAD_TRAIN_BATCH_SIZE=(\d+)",
                      (REPO_ROOT / "scripts" / "lib" / "supervlad_common.sh").read_text(), re.M).group(1))


def finish_step1(root, batch_size=BATCH):
    """Write finished Step 1 screens (all scoring 50) until the sweep reaches MPLC*."""
    sweep.ensure_config(Path(root), EPOCHS, 5.0, batch_size, 200, 3)
    while True:
        state = sweep.plan_sweep(sweep._results_reader(Path(root), EPOCHS, 5.0))
        if state.final is not None:
            return state
        for cfg in state.pending:
            records, run_state = finished(50.0)
            write_run(Path(root) / sweep.run_name(cfg, EPOCHS), state=run_state, records=records)


class RunNameTests(unittest.TestCase):
    def test_names_match_the_launcher(self):
        cases = [
            {"MPLC_V2_ARMS": "clean_ft", "MPLC_V2_FREEZE_TE": "0", "MPLC_V2_LR": "3e-6",
             "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_SEED": "1"},
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_ALIGN_WEIGHT": "0",
             "MPLC_V2_DEFENSE_LOSS": "hinge", "MPLC_V2_MULTI_POSITIVE": "0", "MPLC_V2_FREEZE_TE": "0"},
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_FREEZE_TE": "0",
             "MPLC_V2_NUM_EPOCHS": "12", "MPLC_V2_VAL_EVERY": "3", "MPLC_V2_SEED": "2"},
            {"MPLC_V2_ARMS": "plain_at", "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_LR": "3e-6",
             "MPLC_V2_NUM_EPOCHS": "12", "MPLC_V2_VAL_EVERY": "3"},
            {"MPLC_V2_ARMS": "fare", "MPLC_V2_ALIGN_WEIGHT": "0.1", "MPLC_V2_FREEZE_TE": "0",
             "MPLC_V2_NUM_EPOCHS": "12", "MPLC_V2_VAL_EVERY": "3"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for env in cases:
                with self.subTest(env=env):
                    result = run_script(TRAIN_SCRIPT, {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_RUN_ROOT": tmp, **env})
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn(f"--save_dir={step2.run_name(env)}", result.stdout.split())


class LoadStep1Tests(unittest.TestCase):
    def test_reads_mplc_star_noise_schedule_and_star_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            finish_step1(tmp)
            step1 = step2.load_step1(Path(tmp), BATCH, 200)
        # Every screen scores 50, so each stage keeps its preferred option: linf, 8, 1e-5, 1.0.
        self.assertEqual(step1.star, {"MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_FREEZE_TE": "8",
                                      "MPLC_V2_LR": "1e-5", "MPLC_V2_ALIGN_WEIGHT": "1.0"})
        self.assertEqual((step1.noise, step1.epochs, step1.val_every, step1.budget), (0.0, EPOCHS, 3, 5.0))
        self.assertEqual(step1.star_screen.robust, 50.0)

    def test_unfinished_missing_or_resized_step1_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SystemExit, "mplc_v2_step1.sh"):
                step2.load_step1(root, BATCH, 200)
            sweep.ensure_config(root, EPOCHS, 5.0, BATCH, 200, 3)
            with self.assertRaisesRegex(SystemExit, "not finished"):
                step2.load_step1(root, BATCH, 200)
            finish_step1(root)
            with self.assertRaisesRegex(SystemExit, "batch"):
                step2.load_step1(root, BATCH + 4, 200)


# Step 1a noise 0, as on the server: the plan must decide with the 2.0 noise instead.
STEP1 = step2.Step1(
    {"MPLC_V2_ATTACK_MIX": "all", "MPLC_V2_FREEZE_TE": "4", "MPLC_V2_LR": "3e-6", "MPLC_V2_ALIGN_WEIGHT": "10"},
    noise=0.0, epochs=12, val_every=3, budget=5.0,
    star_screen=sweep.ScreenResult("completed", 12, 90.0, 12, 86.0, 40.0),
)


def fake(scores):
    """results(env) from {run_name: robust or None (loss)}; unlisted runs are unfinished."""

    def results(env):
        name = step2.run_name(env)
        if name not in scores:
            return None
        if scores[name] is None:
            return sweep.ScreenResult("completed", 9, 90.0)
        return sweep.ScreenResult("completed", 9, 90.0, 3, 89.0, float(scores[name]))

    return results


def names(plan, stage):
    return [step2.run_name(run.env) for run in plan.runs if run.stage == stage]


P = "mplc_v2_supervlad_"
NOISE = {P + "mplc_aw10_fte4_lr3e-6_ep12_s1": 41.0, P + "mplc_aw10_fte4_lr3e-6_ep12_s2": 39.0}  # with 40: noise 2
SCREENS = {  # the six baseline screens at N* = 4
    "plain_at 1e-5": P + "plain_at_fte4_ep12_s0", "plain_at 3e-6": P + "plain_at_fte4_lr3e-6_ep12_s0",
    "fare 1.0 1e-5": P + "fare_fte4_ep12_s0", "fare 1.0 3e-6": P + "fare_fte4_lr3e-6_ep12_s0",
    "fare 10 1e-5": P + "fare_aw10_fte4_ep12_s0", "fare 10 3e-6": P + "fare_aw10_fte4_lr3e-6_ep12_s0",
}


def screens(**changes):
    """NOISE plus every baseline screen at 40, with ``changes`` by SCREENS key or run name."""
    scores = {**NOISE, **{name: 40.0 for name in SCREENS.values()}}
    scores.update({SCREENS.get(k, k): v for k, v in changes.items()})
    return scores


class PlanTests(unittest.TestCase):
    def test_order_and_names_before_any_result(self):
        plan = step2.plan_step2(STEP1, fake({}))
        self.assertEqual(names(plan, "2.0"), list(NOISE))
        self.assertEqual(names(plan, "2.1"), [
            P + "clean_ft_fte4_lr3e-6_s0", P + "mplc_aw10_fte4_lr3e-6_s0",
            P + "clean_ft_fte4_lr3e-6_s1", P + "mplc_aw10_fte4_lr3e-6_s1",
        ])
        # Baselines take only freeze_te from MPLC*: never its lr 3e-6 or λ 10 on the preferred screens.
        self.assertEqual(names(plan, "2.2 screen"), list(SCREENS.values()))
        self.assertEqual(names(plan, "2.2 full"), [])
        self.assertEqual(names(plan, "2.3"), [
            P + "mplc_aw10_mixlinf_fte4_lr3e-6_s0", P + "mplc_aw0_fte4_lr3e-6_s0",
            P + "mplc_aw10_hinge_fte4_lr3e-6_s0", P + "mplc_aw10_sp_fte4_lr3e-6_s0",
        ])
        self.assertEqual([run.stage for run in plan.runs][:6], ["2.0"] * 2 + ["2.1"] * 4)
        self.assertEqual(plan.runs[-1].stage, "2.3")
        self.assertIsNone(plan.noise)

    def test_baselines_wait_for_the_noise(self):
        scores = screens(**{"plain_at 3e-6": 45.0})
        for name in NOISE:
            del scores[name]
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertEqual(names(plan, "2.2 full"), [])
        self.assertTrue(any("plain_at" in d and "2.0 noise" in d for d in plan.decisions), plan.decisions)

        plan = step2.plan_step2(STEP1, fake({**scores, **NOISE}))
        self.assertEqual(plan.noise, 2.0)
        self.assertEqual(names(plan, "2.2 full")[0], P + "plain_at_fte4_lr3e-6_s0")  # +5 > noise 2

    def test_within_noise_keeps_defaults_and_fare_extends_down(self):
        scores = screens(**{"plain_at 3e-6": 41.0})  # +1 < 2.0 noise (Step 1a's 0 would call it a gain)
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertIn(P + "fare_aw0.1_fte4_ep12_s0", names(plan, "2.2 screen"))
        self.assertEqual(names(plan, "2.2 full"), [P + "plain_at_fte4_s0"])  # fare waits for λ 0.1

        scores[P + "fare_aw0.1_fte4_ep12_s0"] = 45.0  # +5 > noise: real gain
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertEqual(names(plan, "2.2 full"), [P + "plain_at_fte4_s0", P + "fare_aw0.1_fte4_s0"])
        self.assertEqual(len(names(plan, "2.2 screen")), 7)

    def test_fare_extends_up_once_when_10_wins(self):
        scores = screens(**{"fare 10 3e-6": 50.0})
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertIn(P + "fare_aw30_fte4_lr3e-6_ep12_s0", names(plan, "2.2 screen"))
        scores[P + "fare_aw30_fte4_lr3e-6_ep12_s0"] = 45.0
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertIn(P + "fare_aw10_fte4_lr3e-6_s0", names(plan, "2.2 full"))
        self.assertEqual(len(names(plan, "2.2 screen")), 7)

    def test_all_loss_arm_gets_no_full_run(self):
        scores = screens(**{"plain_at 1e-5": None, "plain_at 3e-6": None})
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertFalse([n for n in names(plan, "2.2 full") if "plain_at" in n])
        self.assertTrue(any("plain_at" in d and "loss" in d for d in plan.decisions), plan.decisions)
        self.assertEqual(len(names(plan, "2.3")), 4)
        self.assertEqual(plan.stop_reason, "")

    def test_noise_screen_loss_stops_after_the_other_runs(self):
        scores = screens(**{P + "mplc_aw10_fte4_lr3e-6_ep12_s2": None})
        plan = step2.plan_step2(STEP1, fake(scores))
        self.assertIn("noise", plan.stop_reason)
        self.assertEqual(names(plan, "2.2 full"), [])
        self.assertEqual(len(names(plan, "2.1")), 4)
        self.assertEqual(len(names(plan, "2.3")), 4)

    def test_baseline_seeds_add_full_runs(self):
        scores = screens(**{P + "fare_aw0.1_fte4_ep12_s0": 40.0})
        plan = step2.plan_step2(STEP1, fake(scores), baseline_seeds=("0", "1"))
        self.assertEqual(names(plan, "2.2 full"), [
            P + "plain_at_fte4_s0", P + "plain_at_fte4_s1", P + "fare_fte4_s0", P + "fare_fte4_s1",
        ])

    def test_ablations_covered_by_mplc_star_are_skipped(self):
        star = dict(STEP1.star, MPLC_V2_ATTACK_MIX="linf", MPLC_V2_ALIGN_WEIGHT="0")
        plan = step2.plan_step2(dataclasses.replace(STEP1, star=star), fake({}))
        self.assertEqual(names(plan, "2.3"), [P + "mplc_aw0_mixlinf_hinge_fte4_lr3e-6_s0",
                                              P + "mplc_aw0_mixlinf_sp_fte4_lr3e-6_s0"])
        self.assertEqual(sum("covered by MPLC*" in d for d in plan.decisions), 2)


SIZE = ["--batch-size", str(BATCH), "--batches-per-epoch", "200"]


class CliTests(unittest.TestCase):
    def test_config_frozen_on_first_real_call(self):
        with tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
            finish_step1(s1)
            args = ["next", "--root", s2, "--step1-root", s1, *SIZE]
            step2.main(args + ["--no-freeze"])
            self.assertFalse((Path(s2) / step2.CONFIG_NAME).exists())
            step2.main(args)
            self.assertIn("MPLC_V2_ATTACK_MIX: linf", (Path(s2) / step2.CONFIG_NAME).read_text())
            step1 = step2.load_step1(Path(s1), BATCH, 200)
            for changed in (dataclasses.replace(step1, star=dict(step1.star, MPLC_V2_LR="3e-6")),
                            dataclasses.replace(step1, star_screen=sweep.ScreenResult("completed", 12, 90.0, 3, 89.0, 49.0))):
                with self.subTest(changed=changed), self.assertRaisesRegex(SystemExit, "another Step 1 result"):
                    step2.ensure_config(Path(s2), changed, Path(s1))

    def test_next_prints_pending_runs_then_stop_when_step1_is_missing(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                step2.main(["next", "--root", s2, "--step1-root", s1, *SIZE, "--no-freeze"])
            self.assertTrue(out.getvalue().startswith("STOP "), out.getvalue())

            finish_step1(s1)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                step2.main(["next", "--root", s2, "--step1-root", s1, *SIZE, "--no-freeze"])
            lines = out.getvalue().splitlines()
            self.assertEqual(lines[0], "MPLC_V2_ATTACK_MIX=linf MPLC_V2_FREEZE_TE=8 MPLC_V2_LR=1e-5 "
                                       "MPLC_V2_ALIGN_WEIGHT=1.0 MPLC_V2_NUM_EPOCHS=12 MPLC_V2_VAL_EVERY=3 "
                                       "MPLC_V2_ARMS=mplc MPLC_V2_SEED=1")
            self.assertEqual(len(lines), 2 + 4 + 6 + 3)  # mix linf already covers one ablation

    def test_table_labels_screens_against_the_preferred_one(self):
        scores = screens(**{"plain_at 3e-6": 43.0})
        plan = step2.plan_step2(STEP1, fake(scores))
        rows = {row["run"]: row for row in step2.table_rows(plan, fake(scores), Path("r"))}
        self.assertEqual(rows["plain_at lr 1e-5"]["noise_label"], "reference")
        self.assertEqual((rows["plain_at lr 3e-6"]["delta_vs_preferred"], rows["plain_at lr 3e-6"]["noise_label"]),
                         ("+3.00", "real gain"))
        self.assertEqual(rows["MPLC* screen s1"]["robust_score"], "41.00")
        self.assertEqual(rows["MPLC* s0"]["state"], "pending")
        self.assertEqual(rows["MPLC* s0"]["noise_label"], "")
        with tempfile.TemporaryDirectory() as tmp:
            step2.write_summary(plan, list(rows.values()), Path(tmp), STEP1)
            text = (Path(tmp) / "runs.md").read_text()
            self.assertIn("2.0 noise = 2.00", text)
            self.assertIn("plain_at: winner plain_at lr 3e-6", text)
            self.assertTrue((Path(tmp) / "runs.csv").is_file())


class DriverTests(unittest.TestCase):
    def dry_run(self, s1, s2, **env):
        return run_script(STEP2_SCRIPT, {"STEP1_ROOT": s1, "STEP2_ROOT": s2, **env}, ["--dry-run"])

    def test_dry_run_lists_pending_runs_in_order(self):
        with tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
            finish_step1(s1)
            result = self.dry_run(s1, s2)
            self.assertEqual(result.returncode, 0, result.stdout)
            commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
            self.assertEqual(len(commands), 15, result.stdout)
            self.assertIn("--save_dir=mplc_v2_supervlad_mplc_mixlinf_ep12_s1 ", commands[0] + " ")
            self.assertIn("--num_epochs=12", commands[0])
            self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0 ", commands[2] + " ")
            self.assertIn("--save_dir=mplc_v2_supervlad_mplc_mixlinf_s0 ", commands[3] + " ")
            self.assertNotIn("--num_epochs=12", commands[3])  # full length: the recipe's 100
            self.assertIn("--save_dir=mplc_v2_supervlad_plain_at_ep12_s0", commands[6])
            self.assertIn("--val_every=3", commands[6])
            self.assertFalse((Path(s2) / step2.CONFIG_NAME).exists())

    def test_stray_launcher_variable_is_ignored(self):
        with tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
            finish_step1(s1)
            result = self.dry_run(s1, s2, MPLC_V2_TAU="0.1", MPLC_V2_ARMS="fare")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("--listwise_tau=0.1", result.stdout)
            self.assertIn("--listwise_tau=0.05", result.stdout)
            self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0", result.stdout)

    def test_unfinished_step1_exits_3(self):
        with tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
            result = self.dry_run(s1, s2)
            self.assertEqual(result.returncode, 3, result.stdout)
            self.assertIn("mplc_v2_step1.sh", result.stdout)

    def test_summary_only_writes_tables(self):
        with tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
            finish_step1(s1)
            records, state = finished(45.0)
            write_run(Path(s2) / "mplc_v2_supervlad_plain_at_ep12_s0", state=state, records=records)
            result = run_script(STEP2_SCRIPT, {"STEP1_ROOT": s1, "STEP2_ROOT": s2}, ["--summary-only"])
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("| 2.2 screen | plain_at lr 1e-5 | completed", (Path(s2) / "summary" / "runs.md").read_text())

    def test_unknown_option_exits_2(self):
        self.assertEqual(run_script(STEP2_SCRIPT, args=["--bogus"]).returncode, 2)


if __name__ == "__main__":
    unittest.main()
