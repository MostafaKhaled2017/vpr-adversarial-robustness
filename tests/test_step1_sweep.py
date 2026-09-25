import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import step1_sweep as sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_train.sh"
STEP1_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_step1.sh"
EPOCHS = 12
SIZE = ["--batch-size", "24", "--batches-per-epoch", "200", "--val-every", "3"]


def record(epoch, clean, robust, budgets=(1.0, 3.0, 5.0)):
    return {"epoch": epoch, "clean": {"R@1": clean}, "robust_score": robust, "eligible_budgets": list(budgets)}


def write_run(run_dir, state="completed", records=()):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (run_dir / "validation_recalls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def finished(robust, clean=89.0, state="completed"):
    """A finished screen whose budget-5 checkpoint scores ``robust`` (None = no eligible epoch)."""
    budgets = (5.0,) if robust is not None else ()
    return [record(-1, 90.0, 10.0), record(0, clean, 0.0 if robust is None else robust, budgets)], state


def fake_results(scores):
    """results(cfg) from {run_name: robust or None (loss) or 'collapse'}; unlisted = unfinished."""

    def results(cfg):
        key = sweep.run_name(cfg, EPOCHS)
        if key not in scores:
            return None
        value = scores[key]
        if value == "collapse":
            return sweep.ScreenResult("collapse_aborted", 1, 90.0, 0, 89.0, 50.0)
        if value is None:
            return sweep.ScreenResult("completed", 9, 90.0)
        return sweep.ScreenResult("completed", 9, 90.0, 3, 89.0, float(value))

    return results


def name(**overrides):
    return sweep.run_name(sweep.config(**overrides), EPOCHS)


def run_script(script, env=None, args=()):
    environment = os.environ.copy()
    environment.update({"PYTHON": sys.executable})
    environment.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args], cwd=REPO_ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


class ReadScreenTests(unittest.TestCase):
    def test_budget_checkpoint_ignores_initial_and_prefers_robust_then_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, records=[
                record(-1, 90.0, 99.0),                      # initial: never selectable
                record(2, 88.0, 40.0),                       # --val_every 2: epochs 2, 4, 5
                record(4, 86.0, 60.0, budgets=(5.0,)),       # over the 3-point budget
                record(5, 88.5, 40.0),                       # same robust, better clean
                record(2, 88.0, 30.0),                       # resumed epoch 2 replaces the first
            ])
            result = sweep.read_screen(Path(tmp), 3.0)
        self.assertEqual((result.epoch, result.clean, result.robust), (5, 88.5, 40.0))
        self.assertEqual(result.initial_clean, 90.0)
        self.assertEqual(result.epochs_run, 5)
        self.assertFalse(result.is_loss)

    def test_no_eligible_epoch_and_collapse_are_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp) / "a", records=[record(-1, 90.0, 10.0), record(0, 80.0, 70.0, budgets=(5.0,))])
            write_run(Path(tmp) / "b", state="collapse_aborted", records=[record(-1, 90.0, 10.0), record(0, 89.0, 70.0)])
            self.assertTrue(sweep.read_screen(Path(tmp) / "a", 3.0).is_loss)
            self.assertTrue(sweep.read_screen(Path(tmp) / "b", 3.0).is_loss)

    def test_unfinished_or_missing_screen_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp) / "running", state="running", records=[record(-1, 90.0, 10.0)])
            self.assertIsNone(sweep.read_screen(Path(tmp) / "running", 3.0))
            self.assertIsNone(sweep.read_screen(Path(tmp) / "missing", 3.0))


class DecisionTests(unittest.TestCase):
    def test_label_boundaries(self):
        self.assertEqual(sweep.label(2.0, 2.0), "within noise")
        self.assertEqual(sweep.label(2.01, 2.0), "real gain")
        self.assertEqual(sweep.label(-2.01, 2.0), "real loss")
        self.assertEqual(sweep.label(None, 2.0), "n/a")

    def test_choose_keeps_preferred_within_noise_and_picks_best_real_gain(self):
        a, b, c = sweep.config(aw="0"), sweep.config(), sweep.config(aw="10")
        within = fake_results({name(aw="0"): 52.0, name(): 50.0, name(aw="10"): 51.0})
        self.assertEqual(sweep.choose([a, b, c], b, within, noise=2.0), b)
        gains = fake_results({name(aw="0"): 53.0, name(): 50.0, name(aw="10"): 55.0})
        self.assertEqual(sweep.choose([a, b, c], b, gains, noise=2.0), c)

    def test_losses_never_win_and_a_lost_preferred_falls_back_to_best(self):
        a, b = sweep.config(aw="0"), sweep.config()
        results = fake_results({name(aw="0"): "collapse", name(): None})
        self.assertIsNone(sweep.choose([a, b], b, results, noise=1.0))
        results = fake_results({name(aw="0"): 30.0, name(): None})
        self.assertEqual(sweep.choose([a, b], b, results, noise=1.0), a)


class PlanSweepTests(unittest.TestCase):
    NOISE_OK = {name(seed="0"): 50.0, name(seed="1"): 51.0, name(seed="2"): 49.0}  # noise 2

    def test_starts_with_the_three_noise_seeds(self):
        state = sweep.plan_sweep(fake_results({}))
        self.assertEqual([c["seed"] for c in state.pending], ["0", "1", "2"])

    def test_noise_above_limit_stops_unless_accepted(self):
        scores = {name(seed="0"): 50.0, name(seed="1"): 54.0, name(seed="2"): 49.0}
        self.assertIn("above 3", sweep.plan_sweep(fake_results(scores)).stop_reason)
        state = sweep.plan_sweep(fake_results(scores), accept_noise=True)
        self.assertEqual(state.pending, [sweep.config(mix="linf")])

    def test_full_walk_carries_winners_and_extends_grid_edges(self):
        scores = dict(self.NOISE_OK)
        scores[name(mix="linf")] = 49.0  # within noise of 'all' -> linf preferred, wins
        grid = {("8", "1e-5"): 49.0, ("8", "3e-6"): 50.0, ("4", "1e-5"): 50.0, ("4", "3e-6"): 51.0,
                ("0", "1e-5"): 52.0, ("0", "3e-6"): 56.0}
        for (fte, lr), score in grid.items():
            scores[name(mix="linf", fte=fte, lr=lr)] = score
        state = sweep.plan_sweep(fake_results(scores))
        self.assertEqual(state.pending, [sweep.config(mix="linf", fte="0", lr="1e-6")])  # grid edge

        scores[name(mix="linf", fte="0", lr="1e-6")] = 55.0
        base = dict(mix="linf", fte="0", lr="3e-6")
        scores.update({name(aw="0", **base): "collapse", name(aw="0.1", **base): 57.0, name(aw="10", **base): 60.0})
        state = sweep.plan_sweep(fake_results(scores))
        self.assertEqual(state.pending, [sweep.config(aw="30", **base)])  # grid edge

        scores[name(aw="30", **base)] = 58.0
        scores[name(aw="10", mix="all", fte="0", lr="3e-6")] = 63.0  # the re-check flips the mix
        state = sweep.plan_sweep(fake_results(scores))
        self.assertEqual(state.final, sweep.config(aw="10", mix="all", fte="0", lr="3e-6"))
        self.assertEqual([s.name for s in state.stages], ["1a", "1b", "1c", "1d", "1e"])
        self.assertEqual(state.noise, 2.0)


class OutputTests(unittest.TestCase):
    def build_finished_sweep(self, root):
        scores = {name(seed="0"): 50.0, name(seed="1"): 51.0, name(seed="2"): 49.0, name(mix="linf"): 49.5}
        for fte in ("8", "4", "0"):
            for lr in ("1e-5", "3e-6"):
                scores.setdefault(name(mix="linf", fte=fte, lr=lr), 50.0 if (fte, lr) != ("4", "1e-5") else 54.0)
        base = dict(mix="linf", fte="4", lr="1e-5")
        scores.update({name(aw="0", **base): None, name(aw="0.1", **base): 53.0, name(aw="10", **base): 53.5})
        scores[name(mix="all", fte="4", lr="1e-5")] = 52.0
        for run, robust in scores.items():
            records, state = finished(robust)
            write_run(root / run, state=state, records=records)
        return base

    def test_summarize_writes_tables_env_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.build_finished_sweep(root)
            sweep.main(["summarize", "--root", str(root), "--epochs", str(EPOCHS), *SIZE])
            out = root / "summary"
            self.assertEqual((out / "mplc_star.env").read_text().strip(), "MPLC_V2_ATTACK_MIX=linf MPLC_V2_FREEZE_TE=4")
            rows = (out / "screens.csv").read_text().splitlines()
            self.assertIn("real gain", "\n".join(rows))
            self.assertIn("loss", "\n".join(rows))
            for figure in ("fig_sensitivity", "fig_tradeoff", "fig_training_curves"):
                self.assertTrue((out / f"{figure}.pdf").is_file(), figure)
                self.assertTrue((out / f"{figure}.png").is_file(), figure)
            self.assertIn("MPLC", (out / "decisions.md").read_text())

            final_dir = root / sweep.run_name(sweep.config(**base), EPOCHS)
            (final_dir / "last_model.pth").write_bytes(b"x")
            (root / name(seed="1") / "last_model.pth").write_bytes(b"x")
            sweep.main(["prune", "--root", str(root), "--epochs", str(EPOCHS), *SIZE])
            self.assertTrue((final_dir / "last_model.pth").is_file())
            self.assertFalse((root / name(seed="1") / "last_model.pth").exists())

    def test_settings_are_frozen_after_first_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep.main(["next", "--root", str(root), "--epochs", "6", *SIZE, "--no-freeze"])
            self.assertFalse((root / sweep.CONFIG_NAME).exists())
            sweep.main(["next", "--root", str(root), "--epochs", "10", *SIZE])
            self.assertIn("budget: 5.0", (root / sweep.CONFIG_NAME).read_text())
            self.assertIn("batches_per_epoch: 200", (root / sweep.CONFIG_NAME).read_text())
            with self.assertRaises(SystemExit):
                sweep.main(["next", "--root", str(root), "--epochs", "6", *SIZE])
            for index, value in ((1, "16"), (3, "400"), (5, "1")):
                changed = list(SIZE)
                changed[index] = value
                with self.subTest(changed=changed), self.assertRaises(SystemExit):
                    sweep.main(["next", "--root", str(root), "--epochs", "10", *changed])


class LauncherResumableModeTests(unittest.TestCase):
    def dry_run(self, root, **env):
        return run_script(
            TRAIN_SCRIPT,
            {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_ARMS": "mplc", "MPLC_V2_NUM_EPOCHS": str(EPOCHS), "MPLC_V2_RUN_ROOT": str(root), **env},
        )

    def test_python_run_names_match_the_launcher(self):
        cases = [{}, {"MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_FREEZE_TE": "0", "MPLC_V2_LR": "3e-6",
                      "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_SEED": "2"}]
        with tempfile.TemporaryDirectory() as tmp:
            for env in cases:
                cfg = sweep.config(**{k: env[v] for k, v in sweep.ENV_NAMES.items() if v in env})
                result = self.dry_run(tmp, **env)
                self.assertIn(f"--save_dir={sweep.run_name(cfg, EPOCHS)} ", result.stdout + " ")

    def test_pending_resumable_finished_and_invalid_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "mplc_v2_supervlad_mplc_ep12_s0"

            result = self.dry_run(root)
            self.assertIn(f"--run_dir={run_dir}", result.stdout)
            self.assertIn("--resume_model_only", result.stdout)

            run_dir.mkdir()
            (run_dir / "initial_model.pth").write_bytes(b"x")
            result = self.dry_run(root)
            self.assertIn("=== RESTART", result.stdout)
            self.assertTrue(run_dir.is_dir(), "a dry run must not move the directory")

            (run_dir / "last_model.pth").write_bytes(b"x")
            result = self.dry_run(root)
            self.assertIn("=== RESUME", result.stdout)
            self.assertIn(f"--resume={run_dir / 'last_model.pth'} --continue", result.stdout)
            self.assertNotIn("--resume_model_only", result.stdout)
            self.assertIn(f"--run_dir={run_dir}", result.stdout)

            (run_dir / "run_status.json").write_text(json.dumps({"state": "completed"}))
            result = self.dry_run(root)
            self.assertIn("=== SKIP", result.stdout)
            self.assertNotIn("+ ", result.stdout)

    def test_run_at_another_batch_size_is_neither_skipped_nor_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "mplc_v2_supervlad_mplc_ep12_s0"
            run_dir.mkdir()
            (run_dir / "last_model.pth").write_bytes(b"x")
            (run_dir / "training_config.yaml").write_text("train_batch_size: 16\n")
            result = self.dry_run(root)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("trained with train_batch_size 16", result.stdout)

            (run_dir / "run_status.json").write_text(json.dumps({"state": "completed"}))
            self.assertEqual(self.dry_run(root).returncode, 1)

            (run_dir / "training_config.yaml").write_text("train_batch_size: 24\nbatches_per_epoch: 400\n")
            result = self.dry_run(root)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("trained with batches_per_epoch 400", result.stdout)

            (run_dir / "training_config.yaml").write_text("train_batch_size: 24\nbatches_per_epoch: 200\n")
            self.assertIn("=== SKIP", self.dry_run(root).stdout)

    def test_recipe_uses_the_single_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.dry_run(tmp)
        self.assertIn("--batch_size=24", result.stdout)
        self.assertNotIn("--batch_size=16", result.stdout)
        self.assertIn("--batches_per_epoch=200", result.stdout)
        self.assertNotIn("--batches_per_epoch=400", result.stdout)


class DriverTests(unittest.TestCase):
    def test_dry_run_shows_noise_screens_then_moves_to_the_mix_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"STEP1_ROOT": str(root)}
            result = run_script(STEP1_SCRIPT, env, ["--dry-run"])
            self.assertEqual(result.returncode, 0, result.stdout)
            for seed in "012":
                self.assertIn(f"--save_dir=mplc_v2_supervlad_mplc_ep12_s{seed}", result.stdout)
            commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
            self.assertEqual(len(commands), 3, result.stdout)
            for command in commands:
                self.assertIn("--num_epochs=12", command)
                self.assertIn("--val_every=3", command)
            self.assertFalse((root / sweep.CONFIG_NAME).exists())

            for seed, robust in zip("012", (50.0, 51.0, 49.0)):
                records, state = finished(robust)
                write_run(root / f"mplc_v2_supervlad_mplc_ep12_s{seed}", state=state, records=records)
            result = run_script(STEP1_SCRIPT, env, ["--dry-run"])
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("--save_dir=mplc_v2_supervlad_mplc_mixlinf_ep12_s0", result.stdout)
            self.assertNotIn("PerceptualPGDAttack", result.stdout)

    def test_noise_stop_exits_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, robust in zip("012", (50.0, 56.0, 49.0)):
                records, state = finished(robust)
                write_run(root / f"mplc_v2_supervlad_mplc_ep12_s{seed}", state=state, records=records)
            result = run_script(STEP1_SCRIPT, {"STEP1_ROOT": str(root)}, ["--dry-run"])
            self.assertEqual(result.returncode, 3, result.stdout)
            self.assertIn("noise", result.stdout)

    def test_unknown_option_exits_2(self):
        self.assertEqual(run_script(STEP1_SCRIPT, args=["--bogus"]).returncode, 2)


if __name__ == "__main__":
    unittest.main()
