"""Run-skipping, resuming and argument checks of the scripts in scripts/."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments  # noqa: E402


def run_script(root, script, *args, **extra_environment):
    environment = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable,
                   "RUN_ROOT": str(root / "runs"), "OUT_ROOT": str(root / "out")}
    environment.pop("BANK_ROOT", None)
    environment.pop("STOP_AT", None)
    environment.update(extra_environment)
    return subprocess.run(["bash", str(REPO_ROOT / "scripts" / script), *args], cwd=REPO_ROOT, env=environment,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def commands(result):
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


COMMON_FLAGS = (
    "--model=supervlad --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth --backbone=dino "
    "--supervlad_clusters=4 --crossimage_encoder --train_resize 322 322 --resize 322 322 --freeze_te=0 "
    "--grad_checkpointing --eval_datasets_folder=datasets --gsv_cities_base_path=datasets/gsv_cities "
    "--eval_dataset_name=msls --optim=adam --lr_schedule=120 --lr_plateau_factor=0.1 --batch_size=20 "
    "--batches_per_epoch=200 --infer_batch_size=16 --mixed_precision --test_method=hard_resize "
    "--randomize_attack --adv_negatives=5 --adv_warmup_epochs=1 --adv_margin=0.1 --adv_loss_weight=1.0 "
    "--adv_align_weight=0.05 --early_stop_min_delta=0.0 --keep_every=0 --val_every=3 "
    "--val_rank_epsilons 0.0342 0.0685 --selection_clean_budgets 1 3 5 7 10 --lr=1e-5 --num_epochs=100 "
    "--patience=6 --lr_plateau_patience=3 --resume=checkpoints/SuperVLAD_base.pth --resume_model_only "
    "--run_dir={run_root}/x --seed=0 --save_dir=x "
)
EXPECTED_COMMANDS = {
    "araft": COMMON_FLAGS + "--attack 'RankLinfAttack(model, epsilon=0.0685, steps=5)' --defense_loss=listwise "
    "--listwise_tau=0.05 --listwise_k=1 --attack_ramp_epochs=5 --collapse_abort_knn=0.15 --align_target=initial "
    "--adv_align_weight=1.0",
    "fare": COMMON_FLAGS + "--attack 'EmbeddingShiftLinfAttack(model, epsilon=0.0685, steps=5)' --adv_loss_weight=0 "
    "--align_target=initial --adv_align_weight=1.0 --attack_ramp_epochs=5 --collapse_abort_knn=0.15",
}


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_finished_run_is_skipped(self):
        run_dir = self.root / "runs" / "supervlad_araft_s0"
        run_dir.mkdir(parents=True)
        (run_dir / "run_status.json").write_text(json.dumps({"state": "early_stopped"}))
        result = run_script(self.root, "train.sh", "supervlad_araft_s0", "supervlad", "araft", "0")
        self.assertIn("=== SKIP supervlad_araft_s0", result.stdout)
        self.assertEqual(commands(result), [])

    def test_stopped_run_resumes_from_last_checkpoint(self):
        for method in ("araft", "fare", "plain_at"):
            with self.subTest(method=method):
                name = f"supervlad_{method}_s0"
                run_dir = self.root / "runs" / name
                run_dir.mkdir(parents=True)
                (run_dir / "run_status.json").write_text(json.dumps({"state": "paused"}))
                (run_dir / "last_model.pth").touch()
                result = run_script(self.root, "train.sh", name, "supervlad", method, "0")
                [argv] = commands(result)
                args = parse_arguments(argv[2:])
                self.assertEqual(args.resume, str(run_dir / "last_model.pth"))
                self.assertTrue(args.continue_training)

    def test_fare_trains_anchor_only_against_embedding_shift(self):
        result = run_script(self.root, "train.sh", "supervlad_fare_s0", "supervlad", "fare", "0")
        [argv] = commands(result)
        args = parse_arguments(argv[2:])
        self.assertEqual(args.attack, ["EmbeddingShiftLinfAttack(model, epsilon=0.0685, steps=5)"])
        self.assertEqual(args.adv_loss_weight, 0.0)
        self.assertEqual(args.adv_align_weight, 1.0)
        self.assertEqual(args.align_target, "initial")
        self.assertEqual(args.attack_ramp_epochs, 5)
        self.assertEqual(args.collapse_abort_knn, 0.15)
        self.assertFalse(args.multi_positive)

    def test_fare_rejects_araft_only_settings(self):
        for setting in ("LOSS=hinge", "POSITIVE=closest"):
            result = run_script(self.root, "train.sh", "x", "supervlad", "fare", "0", setting)
            self.assertNotEqual(result.returncode, 0, setting)

    def test_plain_at_has_no_anchor(self):
        result = run_script(self.root, "train.sh", "supervlad_plain_at_s0", "supervlad", "plain_at", "0")
        [argv] = commands(result)
        args = parse_arguments(argv[2:])
        self.assertEqual(args.attack, ["RankLinfAttack(model, epsilon=0.0685, steps=5)"])
        self.assertEqual(args.adv_align_weight, 0.0)
        self.assertEqual(args.align_target, "current")
        self.assertEqual(args.defense_loss, "listwise")
        self.assertEqual(args.attack_ramp_epochs, 5)
        self.assertEqual(args.collapse_abort_knn, 0.15)

    def test_settings_reach_their_flags(self):
        bank = "logs/banks/supervlad_f0.pt"
        cases = (
            (("EPOCHS=12",), {"epochs_num": 12}),
            (("STOP_AT=2099-01-01 08:00",), {"stop_by": datetime(2099, 1, 1, 8, 0).timestamp()}),
            (("LRW=3",), {"adv_loss_weight": 3.0}),
            (("ATTACK=shift",), {"attack": ["EmbeddingShiftLinfAttack(model, epsilon=0.0685, steps=5)"]}),
            (("ATTACK=combined", "BETA=0.5"),
             {"attack": ["CombinedShiftRankLinfAttack(model, epsilon=0.0685, steps=5, rank_weight=0.5)"]}),
            (("DRIFT=1",), {"clean_drift_weight": 1.0}),
            (("EMA=0.999",), {"ema_decay": 0.999}),
            (("BANK=1",), {"negative_bank": bank}),
            (("BANK=1", "NEG_BANK=8"), {"bank_negatives": 8}),
            (("BANK=1", "BANK_DEVICE=cpu"), {"bank_device": "cpu"}),
            (("BANK=1", "ANCHOR=rel"), {"anchor_kind": "relational"}),
            (("BANK=1", "ANCHOR=both", "REL_M=16", "REL_T=0.5"),
             {"anchor_kind": "both", "relational_neighbours": 16, "relational_temperature": 0.5}),
        )
        for settings, expected in cases:
            with self.subTest(settings=settings):
                result = run_script(self.root, "train.sh", "x", "supervlad", "araft", "0", *settings)
                self.assertEqual(result.returncode, 0, result.stdout)
                [argv] = commands(result)
                args = parse_arguments(argv[2:])
                for name, value in expected.items():
                    self.assertEqual(getattr(args, name), value, name)

    def test_ema_applies_to_every_method(self):
        for method in ("araft", "fare", "plain_at"):
            with self.subTest(method=method):
                result = run_script(self.root, "train.sh", "x", "supervlad", method, "0", "EMA=0.99")
                [argv] = commands(result)
                self.assertEqual(parse_arguments(argv[2:]).ema_decay, 0.99)

    def test_settings_rejected_for_the_wrong_method(self):
        cases = (
            ("plain_at", "AW=2"), ("plain_at", "ATTACK=shift"), ("plain_at", "DRIFT=1"),
            ("fare", "BANK=1"), ("fare", "LRW=2"), ("fare", "ANCHOR=rel"),
            ("araft", "ANCHOR=rel"), ("araft", "NEG_BANK=3"), ("araft", "BETA=2"), ("araft", "REL_T=0.5"),
        )
        for method, setting in cases:
            with self.subTest(method=method, setting=setting):
                result = run_script(self.root, "train.sh", "x", "supervlad", method, "0", setting)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_stop_at_must_be_a_future_time(self):
        for setting in ("STOP_AT=2000-01-01 08:00", "STOP_AT=not a time"):
            with self.subTest(setting=setting):
                result = run_script(self.root, "train.sh", "x", "supervlad", "fare", "0", setting)
                self.assertEqual(result.returncode, 2, result.stdout)

    def test_stop_at_is_read_from_the_environment_unless_given(self):
        cases = (((), datetime(2099, 1, 1, 8, 0)), (("STOP_AT=2098-06-01 09:30",), datetime(2098, 6, 1, 9, 30)))
        for settings, expected in cases:
            with self.subTest(settings=settings):
                result = run_script(self.root, "train.sh", "x", "supervlad", "araft", "0", *settings,
                                    STOP_AT="2099-01-01 08:00")
                [argv] = commands(result)
                self.assertEqual(parse_arguments(argv[2:]).stop_by, expected.timestamp())

    def test_no_settings_prints_todays_command(self):
        for method, expected in EXPECTED_COMMANDS.items():
            with self.subTest(method=method):
                result = run_script(self.root, "train.sh", "x", "supervlad", method, "0")
                [argv] = commands(result)
                self.assertEqual(argv[1:], ["train.py", *shlex.split(expected.format(run_root=self.root / "runs"))])

    def test_done_piece_is_skipped(self):
        piece = self.root / "out" / "supervlad" / "apgd" / "sped" / "araft"
        result_dir = piece / "2026-01-01_00-00-00"
        result_dir.mkdir(parents=True)
        for name in ("rank_eval_results.json", "rank_eval_results.csv", "per_query_ranks.csv"):
            (result_dir / name).touch()
        (piece / "DONE").write_text(json.dumps({"result_dir": str(result_dir)}))
        result = run_script(self.root, "eval.sh", "supervlad", "apgd", "sped", "araft", "x.pth", "0.0685")
        self.assertIn("=== SKIP", result.stdout)
        self.assertEqual(commands(result), [])

    def test_eval_wrapper_tag_picks_results_dir_and_run(self):
        for tag in ("araft", "fare"):
            with self.subTest(tag=tag):
                environment = {} if tag == "araft" else {"TAG": tag}
                result = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "eval_boq.sh")], cwd=REPO_ROOT,
                                        env={**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable,
                                             "RUN_ROOT": str(self.root / "runs"),
                                             "OUT_ROOT": str(self.root / "out"), **environment},
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(len(commands(result)), 8)
                self.assertIn(str(self.root / "out" / "boq" / "apgd" / "msls" / tag), result.stdout)
                self.assertIn(str(self.root / "runs" / f"boq_{tag}_s0" / "best_model.pth"), result.stdout)

    def test_unknown_model_or_method_fails(self):
        for model, method in (("supervlad", "fgsm"), ("mixvpr", "araft")):
            result = run_script(self.root, "train.sh", "x", model, method, "0")
            self.assertNotEqual(result.returncode, 0, (model, method))


    def test_build_bank_dry_run_prints_module_command(self):
        bank_root = str(self.root / "banks")
        result = run_script(self.root, "build_bank.sh", "supervlad", BANK_ROOT=bank_root)
        self.assertEqual(result.returncode, 0, result.stdout)
        [argv] = commands(result)
        self.assertEqual(argv[1:3], ["-m", "src.negative_bank"])
        self.assertIn(f"--out={bank_root}/supervlad_f0.pt", argv)
        self.assertIn("--resume=checkpoints/SuperVLAD_base.pth", argv)
        self.assertIn("--model=supervlad", argv)

    def test_build_bank_skips_existing_file(self):
        bank = self.root / "banks" / "supervlad_f0.pt"
        bank.parent.mkdir(parents=True)
        bank.touch()
        result = run_script(self.root, "build_bank.sh", "supervlad", BANK_ROOT=str(bank.parent))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("=== SKIP", result.stdout)
        self.assertEqual(commands(result), [])

    def soup_sweep(self, *args, **environment):
        return run_script(self.root, "soup_sweep.sh", "supervlad", "mix", "a.pth", "b.pth", *args,
                          SOUP_ROOT=str(self.root / "soups"), **environment)

    def test_soup_sweep_dry_run_prints_soup_screen_and_cleanup_per_alpha(self):
        result = self.soup_sweep("0.3", "0.5")
        self.assertEqual(result.returncode, 0, result.stdout)
        argvs = commands(result)
        self.assertEqual(len(argvs), 6)
        for offset, alpha in ((0, "0.3"), (3, "0.5")):
            soup, screen, remove = argvs[offset:offset + 3]
            path = str(self.root / "soups" / f"mix_a{alpha}.pth")
            self.assertEqual(soup[1:], ["scripts/soup.py", "a.pth", "b.pth", alpha, path])
            self.assertEqual(screen[1:4], ["scripts/validate_only.py", f"mix_a{alpha}", "--initial_clean_r1"])
            self.assertEqual(screen[4], "88.10")
            args = parse_arguments(screen[5:])
            self.assertEqual(args.resume, path)
            self.assertEqual(args.run_dir, str(self.root / "runs" / f"mix_a{alpha}"))
            self.assertEqual(args.infer_batch_size, 16)
            self.assertFalse(args.is_clean_only)
            self.assertEqual(remove, ["rm", "-f", path])
        self.assertFalse((self.root / "soups").exists())
        self.assertFalse((self.root / "runs").exists())

    def test_soup_sweep_boq_and_keep_soup(self):
        result = run_script(self.root, "soup_sweep.sh", "boq", "mix", "a.pth", "b.pth", "0.3", KEEP_SOUP="1")
        self.assertEqual(result.returncode, 0, result.stdout)
        soup, screen = commands(result)
        self.assertEqual(soup[1], "scripts/soup.py")
        self.assertEqual(screen[4], "91.35")
        args = parse_arguments(screen[5:])
        self.assertEqual(args.model, "boq")
        self.assertEqual(args.infer_batch_size, 8)

    def test_soup_sweep_stops_before_a_screen_that_ends_too_late(self):
        soon = datetime.fromtimestamp(datetime.now().timestamp() + 120).strftime("%Y-%m-%d %H:%M")
        for settings, environment in (((f"STOP_AT={soon}",), {}), ((), {"STOP_AT": soon})):
            with self.subTest(settings=settings):
                result = self.soup_sweep("0.3", "0.5", *settings, **environment)
                self.assertEqual(result.returncode, 5, result.stdout)
                self.assertEqual(commands(result), [])
                self.assertIn("0.3 0.5", result.stdout)

    def test_soup_sweep_rejects_a_past_stop_at(self):
        result = self.soup_sweep("0.3", "STOP_AT=2000-01-01 08:00")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("has passed", result.stdout)

    def test_soup_sweep_skips_a_finished_screen(self):
        run_dir = self.root / "runs" / "mix_a0.3"
        run_dir.mkdir(parents=True)
        (run_dir / "run_status.json").write_text(json.dumps({"state": "completed"}))
        result = self.soup_sweep("0.3", "0.5")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("=== SKIP mix_a0.3", result.stdout)
        self.assertEqual([argv[4] for argv in commands(result) if argv[1] == "scripts/soup.py"], ["0.5"])

    def test_soup_sweep_rejects_unknown_model_and_bad_alpha(self):
        result = run_script(self.root, "soup_sweep.sh", "mixvpr", "mix", "a.pth", "b.pth", "0.3")
        self.assertEqual(result.returncode, 2, result.stdout)
        result = self.soup_sweep("half")
        self.assertEqual(result.returncode, 2, result.stdout)

    def test_train_bank_root_is_honoured(self):
        bank_root = str(self.root / "banks")
        result = run_script(self.root, "train.sh", "x", "supervlad", "araft", "0", "BANK=1", BANK_ROOT=bank_root)
        self.assertEqual(result.returncode, 0, result.stdout)
        [argv] = commands(result)
        self.assertEqual(parse_arguments(argv[2:]).negative_bank, f"{bank_root}/supervlad_f0.pt")


if __name__ == "__main__":
    unittest.main()
