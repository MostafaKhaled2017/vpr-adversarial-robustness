import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_train.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_eval.sh"


def run_script(script, extra_env=None, args=()):
    environment = os.environ.copy()
    environment.update({"MPLC_V2_DRY_RUN": "1", "PYTHON": sys.executable})
    if extra_env:
        environment.update(extra_env)
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class MplcV2TrainScriptTests(unittest.TestCase):
    def test_default_seed_trains_clean_ft_then_mplc(self):
        result = run_script(TRAIN_SCRIPT)

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--seed=0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_s0", result.stdout)
        self.assertIn("--validation_protocol=rank_pgd", result.stdout)
        self.assertIn("RankLinfAttack(model, epsilon=0.0685, steps=5)", result.stdout)

        commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        clean_ft_command = next(
            line for line in commands if "mplc_v2_supervlad_clean_ft_s0" in line
        )
        self.assertNotIn("--attack", clean_ft_command)

        mplc_command = next(line for line in commands if "mplc_v2_supervlad_mplc_s0" in line)
        for command in (clean_ft_command, mplc_command):
            self.assertIn("--freeze_te=8", command)
            self.assertIn("--selection_clean_budgets 1 3 5", command)
            self.assertIn("--val_rank_epsilons 0.0342 0.0685", command)
            self.assertNotIn("--grad_checkpointing", command)
        self.assertIn("--align_target=initial", mplc_command)
        self.assertIn("--adv_align_weight=1.0", mplc_command)
        self.assertNotIn("--align_target", clean_ft_command)

    def test_seed_override_trains_seed_one(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_SEED": "1"})

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--seed=1", result.stdout)
        self.assertIn("_s1", result.stdout)

    def test_unknown_arm_exits_with_status_2(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ARMS": "bogus"})

        self.assertEqual(result.returncode, 2, result.stdout)

    def test_finished_arm_is_skipped(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s99" / "x"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "early_stopped"}), encoding="utf-8"
        )
        try:
            result = run_script(
                TRAIN_SCRIPT, {"MPLC_V2_SEED": "99", "MPLC_V2_ARMS": "clean_ft"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("SKIP mplc_v2_supervlad_clean_ft_s99", result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s99", ignore_errors=True
            )

    def test_mplc_tau_override_gets_distinct_save_dir(self):
        result = run_script(
            TRAIN_SCRIPT, {"MPLC_V2_ARMS": "clean_ft mplc", "MPLC_V2_TAU": "0.01"}
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_tau0.01_s0", result.stdout)

    def test_mplc_override_run_is_not_skipped_by_default_finished_run(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s98" / "x"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "completed"}), encoding="utf-8"
        )
        try:
            result = run_script(
                TRAIN_SCRIPT,
                {"MPLC_V2_SEED": "98", "MPLC_V2_ARMS": "mplc", "MPLC_V2_TAU": "0.01"},
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("TRAIN mplc_v2_supervlad_mplc_tau0.01_s98", result.stdout)
            self.assertNotIn("SKIP", result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s98", ignore_errors=True
            )

    def test_freeze_te_override_names_both_arms_and_enables_checkpointing(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_FREEZE_TE": "0"})

        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertEqual(len(commands), 2, result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_fte0_s0", commands[0])
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_fte0_s0", commands[1])
        for command in commands:
            self.assertIn("--freeze_te=0", command)
            self.assertIn("--grad_checkpointing", command)

    def test_align_weight_and_eps_overrides_get_suffix(self):
        result = run_script(
            TRAIN_SCRIPT,
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_TRAIN_EPS": "0.137", "MPLC_V2_TAU": "0.01"},
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_tau0.01_aw10_eps0.137_s0", result.stdout)
        self.assertIn("RankLinfAttack(model, epsilon=0.137, steps=5)", result.stdout)
        self.assertIn("--adv_align_weight=10", result.stdout)

    def test_lr_and_epochs_override_both_arms_after_recipe(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_LR": "3e-6", "MPLC_V2_NUM_EPOCHS": "9", "MPLC_V2_FREEZE_TE": "4"})

        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertEqual(len(commands), 2, result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_fte4_lr3e-6_ep9_s0", commands[0])
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_fte4_lr3e-6_ep9_s0", commands[1])
        for command in commands:
            # argparse keeps the last value, so the override must follow the recipe's default.
            self.assertGreater(command.rindex("--lr=3e-6"), command.index("--lr=1e-5"))
            self.assertGreater(command.rindex("--num_epochs=9"), command.index("--num_epochs=100"))

    def test_default_lr_and_epochs_add_no_flags(self):
        result = run_script(TRAIN_SCRIPT)

        self.assertEqual(result.returncode, 0, result.stdout)
        for command in [line for line in result.stdout.splitlines() if line.startswith("+ ")]:
            self.assertEqual(command.count("--lr="), 1)
            self.assertEqual(command.count("--num_epochs="), 1)
            self.assertNotIn("--val_every", command)

    def test_val_every_applies_to_both_arms(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_VAL_EVERY": "2"})

        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertEqual(len(commands), 2, result.stdout)
        for command in commands:
            self.assertIn("--val_every=2", command)

    def test_linf_attack_mix_keeps_only_rank_attack(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ATTACK_MIX": "linf"})

        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0", commands[0])
        self.assertNotIn("--attack", commands[0])
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_mixlinf_s0", commands[1])
        self.assertEqual(commands[1].count("--attack "), 1)
        self.assertIn("RankLinfAttack(model, epsilon=0.0685, steps=5)", commands[1])
        self.assertNotIn("Perceptual", commands[1])

    def test_unknown_attack_mix_exits_with_status_2(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ATTACK_MIX": "l2"})

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("unknown MPLC_V2_ATTACK_MIX", result.stdout)

    def test_defense_loss_and_single_positive_ablations_get_suffix(self):
        result = run_script(
            TRAIN_SCRIPT,
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_DEFENSE_LOSS": "hinge", "MPLC_V2_MULTI_POSITIVE": "0"},
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        (command,) = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_hinge_sp_s0", command)
        self.assertIn("--defense_loss=hinge", command)
        self.assertNotIn("--multi_positive", command)

    def test_default_mplc_is_listwise_multi_positive(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ARMS": "mplc"})

        self.assertEqual(result.returncode, 0, result.stdout)
        (command,) = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_s0", command)
        self.assertIn("--defense_loss=listwise", command)
        self.assertIn("--multi_positive", command)

    def test_unknown_defense_loss_or_multi_positive_exits_with_status_2(self):
        for env in ({"MPLC_V2_DEFENSE_LOSS": "triplet"}, {"MPLC_V2_MULTI_POSITIVE": "yes"}):
            result = run_script(TRAIN_SCRIPT, env)
            self.assertEqual(result.returncode, 2, result.stdout)

    def test_plain_at_arm_is_hinge_rank_linf_without_anchor(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ARMS": "plain_at"})

        self.assertEqual(result.returncode, 0, result.stdout)
        (command,) = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertIn("--save_dir=mplc_v2_supervlad_plain_at_s0", command)
        self.assertEqual(command.count("--attack "), 1)
        self.assertIn("RankLinfAttack(model, epsilon=0.0685, steps=5)", command)
        self.assertIn("--defense_loss=hinge", command)
        self.assertIn("--adv_align_weight=0", command)
        self.assertIn("--attack_ramp_epochs=5", command)
        self.assertNotIn("--multi_positive", command)
        self.assertNotIn("--align_target", command)

    def test_fare_arm_trains_anchor_only_against_embedding_shift(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ARMS": "fare"})

        self.assertEqual(result.returncode, 0, result.stdout)
        (command,) = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertIn("--save_dir=mplc_v2_supervlad_fare_s0", command)
        self.assertEqual(command.count("--attack "), 1)
        self.assertIn("EmbeddingShiftLinfAttack(model, epsilon=0.0685, steps=5)", command)
        self.assertIn("--adv_loss_weight=0", command)
        self.assertIn("--align_target=initial", command)
        self.assertIn("--adv_align_weight=1.0", command)

    def test_baseline_names_carry_only_the_settings_they_read(self):
        result = run_script(
            TRAIN_SCRIPT,
            {"MPLC_V2_ARMS": "plain_at fare", "MPLC_V2_LR": "3e-6", "MPLC_V2_NUM_EPOCHS": "9",
             "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_TAU": "0.01"},
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_plain_at_lr3e-6_ep9_s0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_fare_aw10_lr3e-6_ep9_s0", result.stdout)


class MplcV2EvalScriptTests(unittest.TestCase):
    def test_explicit_models_and_single_dataset(self):
        result = run_script(
            EVAL_SCRIPT,
            {
                "MPLC_V2_MODELS": "a=checkpoints/SuperVLAD_base.pth",
                "MPLC_V2_DATASETS": "sped",
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        eval_lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("+ ") and "eval.py" in line
        ]
        self.assertEqual(len(eval_lines), 1, result.stdout)
        self.assertIn("--model_tags a", eval_lines[0])
        self.assertIn("--rank_steps=20", eval_lines[0])
        self.assertIn("--datasets sped", eval_lines[0])

    def test_interrupted_run_without_status_is_not_evaluated(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s97" / "run_a"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
        try:
            result = run_script(
                EVAL_SCRIPT, {"MPLC_V2_SEEDS": "97", "MPLC_V2_DATASETS": "sped"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "WARNING: no finished checkpoint for mplc_v2_supervlad_clean_ft_s97",
                result.stdout,
            )
            self.assertNotIn(str(run_dir), result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s97", ignore_errors=True
            )

    def test_running_state_run_is_not_evaluated(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s96" / "run_a"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "running"}), encoding="utf-8"
        )
        try:
            result = run_script(
                EVAL_SCRIPT, {"MPLC_V2_SEEDS": "96", "MPLC_V2_DATASETS": "sped"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "WARNING: no finished checkpoint for mplc_v2_supervlad_mplc_s96",
                result.stdout,
            )
            self.assertNotIn(str(run_dir), result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s96", ignore_errors=True
            )

    def test_finished_run_is_evaluated_and_newest_wins(self):
        base = REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s95"
        old_dir = base / "2024-01-01_run"
        new_dir = base / "2024-01-02_run"
        for run_dir, state in ((old_dir, "completed"), (new_dir, "early_stopped")):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
            (run_dir / "run_status.json").write_text(
                json.dumps({"state": state}), encoding="utf-8"
            )
        try:
            result = run_script(
                EVAL_SCRIPT, {"MPLC_V2_SEEDS": "95", "MPLC_V2_DATASETS": "sped"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            eval_lines = [
                line
                for line in result.stdout.splitlines()
                if line.startswith("+ ") and "eval.py" in line
            ]
            self.assertEqual(len(eval_lines), 1, result.stdout)
            self.assertIn(
                f"{new_dir.relative_to(REPO_ROOT)}/best_model.pth", eval_lines[0]
            )
            self.assertNotIn(
                f"{old_dir.relative_to(REPO_ROOT)}/best_model.pth", eval_lines[0]
            )
        finally:
            shutil.rmtree(base, ignore_errors=True)


def eval_lines(output):
    return [line for line in output.splitlines() if line.startswith("+ ") and "eval.py" in line]


class MplcV2EvalScriptOptionTests(unittest.TestCase):
    def setUp(self):
        self.base = REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s94"
        run_dir = self.base / "2024-01-01_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
        (run_dir / "run_status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_datasets_option_runs_one_eval_per_dataset(self):
        result = run_script(EVAL_SCRIPT, args=["--datasets", "sped nordland", "--seeds", "94"])
        self.assertEqual(result.returncode, 0, result.stdout)
        lines = eval_lines(result.stdout)
        self.assertEqual(len(lines), 2, result.stdout)
        self.assertIn("--datasets sped", lines[0])
        self.assertIn("--datasets nordland", lines[1])

    def test_arms_and_no_pretrained_select_models(self):
        result = run_script(
            EVAL_SCRIPT, args=["--datasets", "sped", "--seeds", "94", "--arms", "mplc", "--no-pretrained"]
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        (line,) = eval_lines(result.stdout)
        self.assertIn("--model_tags mplc_s94 ", line)
        self.assertNotIn("pretrained", line.split("--model_tags", 1)[1].split("--", 1)[0])
        self.assertNotIn("WARNING: no finished checkpoint for mplc_v2_supervlad_clean_ft_s94", result.stdout)

    def test_explicit_models_replace_discovery(self):
        result = run_script(
            EVAL_SCRIPT,
            args=["--datasets", "sped", "--model", "a=x/a.pth", "--model", "b=x/b.pth"],
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        (line,) = eval_lines(result.stdout)
        self.assertIn("--model_paths x/a.pth x/b.pth ", line)
        self.assertIn("--model_tags a b ", line)
        self.assertNotIn("WARNING: no finished checkpoint", result.stdout)

    def test_explicit_models_with_discovered(self):
        result = run_script(
            EVAL_SCRIPT,
            args=["--datasets", "sped", "--seeds", "94", "--arms", "mplc", "--model", "a=x/a.pth", "--with-discovered"],
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        (line,) = eval_lines(result.stdout)
        self.assertIn("--model_tags a pretrained mplc_s94 ", line)

    def test_invalid_arm_and_unknown_option_exit_2(self):
        self.assertEqual(run_script(EVAL_SCRIPT, args=["--arms", "pat"]).returncode, 2)
        self.assertEqual(run_script(EVAL_SCRIPT, args=["--bogus"]).returncode, 2)
        self.assertEqual(run_script(EVAL_SCRIPT, args=["--model", "no_equals_sign"]).returncode, 2)

    def test_help_exits_zero(self):
        result = run_script(EVAL_SCRIPT, args=["--help"])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--datasets", result.stdout)

    def test_default_epsilon_grid_is_pixel_referenced(self):
        result = run_script(EVAL_SCRIPT, args=["--datasets", "sped", "--seeds", "94", "--arms", "mplc"])
        (line,) = eval_lines(result.stdout)
        self.assertIn("--epsilons 0.01712 0.03425 0.0685 0.137 ", line)
        self.assertIn("--rank_attack_goal=untargeted", line)
        self.assertIn("output/mplc_v2/supervlad_sped/rank_eval_results.json", line)

    def test_attack_goal_and_shared_options_reach_eval_and_output_dir(self):
        result = run_script(
            EVAL_SCRIPT,
            args=["--datasets", "sped", "--seeds", "94", "--arms", "mplc", "--attack", "rank_apgd_linf",
                  "--steps", "100", "--restarts", "3", "--goal", "targeted", "--shared-attacks"],
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        (line,) = eval_lines(result.stdout)
        for expected in ("--rank_attack=rank_apgd_linf", "--rank_steps=100", "--rank_restarts=3",
                         "--rank_attack_goal=targeted", "--shared_attacks",
                         "supervlad_sped_rank_apgd_linf_targeted_steps100_r3_shared/rank_eval_results.csv"):
            self.assertIn(expected, line)

    def test_checkpoint_option_selects_budget_file_and_tags_it(self):
        result = run_script(
            EVAL_SCRIPT,
            args=["--datasets", "sped", "--seeds", "94", "--arms", "mplc", "--no-pretrained", "--checkpoint", "best_model_budget3.pth"],
        )
        (line,) = eval_lines(result.stdout)
        self.assertIn("2024-01-01_run/best_model_budget3.pth", line)
        self.assertIn("--model_tags mplc_s94_best_model_budget3 ", line)
        self.assertIn("output/mplc_v2/supervlad_sped_best_model_budget3/rank_eval_results.json", line)

    def test_invalid_goal_exits_2(self):
        self.assertEqual(run_script(EVAL_SCRIPT, args=["--goal", "sideways"]).returncode, 2)


if __name__ == "__main__":
    unittest.main()
