import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_rank_pgd_strength_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_rank_pgd_strength_sweep", SCRIPT_PATH)
sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)

import rank_eval


class RankPgdStrengthSweepTests(unittest.TestCase):
    def make_report(self, dataset="msls", epsilon=0.01):
        condition_name = f"rank_pgd_linf_eps_{epsilon:g}"
        model_block = {
            "clean_attacked_subset": {
                "recalls": {"R@1": 80.0, "R@5": 90.0, "R@10": 95.0, "R@100": 99.0}
            },
            condition_name: {
                "recalls": {"R@1": 25.0, "R@5": 50.0, "R@10": 70.0, "R@100": 90.0},
                "attacked_queries": 100,
                "runtime_per_query_seconds": 0.4,
                "attack_success": {
                    "clean_correct": {"rate": 60.0},
                    "all_valid": {"rate": 55.0},
                },
                "rank_displacement": {"mean": 4.0, "p95": 12.0},
                "attack_metadata": {"perturbation_norm": {"mean": 0.049}},
            },
        }
        return {
            "attack": {"mode": "rank_pgd_linf"},
            "results": {
                dataset: {
                    "base": model_block,
                    "checkpoint": model_block,
                }
            },
        }

    def write_report(self, root: Path, dataset="msls", epsilon=0.01) -> Path:
        path = root / "2026-07-06_00-00-00" / "rank_eval_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.make_report(dataset=dataset, epsilon=epsilon)), encoding="utf-8")
        return path

    def test_default_conditions_match_twenty_condition_budget(self):
        config = sweep.SweepConfig()

        conditions = sweep.selected_conditions(config)

        self.assertEqual(len(conditions), 20)
        self.assertEqual(conditions[0].group, "epsilon_baseline")
        self.assertEqual(conditions[0].epsilon, 0.01)
        self.assertEqual(conditions[-1].group, "strong_candidate_sweep")
        self.assertEqual(conditions[-1].epsilon, 0.20)

    def test_default_config_uses_sped_in_process_and_full_dataset(self):
        config = sweep.SweepConfig()

        self.assertEqual(config.datasets, ("sped",))
        self.assertEqual(config.parallel_runs, 1)
        self.assertEqual(config.execution_mode, "in_process")
        self.assertIsNone(config.max_dataset_samples)
        self.assertFalse(config.compute_diagnostics)

    def test_default_jobs_expand_to_twenty_sped_conditions(self):
        config = sweep.SweepConfig()

        jobs = sweep.expand_jobs(config, sweep.selected_conditions(config))

        self.assertEqual(len(jobs), 20)
        self.assertEqual(jobs[0].dataset, "sped")
        self.assertEqual(jobs[0].condition_id, "condition_01")

    def test_max_experiments_per_dataset_truncates_conditions(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=7)

        conditions = sweep.selected_conditions(config)

        self.assertEqual(len(conditions), 7)
        self.assertEqual(conditions[-1].condition_id, "condition_07")

    def test_invalid_parallel_runs_fails_validation(self):
        with self.assertRaises(ValueError):
            sweep.validate_config(sweep.SweepConfig(parallel_runs=0))

    def test_in_process_parallel_runs_cannot_exceed_dataset_count(self):
        with self.assertRaises(ValueError):
            sweep.validate_config(sweep.SweepConfig(datasets=("sped",), parallel_runs=2))

    def test_subprocess_parallel_runs_can_exceed_dataset_count(self):
        sweep.validate_config(sweep.SweepConfig(datasets=("sped",), parallel_runs=2, execution_mode="subprocess"))

    def test_invalid_max_dataset_samples_fails_validation(self):
        with self.assertRaises(ValueError):
            sweep.validate_config(sweep.SweepConfig(max_dataset_samples=0))

    def test_max_dataset_samples_must_cover_largest_negative_count(self):
        config = sweep.SweepConfig(max_dataset_samples=10)
        with self.assertRaises(ValueError):
            sweep.validate_condition_dependent_config(config, sweep.selected_conditions(config))

    def test_more_than_twenty_experiments_fails_validation(self):
        with self.assertRaises(ValueError):
            sweep.validate_config(sweep.SweepConfig(max_experiments_per_dataset=21))

    def test_datasets_are_limited_to_msls_and_sped(self):
        with self.assertRaises(ValueError):
            sweep.validate_config(sweep.SweepConfig(datasets=("msls", "nordland")))

    def test_config_file_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "sweep_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": ["sped"],
                        "parallel_runs": 3,
                        "max_experiments_per_dataset": 6,
                        "epsilons": [0.01, 0.02],
                    }
                ),
                encoding="utf-8",
            )

            args = sweep.parse_arguments(
                [
                    "--config",
                    str(config_path),
                    "--parallel_runs",
                    "2",
                    "--datasets",
                    "msls",
                    "sped",
                ]
            )
            config = sweep.build_config(args)

        self.assertEqual(config.datasets, ("msls", "sped"))
        self.assertEqual(config.parallel_runs, 2)
        self.assertEqual(config.max_experiments_per_dataset, 6)
        self.assertEqual(config.epsilons, (0.01, 0.02))

    def test_smoke_mode_defaults_to_five_queries(self):
        config = sweep.build_config(sweep.parse_arguments(["--smoke"]))

        self.assertTrue(config.smoke)
        self.assertEqual(config.max_queries, 5)

    def test_final_runs_reject_query_cap(self):
        with self.assertRaises(ValueError):
            sweep.build_config(sweep.parse_arguments(["--max_queries", "5"]))

    def test_step_size_probe_uses_absolute_multiplier_value(self):
        conditions = sweep.selected_conditions(sweep.SweepConfig())
        one_x = next(
            condition
            for condition in conditions
            if condition.group == "step_size_probe" and condition.step_size_multiplier == 1.0
        )
        four_x = next(
            condition
            for condition in conditions
            if condition.group == "step_size_probe" and condition.step_size_multiplier == 4.0
        )

        self.assertAlmostEqual(one_x.rank_step_size, 0.0025)
        self.assertAlmostEqual(one_x.resolved_step_size, 0.0025)
        self.assertAlmostEqual(four_x.rank_step_size, 0.01)

    def test_command_for_job_contains_single_dataset_and_parallel_safe_output_paths(self):
        config = sweep.SweepConfig(smoke=True, max_queries=5)
        condition = sweep.selected_conditions(config)[0]
        job = sweep.SweepJob(dataset="sped", condition=condition)
        command = sweep.command_for_job(config, job, Path("test/rank_eval/sweeps/run/runs/sped/condition_01"))

        self.assertIn("--datasets", command)
        self.assertIn("sped", command)
        self.assertNotIn("msls", command)
        self.assertIn("--model_type=supervlad", command)
        self.assertIn("--model_paths", command)
        self.assertNotIn("--models", command)
        self.assertIn("--max_queries=5", command)
        self.assertIn("--crossimage_encoder", command)
        self.assertNotIn("--compute_diagnostics", command)
        self.assertTrue(any(part.endswith("runs/sped/condition_01/rank_eval_results.json") for part in command))

    def test_compute_diagnostics_is_propagated_to_job_command(self):
        config = sweep.SweepConfig(compute_diagnostics=True, max_experiments_per_dataset=1)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]

        command = sweep.command_for_job(config, job, Path("test/sweep/job"))

        self.assertIn("--compute_diagnostics", command)

    def test_compute_diagnostics_cli_is_opt_in(self):
        config = sweep.build_config(sweep.parse_arguments(["--compute_diagnostics"]))

        self.assertTrue(config.compute_diagnostics)

    def test_in_process_diagnostics_path_is_none_when_disabled(self):
        disabled_args = SimpleNamespace(compute_diagnostics=False)
        enabled_args = SimpleNamespace(compute_diagnostics=True)

        sweep.configure_rank_eval_output_dirs(disabled_args, Path("test/job"))
        sweep.configure_rank_eval_output_dirs(enabled_args, Path("test/job"))

        self.assertIsNone(disabled_args.diagnostics_output_dir_path)
        self.assertEqual(enabled_args.diagnostics_output_dir_path, Path("test/job/diagnostics"))

    def test_dry_run_output_reports_count_and_parallelism(self):
        config = sweep.SweepConfig(datasets=("msls", "sped"), max_experiments_per_dataset=2, parallel_runs=2)
        conditions = sweep.selected_conditions(config)
        jobs = sweep.expand_jobs(config, conditions)
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            sweep.print_dry_run(config, Path("test/sweep"), conditions, jobs)

        output = buffer.getvalue()
        self.assertIn("Execution mode: in_process", output)
        self.assertIn("Conditions per dataset: 2", output)
        self.assertIn("Expected condition results: 4", output)
        self.assertIn("Dataset passes: 2", output)
        self.assertIn("Dataset mode: full_dataset", output)
        self.assertIn("Parallel runs: 2", output)

    def test_rows_from_report_collates_clean_attacked_and_metadata(self):
        condition = sweep.SweepCondition(
            condition_id="condition_01",
            group="epsilon_baseline",
            epsilon=0.05,
            rank_steps=20,
            rank_step_size=None,
            step_size_mode="default_2eps_over_steps",
            step_size_multiplier=None,
            rank_restarts=1,
            adv_negatives=5,
        )
        report = self.make_report(epsilon=0.05)

        rows = sweep.rows_from_report(report, condition)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dataset"], "msls")
        self.assertEqual(rows[0]["clean_R@1"], 80.0)
        self.assertEqual(rows[0]["attacked_R@1"], 25.0)
        self.assertEqual(rows[0]["p95_rank_displacement"], 12.0)
        self.assertEqual(rows[0]["mean_perturbation_norm"], 0.049)

    def test_valid_result_detection_skips_completed_job(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=1)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]
        with tempfile.TemporaryDirectory() as directory:
            sweep_dir = Path(directory)
            self.write_report(sweep.job_run_dir(sweep_dir, job), dataset=job.dataset, epsilon=job.condition.epsilon)

            states = sweep.inspect_jobs(sweep_dir, [job], config)

        self.assertEqual(states[0].status, "completed")
        self.assertEqual(sweep.pending_jobs_from_states(states), [])

    def test_interrupted_output_without_json_is_pending(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=1)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]
        with tempfile.TemporaryDirectory() as directory:
            sweep_dir = Path(directory)
            run_dir = sweep.job_run_dir(sweep_dir, job)
            run_dir.mkdir(parents=True)
            (run_dir / "stderr.log").write_text("KeyboardInterrupt", encoding="utf-8")

            states = sweep.inspect_jobs(sweep_dir, [job], config)

        self.assertEqual(states[0].status, "invalid")
        self.assertEqual(sweep.pending_jobs_from_states(states), [job])

    def test_completed_job_requires_diagnostic_files_when_enabled(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=1, compute_diagnostics=True)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]
        with tempfile.TemporaryDirectory() as directory:
            sweep_dir = Path(directory)
            run_dir = sweep.job_run_dir(sweep_dir, job)
            report_path = self.write_report(run_dir, dataset=job.dataset, epsilon=job.condition.epsilon)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            diagnostics_dir = run_dir / "diagnostics"
            report["diagnostics_output_dir"] = str(diagnostics_dir)
            report_path.write_text(json.dumps(report), encoding="utf-8")

            missing_state = sweep.inspect_job_state(sweep_dir, job, config)
            diagnostics_dir.mkdir(parents=True)
            epsilon_label = f"{job.condition.epsilon:g}"
            for model_tag in config.model_tags:
                path = diagnostics_dir / f"{job.dataset}_{model_tag}_{config.rank_attack}_eps_{epsilon_label}.csv"
                path.write_text("query_index\n0\n", encoding="utf-8")
            completed_state = sweep.inspect_job_state(sweep_dir, job, config)

        self.assertEqual(missing_state.status, "invalid")
        self.assertEqual(completed_state.status, "completed")

    def test_legacy_combined_layout_can_satisfy_dataset_job(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=1)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]
        with tempfile.TemporaryDirectory() as directory:
            sweep_dir = Path(directory)
            self.write_report(
                sweep.legacy_condition_run_dir(sweep_dir, job),
                dataset=job.dataset,
                epsilon=job.condition.epsilon,
            )

            state = sweep.inspect_job_state(sweep_dir, job, config)

        self.assertEqual(state.status, "completed_legacy")

    def test_resume_validation_allows_parallelism_and_batch_size_changes(self):
        old_config = sweep.SweepConfig(
            datasets=("msls", "sped"),
            max_experiments_per_dataset=1,
            parallel_runs=1,
            infer_batch_size=8,
        )
        new_config = sweep.SweepConfig(
            datasets=("msls", "sped"),
            max_experiments_per_dataset=1,
            parallel_runs=2,
            infer_batch_size=16,
        )
        conditions = sweep.selected_conditions(old_config)
        manifest = {
            "portable_config": sweep.portable_config(old_config, conditions),
        }

        sweep.validate_resume_manifest(manifest, new_config, sweep.selected_conditions(new_config))

    def test_resume_without_manifest_scans_existing_results(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=1)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]
        with tempfile.TemporaryDirectory() as directory:
            sweep_dir = Path(directory)
            self.write_report(sweep.job_run_dir(sweep_dir, job), dataset=job.dataset, epsilon=job.condition.epsilon)

            self.assertIsNone(sweep.load_existing_manifest(sweep_dir))
            states = sweep.inspect_jobs(sweep_dir, [job], config)

        self.assertEqual(states[0].status, "completed")

    def test_resume_missing_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                sweep.main(["--resume_sweep_dir", str(missing), "--dry_run"])

    def test_dry_run_resume_reports_completed_and_pending_counts(self):
        config = sweep.SweepConfig(datasets=("msls", "sped"), max_experiments_per_dataset=1)
        conditions = sweep.selected_conditions(config)
        jobs = sweep.expand_jobs(config, conditions)
        states = [
            sweep.JobState(job=jobs[0], status="completed", run_dir="runs/msls/condition_01", output_json="result.json"),
            sweep.JobState(job=jobs[1], status="pending", run_dir="runs/sped/condition_01"),
        ]
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            sweep.print_dry_run(config, Path("test/sweep"), conditions, jobs, states)

        output = buffer.getvalue()
        self.assertIn("Completed jobs: 1", output)
        self.assertIn("Pending jobs: 1", output)
        self.assertIn("Invalid/incomplete jobs: 0", output)

    def test_in_process_runner_groups_conditions_by_dataset_pass(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=2)
        jobs = sweep.expand_jobs(config, sweep.selected_conditions(config))
        calls = []
        original_run_dataset_pass = sweep.run_dataset_pass

        def fake_run_dataset_pass(config_arg, sweep_dir_arg, dataset, dataset_jobs):
            calls.append((dataset, len(dataset_jobs)))
            return [
                sweep.JobResult(
                    dataset=job.dataset,
                    condition_id=job.condition_id,
                    command=[],
                    returncode=0,
                    stdout_path="",
                    stderr_path="",
                    run_dir="",
                    output_json=None,
                    output_csv=None,
                    started_at="",
                    finished_at="",
                )
                for job in dataset_jobs
            ]

        try:
            sweep.run_dataset_pass = fake_run_dataset_pass
            results = sweep.run_in_process_jobs(config, Path("test/sweep"), jobs)
        finally:
            sweep.run_dataset_pass = original_run_dataset_pass

        self.assertEqual(calls, [("sped", 2)])
        self.assertEqual(len(results), 2)

    def test_deterministic_dataset_sampling_is_seed_stable(self):
        positives = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ]

        first = rank_eval.select_sampled_query_indices(positives, "sped", 7, 3, None)
        second = rank_eval.select_sampled_query_indices(positives, "sped", 7, 3, None)
        different = rank_eval.select_sampled_query_indices(positives, "sped", 8, 3, None)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))

    def test_sampled_database_selection_keeps_query_positives(self):
        positives = [
            np.array([4, 5], dtype=np.int64),
            np.array([6], dtype=np.int64),
            np.array([7], dtype=np.int64),
        ]

        selected = rank_eval.select_sampled_database_indices(
            positives,
            np.array([0, 1], dtype=np.int64),
            database_size=10,
            dataset_name="sped",
            seed=3,
            requested_size=5,
        )

        self.assertIn(4, selected)
        self.assertIn(6, selected)
        self.assertLessEqual(len(selected), 5)

    def test_attempt_logs_do_not_overwrite_previous_attempts(self):
        config = sweep.SweepConfig(max_experiments_per_dataset=1)
        job = sweep.expand_jobs(config, sweep.selected_conditions(config))[0]
        original_run = sweep.subprocess.run

        def fake_run(command, cwd, env, stdout, stderr, check):
            output_json_arg = next(part for part in command if part.startswith("--output_json="))
            output_parent = Path(output_json_arg.split("=", 1)[1]).parent
            report_path = output_parent / datetime_token() / "rank_eval_results.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(self.make_report(dataset=job.dataset, epsilon=job.condition.epsilon)),
                encoding="utf-8",
            )
            stdout.write("ok\n")
            stderr.write("ok\n")
            return SimpleNamespace(returncode=0)

        def datetime_token():
            return f"2026-07-06_00-00-{len(list(run_dir.glob('attempts/*'))) + 1:02d}"

        with tempfile.TemporaryDirectory() as directory:
            sweep_dir = Path(directory)
            run_dir = sweep.job_run_dir(sweep_dir, job)
            try:
                sweep.subprocess.run = fake_run
                sweep.run_job(config, sweep_dir, job)
                sweep.run_job(config, sweep_dir, job)
            finally:
                sweep.subprocess.run = original_run

            stdout_logs = sorted((run_dir / "attempts").glob("*/stdout.log"))

        self.assertEqual(len(stdout_logs), 2)

    def test_select_strongest_setting_uses_tie_breaks(self):
        rows = [
            {
                "condition_id": "condition_01",
                "dataset": "msls",
                "model": "base",
                "attacked_R@1": 10.0,
                "clean_correct_attack_success_rate": 70.0,
                "runtime_per_query_seconds": 0.5,
                "compute_budget": 100,
            },
            {
                "condition_id": "condition_02",
                "dataset": "msls",
                "model": "base",
                "attacked_R@1": 10.0,
                "clean_correct_attack_success_rate": 75.0,
                "runtime_per_query_seconds": 0.6,
                "compute_budget": 100,
            },
            {
                "condition_id": "condition_03",
                "dataset": "msls",
                "model": "base",
                "attacked_R@1": 12.0,
                "clean_correct_attack_success_rate": 90.0,
                "runtime_per_query_seconds": 0.1,
                "compute_budget": 10,
            },
        ]

        selected = sweep.select_strongest_setting(rows)

        self.assertEqual(selected["condition_id"], "condition_02")


if __name__ == "__main__":
    unittest.main()
