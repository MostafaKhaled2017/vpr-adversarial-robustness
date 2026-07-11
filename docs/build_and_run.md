# Build and Run

## Build commands

No build system command is documented for the root repository. This appears to be a Python script/package workflow.

## Run commands

Repository-documented examples from `commands.bash`:

- `python adv_train.py ...`
- `python3 fgsm_eval.py ...`
- `python3 perceptual_eval.py ...`
- `python3 rank_eval.py ...`

Repository-documented shell wrappers:

- `scripts/run_perceptual_adv_training.sh`
- `scripts/run_rank_eval.sh`

Phase 5 Rank-PGD strength sweep dry run:

- `python3 scripts/run_rank_pgd_strength_sweep.py --dry_run`

Delayed MSLS sampled-gallery Phase 5 sweep:

- `scripts/run_delayed_msls_rank_pgd_strength_sweep.sh`

Phase 5 smoke sweep command shape:

- `python3 scripts/run_rank_pgd_strength_sweep.py --smoke --max_dataset_samples 21`

The default full Phase 5 sweep uses SPED only, `--execution_mode in_process`, and `--parallel_runs 1`. When `--max_dataset_samples` is omitted, it uses the full selected dataset. To run MSLS and SPED concurrently in in-process mode, pass both datasets and set `--parallel_runs 2`:

- `python3 scripts/run_rank_pgd_strength_sweep.py --datasets msls sped --parallel_runs 2`

Use `--max_dataset_samples N` only for deterministic sampled-gallery attack-setting comparisons; sampled-gallery metrics are not full-benchmark comparable.

Phase 5 resume command shape:

- `python3 scripts/run_rank_pgd_strength_sweep.py --resume_sweep_dir test/rank_eval/sweeps/<sweep_id> --parallel_runs 1 --infer_batch_size 8`

Phase 5 resume status preview:

- `python3 scripts/run_rank_pgd_strength_sweep.py --resume_sweep_dir test/rank_eval/sweeps/<sweep_id> --dry_run`

## Test commands

Lightweight tests added for native rank attacks and retrieval metrics:

- `python3 -m unittest discover tests`
- `python3 -m unittest tests/test_rank_eval_interface.py`
- `python3 -m unittest tests/test_rank_pgd_strength_sweep.py`

Syntax validation for the native rank evaluator:

- `python3 -m py_compile rank_eval.py perceptual_adv_training/rank_attacks.py perceptual_adv_training/retrieval_metrics.py perceptual_adv_training/targets.py`
- `python3 -m py_compile rank_eval.py tests/test_rank_eval_interface.py`
- `python3 -m py_compile scripts/run_rank_pgd_strength_sweep.py tests/test_rank_pgd_strength_sweep.py`
- `python3 rank_eval.py --help`
- `bash -n commands.bash`
- `bash -n scripts/run_rank_eval.sh`

Validation commands run during the 2026-06-04 compatibility fix:

- `python3 -m py_compile third_party/SuperVLAD/datasets_ws.py`
- `rg -n "np\\.float\\b" third_party/SuperVLAD perceptual_eval.py fgsm_eval.py adv_train.py perceptual_adv_training.py perceptual_adv_training`

## Validation notes

The native rank attack unit tests, syntax checks, CLI help checks, and Phase 5 sweep-runner dry runs pass with `python3`. Full training and full dataset evaluation were not run because they require longer dataset, checkpoint, and likely GPU resources. In in-process sweep mode, `--parallel_runs` cannot exceed the number of selected datasets.
