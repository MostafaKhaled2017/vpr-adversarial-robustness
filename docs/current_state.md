# Current State

## Working status

The repository is a Python ML/CV experiment workspace. Lightweight unit tests and syntax checks pass for the native rank attack evaluator and the Phase 5 Rank-PGD strength sweep runner. A full multi-dataset `rank_pgd_linf` evaluation for MSLS, SPED, and Nordland is available under `test/rank_eval/2026-06-08_18-46-59/`. Full training status is unknown because training requires longer dataset/checkpoint/GPU runs.

## Active task

Improve the Phase 5 Rank-PGD strength sweep runner so the default SPED sweep reuses dataset/model/clean descriptor setup across conditions and supports deterministic sampled-gallery attack comparisons.

## Recent changes

- Added `rank_eval.py` for standalone query-only Rank-PGD-Linf, Rank-PGD-L2, and APGD-style Linf evaluation.
- Updated `rank_eval.py` to use unified `--datasets`, `--models`, and `--model_tags` arguments and to support shared-base attacks across one or more models.
- Updated `rank_eval.py` to add the vendored `third_party/SuperVLAD` path at runtime, so `python3 rank_eval.py ...` works from the repository root without a manual `PYTHONPATH` export.
- Updated `rank_eval.py` to save each run under a timestamped subdirectory such as `test/rank_eval/YYYY-MM-DD_HH-MM-SS/`, including when explicit JSON/CSV filenames are provided.
- Added a multi-dataset base/checkpoint `rank_eval.py` example to `commands.bash`.
- Added `reports/native-rank-attacks-evaluation.tex` and compiled `reports/native-rank-attacks-evaluation.pdf` summarizing the final native `rank_pgd_linf` evaluation.
- Updated `reports/native-rank-attacks-evaluation.tex` so the recall table is pinned at its source location in the compiled PDF.
- Added `scripts/run_rank_pgd_strength_sweep.py` to orchestrate the Phase 5 Rank-PGD strength sweep with JSON config support, `--parallel_runs`, dry-run output, smoke mode, resume support, manifest writing, summary collation, and strongest-setting selection.
- Updated the Phase 5 sweep runner default to SPED-only in-process execution, so each dataset pass reuses loaded models, dataset state, clean descriptors, and cached attack targets across pending conditions.
- Added deterministic sampled-gallery support through `--max_dataset_samples`, which caps query and database samples from `--seed` and marks outputs as not full-benchmark comparable.
- Updated the in-process sweep runner to add the repository root and vendored SuperVLAD paths at startup, so it can import `rank_eval.py` and SuperVLAD helpers from the repository root.
- Added a CUDA cache flush before each in-process rank-attack condition to reduce failures from PyTorch reserved-but-unallocated memory during resumed sweeps.
- Added unit tests for the Phase 5 sweep runner condition budget, config precedence, smoke/query-cap validation, resume behavior, in-process grouping, deterministic sampling, summary collation, and selection tie-breaks.
- Added reusable rank attack and retrieval metric modules under `perceptual_adv_training/`.
- Added `scripts/run_rank_eval.sh` and lightweight `unittest` coverage for rank attacks and metrics.
- Added `reports/phase1_native_rank_attack_implementation.md` to explain the Phase 1 implementation, outputs, validation, and limitations.
- Added an implementation plan for the next VPR robustness stage under `reports/`.
- Replaced `np.float` with `float` for database and query UTM coordinate arrays in `third_party/SuperVLAD/datasets_ws.py`.
- Created the initial `docs/` project documentation baseline.

## Known blockers

- Root `README.md` is empty.
- Full evaluation requires datasets under `--eval_datasets_folder`.
- Checkpoints such as `checkpoints/SuperVLAD.pth`, `checkpoints/perceptual_adv_checkpoint.pth`, and the DINOv2 foundation checkpoint are expected by example commands.
- Full SPED in-process sweeps can still exceed a 10 GB GPU at larger batch sizes; reduce `--infer_batch_size` or use `--max_dataset_samples` for attack-setting smoke comparisons if CUDA OOM occurs.
- Python alias `python` was not found in the active shell; repository examples use both `python` and `python3`.
- Other root scripts that import SuperVLAD modules may still require the `PYTHONPATH` setup shown in `commands.bash`; `rank_eval.py` no longer requires it.
- If two `rank_eval.py` runs start within the same second, the timestamped run directory name could still collide because SuperVLAD logging refuses existing output directories.
- Full Phase 5 sweep execution has not been run; only sweep runner dry runs, syntax checks, and unit tests have been validated.

## Next recommended steps

- Run a sampled-gallery Phase 5 smoke sweep with `--smoke --max_dataset_samples 21` before launching a full SPED or MSLS/SPED sweep.
- Run the full Phase 5 Rank-PGD strength sweep when GPU resources are available. Use `--datasets msls sped --parallel_runs 2` only when enough memory is available for two concurrent dataset passes.
- Compare the native `rank_pgd_linf` report against the proxy/perceptual attack report and decide which robustness findings belong in the final write-up.
