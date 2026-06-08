# Current State

## Working status

The repository appears to be a Python ML/CV experiment workspace. Lightweight unit tests and syntax checks pass for the native rank attack evaluator added on 2026-06-06. Full training and full MSLS evaluation status are unknown because they require longer dataset/checkpoint/GPU runs.

## Active task

Document the Phase 1 native rank attack implementation.

## Recent changes

- Added `rank_eval.py` for standalone query-only Rank-PGD-Linf, Rank-PGD-L2, and APGD-style Linf evaluation.
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
- Python alias `python` was not found in the active shell; repository examples use both `python` and `python3`.

## Next recommended steps

- Run `scripts/run_rank_eval.sh` on the intended MSLS setup and compare Rank-PGD-Linf against FGSM `training_style` at comparable epsilon.
- Implement rank-margin diagnostics to compare clean margins and attack success.
- Rerun the original `perceptual_eval.py` command with the intended datasets and checkpoints when dataset/checkpoint access is available.
