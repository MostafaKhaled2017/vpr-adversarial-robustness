# Build and Run

## Build commands

No build system command is documented for the root repository. This appears to be a Python script/package workflow.

## Run commands

Repository-documented examples from `commands.bash`:

- `python adv_train.py ...`
- `python3 fgsm_eval.py ...`
- `python3 perceptual_eval.py ...`

Repository-documented shell wrappers:

- `scripts/run_perceptual_adv_training.sh`
- `scripts/run_rank_eval.sh`

## Test commands

Lightweight tests added for native rank attacks and retrieval metrics:

- `python3 -m unittest discover tests`

Syntax validation for the native rank evaluator:

- `python3 -m py_compile rank_eval.py perceptual_adv_training/rank_attacks.py perceptual_adv_training/retrieval_metrics.py perceptual_adv_training/targets.py`

Validation commands run during the 2026-06-04 compatibility fix:

- `python3 -m py_compile third_party/SuperVLAD/datasets_ws.py`
- `rg -n "np\\.float\\b" third_party/SuperVLAD perceptual_eval.py fgsm_eval.py adv_train.py perceptual_adv_training.py perceptual_adv_training`

## Validation notes

The native rank attack unit tests and syntax checks pass with `python3`. Full training and full dataset evaluation were not run because they require longer dataset, checkpoint, and likely GPU resources.
