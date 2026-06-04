# Build and Run

## Build commands

No build system command is documented for the root repository. This appears to be a Python script/package workflow.

## Run commands

Repository-documented examples from `commands.bash`:

- `python adv_train.py ...`
- `python3 fgsm_eval.py ...`
- `python3 perceptual_eval.py ...`

Repository-documented shell wrapper:

- `scripts/run_perceptual_adv_training.sh`

## Test commands

No dedicated automated test command was found in the root repository.

Validation commands run during the 2026-06-04 compatibility fix:

- `python3 -m py_compile third_party/SuperVLAD/datasets_ws.py`
- `rg -n "np\\.float\\b" third_party/SuperVLAD perceptual_eval.py fgsm_eval.py adv_train.py perceptual_adv_training.py perceptual_adv_training`

## Validation notes

The touched dataset module compiles with `python3`. Full training, dataset loading, and evaluation were not run because they require datasets, checkpoints, and likely GPU resources.
