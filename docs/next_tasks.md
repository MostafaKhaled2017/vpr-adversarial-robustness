# Next Tasks

## High priority

- Rerun the original perceptual evaluation.
  - Reason: Confirm the NumPy compatibility fix against the real dataset and checkpoint setup.
  - Related files: `perceptual_eval.py`, `third_party/SuperVLAD/datasets_ws.py`

## Medium priority

- Standardize interpreter usage in command examples.
  - Reason: `python` was not available in the active shell, while `python3` was available.
  - Related files: `commands.bash`, `installation.bash`

- Add lightweight smoke tests for dataset filename UTM parsing.
  - Reason: The dataset loader depends on filename format and currently has no root-level test command.
  - Related files: `third_party/SuperVLAD/datasets_ws.py`

## Low priority

- Fill the root README or link it to `docs/`.
  - Reason: The root README is empty.
  - Related files: `README.md`, `docs/index.md`
