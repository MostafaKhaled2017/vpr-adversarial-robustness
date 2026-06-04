# Current State

## Working status

The repository appears to be a Python ML/CV experiment workspace. The touched dataset module compiles with `python3`. Full build, training, and evaluation status are unknown because datasets, checkpoints, and GPU runtime validation were not run.

## Active task

Fix NumPy 2.x compatibility in the SuperVLAD dataset loader after `perceptual_eval.py` failed on the removed `np.float` alias.

## Recent changes

- Replaced `np.float` with `float` for database and query UTM coordinate arrays in `third_party/SuperVLAD/datasets_ws.py`.
- Created the initial `docs/` project documentation baseline.

## Known blockers

- Root `README.md` is empty.
- Full evaluation requires datasets under `--eval_datasets_folder`.
- Checkpoints such as `checkpoints/SuperVLAD.pth`, `checkpoints/perceptual_adv_checkpoint.pth`, and the DINOv2 foundation checkpoint are expected by example commands.
- Python alias `python` was not found in the active shell; repository examples use both `python` and `python3`.

## Next recommended steps

- Rerun the original `perceptual_eval.py` command with the intended datasets and checkpoints.
- Standardize command examples on the interpreter name available in the target environment.
- Add a small automated smoke test for UTM filename parsing if lightweight test infrastructure is introduced.
