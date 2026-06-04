# Training

## Rank-aware adversarial training

Entry point:

- `adv_train.py`

Command example:

- Found in `commands.bash`.

Key options include `--adv_epsilon`, `--adv_steps`, `--adv_loss_weight`, `--adv_align_weight`, `--adv_negatives`, and `--adv_warmup_epochs`.

## Perceptual adversarial training

Entry points:

- `perceptual_adv_training.py`
- `scripts/run_perceptual_adv_training.sh`

The training package wraps attacks from `submodules/perceptual-advex` with retrieval-specific targets and losses.

## Outputs

Training writes logs, checkpoints, and TensorBoard data under the configured log/save directories.

## Notes

Full training was not run during initial documentation. Training requires datasets, checkpoints, dependencies, and likely GPU resources.
