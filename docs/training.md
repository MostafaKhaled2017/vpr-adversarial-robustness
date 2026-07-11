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

## Candidate model transfer analysis

`docs/vpr_model_adversarial_training_integration_analysis.md` compares MegaLoc, Bag-of-Queries, AnyLoc, FoL, CricaVPR, SelaVPR, EigenPlaces, and MixVPR against the current trainer's model, data, loss, attack-gradient, checkpoint, and retrieval-evaluation interfaces. It recommends Bag-of-Queries for the main cross-model generalization experiment, with CricaVPR as the lowest-effort integration pilot.

## Outputs

Training writes logs, checkpoints, and TensorBoard data under the configured log/save directories.

## Notes

Full training was not run during initial documentation. Training requires datasets, checkpoints, dependencies, and likely GPU resources.
