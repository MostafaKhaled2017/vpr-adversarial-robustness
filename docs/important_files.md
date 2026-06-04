# Important Files

## Source files

- `adv_train.py`: rank-aware adversarial training script.
- `perceptual_adv_training.py`: perceptual adversarial training script.
- `fgsm_eval.py`: FGSM robustness evaluation script.
- `perceptual_eval.py`: clean and perceptual attack comparison script.
- `perceptual_adv_training/`: local package for training, attacks, losses, targets, checkpoints, and evaluation support.
- `third_party/SuperVLAD/datasets_ws.py`: SuperVLAD dataset loader for database/query image splits and UTM coordinates.
- `third_party/SuperVLAD/model/`: SuperVLAD model implementation.

## Configuration files

- `requirements.txt`: pinned Python dependencies.
- `installation.bash`: additional pip installation commands.
- `.gitmodules`: submodule metadata.

## Launch or entry-point files

- `commands.bash`: example commands for training and evaluation.
- `scripts/run_perceptual_adv_training.sh`: perceptual adversarial training wrapper.
- `scripts/run_fgsm_dual_eval.sh`: FGSM evaluation wrapper.
- `scripts/run_fgsm_checkpoint_list_eval.sh`: FGSM checkpoint-list evaluation wrapper.
- `scripts/adv_training_different_epsilons.sh`: adversarial training sweep wrapper.

## Documentation files

- `README.md`: empty root README.
- `third_party/SuperVLAD/README.md`: upstream SuperVLAD usage and dataset layout notes.
- `submodules/perceptual-advex/README.md`: perceptual-advex documentation.
- `docs/`: repository project documentation.

## Generated or output files

- `logs/`: training logs and checkpoints, referenced by command examples.
- `test/perceptual_eval/`: perceptual evaluation logs and result outputs.
- `checkpoints/`: expected checkpoint location in command examples.
- `datasets/`: expected dataset location in command examples.
