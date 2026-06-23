# Important Files

## Source files

- `adv_train.py`: rank-aware adversarial training script.
- `perceptual_adv_training.py`: perceptual adversarial training script.
- `fgsm_eval.py`: FGSM robustness evaluation script.
- `rank_eval.py`: native retrieval rank attack evaluation script.
- `perceptual_eval.py`: clean and perceptual attack comparison script.
- `perceptual_adv_training/`: local package for training, attacks, losses, targets, checkpoints, and evaluation support.
- `perceptual_adv_training/rank_attacks.py`: Rank-PGD and APGD-style rank attack implementations.
- `perceptual_adv_training/retrieval_metrics.py`: recall, nearest-positive rank, rank displacement, and attack success metric helpers.
- `third_party/SuperVLAD/datasets_ws.py`: SuperVLAD dataset loader for database/query image splits and UTM coordinates.
- `third_party/SuperVLAD/model/`: SuperVLAD model implementation.

## Configuration files

- `requirements.txt`: pinned Python dependencies.
- `installation.bash`: additional pip installation commands.
- `.gitmodules`: submodule metadata.

## Launch or entry-point files

- `commands.bash`: example commands for training and evaluation.
- `scripts/run_perceptual_adv_training.sh`: perceptual adversarial training wrapper.
- `scripts/run_rank_eval.sh`: native rank attack evaluation wrapper.
- `scripts/run_fgsm_dual_eval.sh`: FGSM evaluation wrapper.
- `scripts/run_fgsm_checkpoint_list_eval.sh`: FGSM checkpoint-list evaluation wrapper.
- `scripts/adv_training_different_epsilons.sh`: adversarial training sweep wrapper.
- `tests/`: lightweight unit tests for rank attack mechanics and retrieval metrics.

## Documentation files

- `README.md`: empty root README.
- `third_party/SuperVLAD/README.md`: upstream SuperVLAD usage and dataset layout notes.
- `submodules/perceptual-advex/README.md`: perceptual-advex documentation.
- `docs/`: repository project documentation.
- `reports/vpr_next_steps_implementation_plan.md`: roadmap for the next VPR robustness implementation phases.
- `reports/phase1_native_rank_attack_implementation.md`: detailed explanation of the implemented Phase 1 native rank attack evaluator.
- `reports/native-rank-attacks-evaluation.tex`: LaTeX report summarizing the final native rank attack evaluation.
- `reports/native-rank-attacks-evaluation.pdf`: compiled PDF report for the final native rank attack evaluation.

## Generated or output files

- `logs/`: training logs and checkpoints, referenced by command examples.
- `test/rank_eval/`: default native rank evaluation output location.
- `test/rank_eval/2026-06-08_18-46-59/msls_sped_nordland_rank_comparison.csv`: completed native `rank_pgd_linf` evaluation summary used by the report.
- `test/perceptual_eval/`: perceptual evaluation logs and result outputs.
- `checkpoints/`: expected checkpoint location in command examples.
- `datasets/`: expected dataset location in command examples.
