#!/usr/bin/env bash
# Shared SuperVLAD configuration for Phases 1 and 2 of the full-conference plan.
#
# Phase 1 requires the clean-only control and the adversarial (PAT) arm to be *matched*:
# identical data, epochs, optimizer, LR schedule, frozen layers, and checkpoint-selection
# rule, differing only in the attack configuration (plan, Global Constraints). Defining the
# flag list once and appending SUPERVLAD_ATTACK_FLAGS only for the adversarial arm makes that
# matching structural rather than something a reader has to diff by eye.
#
# PROVENANCE: these flags are recovered verbatim from the run metadata of
#   logs/default/2026-07-31_22-57-12/training_config.yaml
# which is the SuperVLAD PAT run behind the checkpoint currently under evaluation. They are
# NOT the flags in scripts/super_vlad_train.sh — that launcher has drifted from the run
# (different lr, epochs, patience, and mixed precision). The plan (Task 1.2) requires every
# non-attack flag to be byte-identical to the run behind the paper, so the metadata wins.
#
# Values that happen to equal a current src/cli.py default are still written out explicitly:
# a matched-control experiment must not silently change when a default changes.

# Every flag except the attack configuration and --seed/--save_dir. Shared by BOTH arms.
SUPERVLAD_BASE_FLAGS=(
  --model=supervlad
  --eval_datasets_folder=datasets
  --gsv_cities_base_path=datasets/gsv_cities
  --eval_dataset_name=msls
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
  --backbone=dino
  --supervlad_clusters=4
  --crossimage_encoder
  --freeze_te=8
  --optim=adam
  --lr=0.00005
  --lr_schedule=120
  --lr_plateau_patience=3
  --lr_plateau_factor=0.5
  --num_epochs=50
  --patience=6
  --batch_size=16
  --infer_batch_size=16
  --train_resize 322 322
  --resize 322 322
  --test_method=hard_resize
  --shuffle
  --randomize_attack
  --adv_negatives=5
  --adv_warmup_epochs=2
  --adv_margin=0.1
  --adv_loss_weight=1.0
  --adv_align_weight=0.05
  --selection_robust_weight=0.75
  --early_stop_min_delta=0.0
  --keep_every=6
  --val_batches=250
)

# The adversarial arm's only addition. Frozen at the paper's configuration.
SUPERVLAD_ATTACK_FLAGS=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)"
)

# Model flags eval.py needs to rebuild a SuperVLAD checkpoint.
SUPERVLAD_EVAL_MODEL_FLAGS=(
  --model_type=supervlad
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
  --backbone=dino
  --supervlad_clusters=4
  --crossimage_encoder
  --freeze_te=8
  # SuperVLAD's attack phase does not fit in 10 GB at the training-time batch of 16.
  # Batch size and gradient checkpointing affect memory, not metrics.
  --infer_batch_size=8
  --grad_checkpointing
)
