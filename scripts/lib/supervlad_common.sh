#!/usr/bin/env bash
# Shared SuperVLAD configuration for Phases 1 and 2 of the full-conference plan.
#
# Phase 1 requires the clean-only control and the adversarial (PAT) arm to be *matched*:
# identical data, epochs, optimizer, LR schedule, frozen layers, and checkpoint-selection
# rule, differing only in the attack configuration (plan, Global Constraints). Defining the
# flag list once and appending SUPERVLAD_ATTACK_FLAGS only for the adversarial arm makes that
# matching structural rather than something a reader has to diff by eye.
#
# PROVENANCE: these flags reproduce scripts/super_vlad_train.sh, which was verified to
# reproduce the run metadata of logs/default/2026-07-27_10-29-47 with an empty diff. That is
# the project's standard SuperVLAD recipe: fine-tuning FROM the pretrained SuperVLAD release
# (--resume=checkpoints/SuperVLAD_base.pth --resume_model_only), which is what 4 of the 5
# recorded SuperVLAD runs did.
#
# This matters for Phase 1's design. Because training starts from SuperVLAD_base.pth, the
# "pretrained" evaluation arm is the *same lineage* as clean-FT and PAT, so
# pretrained -> clean-FT is a genuine fine-tuning ablation and clean-FT -> PAT isolates
# adversarial training. Starting from the DINOv2 backbone with a fresh VLAD head instead
# (SUPERVLAD_FROM_SCRATCH=1 below, i.e. scripts/super_vlad_scratch_train.sh, which produced
# logs/default/2026-07-31_22-57-12) would break that: the pretrained arm would then be a
# different training lineage rather than an un-fine-tuned control.
#
# Values that happen to equal a current src/cli.py default are still written out explicitly:
# a matched-control experiment must not silently change when a default changes.

# Initialization. Default: fine-tune from the pretrained SuperVLAD release.
# Set SUPERVLAD_FROM_SCRATCH=1 to train the VLAD head from scratch on the DINOv2 backbone.
SUPERVLAD_INIT_FLAGS=(
  --resume="${SUPERVLAD_BASE_CHECKPOINT:-checkpoints/SuperVLAD_base.pth}"
  --resume_model_only
)
if [ "${SUPERVLAD_FROM_SCRATCH:-0}" = "1" ]; then
  SUPERVLAD_INIT_FLAGS=()
fi

# Every flag except the attack configuration and --seed/--save_dir. Shared by BOTH arms.
SUPERVLAD_BASE_FLAGS=(
  --model=supervlad
  --eval_datasets_folder=datasets
  --gsv_cities_base_path=datasets/gsv_cities
  --eval_dataset_name=msls
  "${SUPERVLAD_INIT_FLAGS[@]}"
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
  --backbone=dino
  --supervlad_clusters=4
  --crossimage_encoder
  --freeze_te=8
  --optim=adam
  --lr=1e-5
  --lr_schedule=120
  --lr_plateau_patience=5
  --lr_plateau_factor=0.1
  --num_epochs=100
  --patience=12
  --batch_size=16
  --infer_batch_size=16
  --batches_per_epoch=400
  --mixed_precision
  --train_resize 322 322
  --resize 322 322
  --test_method=hard_resize
  --randomize_attack
  --adv_negatives=5
  --adv_warmup_epochs=1
  --adv_margin=0.1
  --adv_loss_weight=1.0
  --adv_align_weight=0.05
  --selection_robust_weight=0.75
  --early_stop_min_delta=0.0
  --keep_every=6
  --val_batches=200
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
