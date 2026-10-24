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

# Training batch size in places (each place contributes 4 images). This is the single
# definition for every sprint 5 training run: Step 1 screens, full runs, clean twins,
# baselines, ablations and Step 4. It is deliberately not an environment override. The
# launchers refuse to reuse or resume a run trained at another size, and the Step 1 sweep
# records it in sweep_config.yaml. Change it only before any sprint 5 run exists.
# 20 is what the Step 1 sweep (logs/mplc_v2_step1, 2026-09-25/27) ran at.
SUPERVLAD_TRAIN_BATCH_SIZE=20

# Training batches per epoch, fixed the same way as the batch size. An epoch is
# 20 x 200 = 4,000 places. Step 1 tunes lr, depth and screen length in these epochs, so every
# later run must use the same epoch.
SUPERVLAD_BATCHES_PER_EPOCH=200

# At freeze_te=8 no gradient checkpointing is used, yet the training attacks backpropagate
# through all 12 blocks; batch 32 ran out of memory on a 24 GB GPU. Expandable segments
# recover the reserved-but-fragmented memory. Allocator setting only; results are unchanged.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Initialization. Default: fine-tune from the pretrained SuperVLAD release.
# Set SUPERVLAD_FROM_SCRATCH=1 to train the VLAD head from scratch on the DINOv2 backbone.
SUPERVLAD_INIT_FLAGS=(
  --resume="${SUPERVLAD_BASE_CHECKPOINT:-checkpoints/SuperVLAD_base.pth}"
  --resume_model_only
)
if [ "${SUPERVLAD_FROM_SCRATCH:-0}" = "1" ]; then
  SUPERVLAD_INIT_FLAGS=()
fi

# Every flag except initialization, attack configuration, and --seed/--save_dir. Keeping
# initialization separate lets a managed run replace it with --resume=<last_model> --continue.
SUPERVLAD_RECIPE_FLAGS=(
  --model=supervlad
  # Phase 1/Phase 2 runs select checkpoints with the original rule; scripts/mplc_v2_train.sh
  # overrides this with --validation_protocol=rank_pgd (the last occurrence wins).
  --validation_protocol=legacy
  --eval_datasets_folder=datasets
  --gsv_cities_base_path=datasets/gsv_cities
  --eval_dataset_name=msls
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
  --patience=8
  --batch_size="${SUPERVLAD_TRAIN_BATCH_SIZE}"
  --infer_batch_size=16
  --batches_per_epoch="${SUPERVLAD_BATCHES_PER_EPOCH}"
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
  --keep_every=0
  --val_batches=200
)

# Backwards-compatible complete flag list used by Phase 1 and standalone launchers.
SUPERVLAD_BASE_FLAGS=(
  "${SUPERVLAD_RECIPE_FLAGS[@]}"
  "${SUPERVLAD_INIT_FLAGS[@]}"
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

# Step 4 (runbook §4): BoQ and MixVPR trained with this same recipe. SHARED_RECIPE_FLAGS is
# SUPERVLAD_RECIPE_FLAGS minus its SuperVLAD-only model flags and image sizes, so the recipe is
# still defined once above; each model adds its own model flags and sizes.
SHARED_RECIPE_FLAGS=()
_skip=0
for _flag in "${SUPERVLAD_RECIPE_FLAGS[@]}"; do
  if [ "${_skip}" -gt 0 ]; then
    _skip=$((_skip - 1))
    continue
  fi
  case "${_flag}" in
    --model=* | --foundation_model_path=* | --backbone=* | --supervlad_clusters=* | --crossimage_encoder | --freeze_te=*) ;;
    --train_resize | --resize) _skip=2 ;;
    *) SHARED_RECIPE_FLAGS+=("${_flag}") ;;
  esac
done
unset _flag _skip

# Initial weights: the official releases.
BOQ_BASE_CHECKPOINT="${BOQ_BASE_CHECKPOINT:-checkpoints/boq_dinov2_12288.pth}"
MIXVPR_BASE_CHECKPOINT="${MIXVPR_BASE_CHECKPOINT:-third_party/VPR-methods-evaluation/trained_models/mixvpr/resnet50_MixVPR_4096_channels(1024)_rows(4)}"

# Each model trains at the size its release was trained at and validates at its evaluation
# size: BoQ (DINOv2) 224 / 322 (third_party/Bag-of-Queries/train.py), MixVPR 320 / 320.
BOQ_MODEL_FLAGS=(
  --model=boq
  --boq_backbone=Dinov2
  --boq_descriptors_dimension=12288
  --train_resize 224 224
  --resize 322 322
)
MIXVPR_MODEL_FLAGS=(
  --model=mixvpr
  --mixvpr_descriptors_dimension=4096
  --train_resize 320 320
  --resize 320 320
)

# Model flags eval.py needs to rebuild a BoQ or MixVPR checkpoint. Batch size and gradient
# checkpointing affect memory, not metrics; checkpointing only wraps DINOv2 blocks.
BOQ_EVAL_MODEL_FLAGS=(
  --model_type=boq
  --boq_backbone=Dinov2
  --boq_descriptors_dimension=12288
  --infer_batch_size=8
  --grad_checkpointing
)
MIXVPR_EVAL_MODEL_FLAGS=(
  --model_type=mixvpr
  --mixvpr_descriptors_dimension=4096
  --infer_batch_size=8
)

# DINOMix and QAA (Step 4, sprint 6): DINOv2 ViT-B/14 with MixVPR's and QAA's aggregators. Their
# initial weights are the releases written by src/fetch_pretrained.py. DINOMix runs at 224 only
# (its MixVPR aggregator expects a 16 x 16 token grid); QAA trains at 224 and validates and
# evaluates at 322, as its release (and BoQ) does.
DINOMIX_BASE_CHECKPOINT="${DINOMIX_BASE_CHECKPOINT:-checkpoints/dinomix_dinov2_4096.pth}"
QAA_BASE_CHECKPOINT="${QAA_BASE_CHECKPOINT:-checkpoints/qaa_dinov2_8192.pth}"
DINOMIX_MODEL_FLAGS=(
  --model=dinomix
  --train_resize 224 224
  --resize 224 224
)
QAA_MODEL_FLAGS=(
  --model=qaa
  --train_resize 224 224
  --resize 322 322
)
DINOMIX_EVAL_MODEL_FLAGS=(
  --model_type=dinomix
  --infer_batch_size=8
  --grad_checkpointing
)
QAA_EVAL_MODEL_FLAGS=(
  --model_type=qaa
  --infer_batch_size=8
  --grad_checkpointing
)
