# Flags shared by scripts/train.sh and scripts/eval.sh (sourced, not run).

TRAIN_RECIPE_FLAGS=(
  --eval_datasets_folder=datasets
  --gsv_cities_base_path=datasets/gsv_cities
  --eval_dataset_name=msls
  --optim=adam
  --lr_schedule=120
  --lr_plateau_factor=0.1
  --batch_size=20
  --batches_per_epoch=200
  --infer_batch_size=16
  --mixed_precision
  --test_method=hard_resize
  --randomize_attack
  --adv_negatives=5
  --adv_warmup_epochs=1
  --adv_margin=0.1
  --adv_loss_weight=1.0
  --adv_align_weight=0.05
  --early_stop_min_delta=0.0
  --keep_every=0
  --val_every=3
  --val_rank_epsilons 0.0342 0.0685
  # Clean budgets only decide which extra best_model_budget<b>.pth files are kept.
  --selection_clean_budgets 1 3 5 7 10
)

# Each model trains at its release's training size and validates at its test size.
SUPERVLAD_TRAIN_FLAGS=(
  --model=supervlad
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
  --backbone=dino
  --supervlad_clusters=4
  --crossimage_encoder
  --train_resize 322 322
  --resize 322 322
)
BOQ_TRAIN_FLAGS=(
  --model=boq
  --boq_backbone=Dinov2
  --boq_descriptors_dimension=12288
  --train_resize 224 224
  --resize 322 322
)
SUPERVLAD_PRETRAINED=checkpoints/SuperVLAD_base.pth
BOQ_PRETRAINED=checkpoints/boq_dinov2_12288.pth

# Model flags eval.py needs to rebuild a checkpoint. Batch size and gradient checkpointing
# change memory use, not results.
SUPERVLAD_EVAL_FLAGS=(
  --model_type=supervlad
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
  --backbone=dino
  --supervlad_clusters=4
  --crossimage_encoder
  --freeze_te=8
  --infer_batch_size=8
  --grad_checkpointing
)
BOQ_EVAL_FLAGS=(
  --model_type=boq
  --boq_backbone=Dinov2
  --boq_descriptors_dimension=12288
  --infer_batch_size=4
  --grad_checkpointing
)

# Training at freeze_te 0 backpropagates through all 12 DINOv2 blocks. Allocator setting only.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
