#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python3 rank_eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls \
  --model_type=boq \
  --model_paths checkpoints/boq_dinov2_12288.pth logs/boq_perceptual_adv_training/2026-07-13_07-23-11/checkpoint_epoch_0003.pth logs/boq_perceptual_adv_training/2026-07-13_07-23-11/checkpoint_epoch_0005.pth \
  --model_tags base adversarial_epoch_3 adversarial_epoch_5 \
  --boq_backbone=Dinov2 \
  --boq_descriptors_dimension=12288 \
  --infer_batch_size=16 \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf \
  --rank_steps=20 \
  --rank_restarts=1 \
  --epsilons 0.01 \
  --output_json=test/rank_eval/boq_msls_eps_0.01_rank_comparison.json \
  --output_csv=test/rank_eval/boq_msls_eps_0.01_rank_comparison.csv

python3 rank_eval.py \
  --eval_datasets_folder=datasets \
  --datasets sped \
  --model_type=mixvpr \
  --model_paths "third_party/VPR-methods-evaluation/trained_models/mixvpr/resnet50_MixVPR_4096_channels(1024)_rows(4)" logs/mixvpr_perceptual_adv_training/2026-07-12_23-35-14/checkpoint_epoch_0003.pth logs/mixvpr_perceptual_adv_training/2026-07-12_23-35-14/checkpoint_epoch_0005.pth \
  --model_tags base adversarial_epoch_3 adversarial_epoch_5 \
  --mixvpr_descriptors_dimension=4096 \
  --infer_batch_size=16 \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf \
  --rank_steps=20 \
  --rank_restarts=1 \
  --epsilons 0.1 \
  --output_json=test/rank_eval/mixvpr_sped_eps_0.1_rank_comparison.json \
  --output_csv=test/rank_eval/mixvpr_sped_eps_0.1_rank_comparison.csv