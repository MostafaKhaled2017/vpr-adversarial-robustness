#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls sped \
  --model_type=boq \
  --model_paths checkpoints/boq_adv_trained.pth \
  --model_tags trained \
  --boq_backbone=Dinov2 \
  --boq_descriptors_dimension=12288 \
  --infer_batch_size=8 \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf \
  --rank_steps=20 \
  --rank_restarts=1 \
  --epsilons 0.01 0.1 \
  --output_json=output/boq_msls_sped_eps_0.01_rank_comparison.json \
  --output_csv=output/boq_msls_sped_eps_0.01_rank_comparison.csv