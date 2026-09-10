#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="${PWD}:${PWD}/third_party/SuperVLAD:${PYTHONPATH:-}"

python3 eval.py \
  --eval_datasets_folder=datasets \
  --datasets sped \
  --model_type=supervlad \
  --model_paths logs/default/2026-07-31_22-57-12/best_model.pth \
  --model_tags trained \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --infer_batch_size=8 \
  --rank_attack=rank_pgd_linf \
  --rank_steps=20 \
  --rank_restarts=1 \
  --grad_checkpointing \
  --epsilons 0.01 0.1 \
  --output_json=output/supervlad_sped_eps_0.01_0.1_rank_comparison.json \
  --output_csv=output/supervlad_sped_eps_0.01_0.1_rank_comparison.csv
