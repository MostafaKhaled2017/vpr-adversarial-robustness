# Run Adverserial training for BOQ model
./scripts/run_boq_vpr_adv_training.sh

# Evaluaion command for CricaVPR
python3 eval.py --eval_datasets_folder=/home/mostafa/git_repos/vpr-adversarial-robustness/datasets  \
                --eval_dataset_name=msls 
                --resume=/home/mostafa/git_repos/vpr-adversarial-robustness/checkpoints/CricaVPR.pth

python -m src.fgsm_train \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --resume=checkpoints/SuperVLAD.pth \
  --resume_model_only \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=8 \
  --lr=1e-5 \
  --patience=5 \
  --epochs_num=5 \
  --train_batch_size=32 \
  --adv_epsilon=0.001 \
  --adv_steps=3 \
  --adv_loss_weight=0.25 \
  --adv_align_weight=0.05 \
  --adv_warmup_epochs=2 \
  --adv_negatives=5

python3 fgsm_eval.py --eval_datasets_folder=datasets --eval_dataset_name=msls \
  --resume=logs/default/2026-04-19_14-15-55/checkpoint_epoch_0002.pth \
  --backbone=dino --supervlad_clusters=4 --crossimage_encoder \
  --infer_batch_size=32 --epsilons 0.01 0.1 0.2 --fgsm_loss positive_distance

python3 perceptual_eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls sped nordland \
  --base_resume=checkpoints/SuperVLAD.pth \
  --trained_resume=checkpoints/perceptual_adv_checkpoint.pth \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=8 \
  --infer_batch_size=16 \
  --test_method=hard_resize \
  --output_json=test/perceptual_eval/msls_sped_nordland_comparison.json \
  --output_csv=test/perceptual_eval/msls_sped_nordland_comparison.csv

python3 rank_eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls sped nordland \
  --model_type=supervlad \
  --model_paths checkpoints/SuperVLAD.pth checkpoints/perceptual_adv_checkpoint.pth \
  --model_tags base checkpoint \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=8 \
  --infer_batch_size=16 \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf \
  --rank_steps=20 \
  --rank_restarts=1 \
  --epsilons 0.01 0.1 \
  --output_json=test/rank_eval/msls_sped_nordland_rank_comparison.json \
  --output_csv=test/rank_eval/msls_sped_nordland_rank_comparison.csv

# BoQ base checkpoint versus an adversarially trained checkpoint.
python3 rank_eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls sped \
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
  --epsilons 0.01 0.1 \
  --output_json=test/rank_eval/boq_msls_sped_nordland_rank_comparison.json \
  --output_csv=test/rank_eval/boq_msls_sped_nordland_rank_comparison.csv

# MixVPR base checkpoint versus an adversarially trained checkpoint.
python3 rank_eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls sped \
  --model_type=mixvpr \
  --model_paths "third_party/VPR-methods-evaluation/trained_models/mixvpr/resnet50_MixVPR_4096_channels(1024)_rows(4)" logs/mixvpr_perceptual_adv_training/2026-07-12_23-35-14/checkpoint_epoch_0003.pth logs/mixvpr_perceptual_adv_training/2026-07-12_23-35-14/checkpoint_epoch_0005.pth logs/mixvpr_perceptual_adv_training/2026-07-12_23-35-14/checkpoint_epoch_0007.pth \
  --model_tags base adversarial_epoch_3 adversarial_epoch_5 adversarial_epoch_7 \
  --mixvpr_descriptors_dimension=4096 \
  --infer_batch_size=16 \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf \
  --rank_steps=20 \
  --rank_restarts=1 \
  --epsilons 0.01 0.1 \
  --output_json=test/rank_eval/mixvpr_msls_sped_nordland_rank_comparison.json \
  --output_csv=test/rank_eval/mixvpr_msls_sped_nordland_rank_comparison.csv

# Smoke test
python rank_eval.py \
  --eval_datasets_folder=datasets \
  --datasets msls \
  --model_type=supervlad \
  --model_paths checkpoints/SuperVLAD.pth \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=8 \
  --infer_batch_size=4 \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf \
  --rank_steps=3 \
  --rank_restarts=1 \
  --epsilons 0.01 \
  --max_queries 5 \
  --audit_attack_implementation \
  --audit_sample_database_size 64 \
  --output_json=test/rank_eval/phase1_audit_sample.json \
  --output_csv=test/rank_eval/phase1_audit_sample.csv

# Attack sweep
python3 scripts/run_rank_pgd_strength_sweep.py \
  --parallel_runs 1 \
  --infer_batch_size 8
#  --resume_sweep_dir test/rank_eval/sweeps/2026-07-06_13-17-09_full \

## To preview resume status only
python3 scripts/run_rank_pgd_strength_sweep.py \
  --resume_sweep_dir test/rank_eval/sweeps/2026-07-06_13-17-09_full \
  --dry_run

# Render the paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd docs/reports/paper/main.tex