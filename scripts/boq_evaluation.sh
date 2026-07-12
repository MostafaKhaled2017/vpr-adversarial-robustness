cd third_party/VPR-methods-evaluation

python3 main.py \
  --method boq \
  --backbone Dinov2 \
  --database_folder ../../datasets/msls/images/val/database \
  --queries_folder ../../datasets/msls/images/val/queries \
  --device cuda \
  --batch_size 16 \
  --num_workers 4 \
  --positive_dist_threshold 25 \
  --recall_values 1 5 10 100 \
  --log_dir boq_dinov2_msls_val

python3 main.py \
  --method boq \
  --backbone Dinov2 \
  --database_folder ../../datasets/sped/images/test/database \
  --queries_folder ../../datasets/sped/images/test/queries \
  --device cuda \
  --batch_size 16 \
  --num_workers 4 \
  --positive_dist_threshold 25 \
  --recall_values 1 5 10 100 \
  --log_dir boq_dinov2_sped_test