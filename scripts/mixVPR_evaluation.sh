cd third_party/VPR-methods-evaluation

python3 main.py \
  --method mixvpr \
  --backbone ResNet50 \
  --descriptors_dimension 4096 \
  --database_folder ../../datasets/msls/images/val/database \
  --queries_folder ../../datasets/msls/images/val/queries \
  --image_size 320 320 \
  --device cuda \
  --batch_size 16 \
  --num_workers 4 \
  --positive_dist_threshold 25 \
  --recall_values 1 5 10 100 \
  --log_dir mixvpr_4096_msls_val

python3 main.py \
  --method mixvpr \
  --backbone ResNet50 \
  --descriptors_dimension 4096 \
  --database_folder ../../datasets/sped/images/test/database \
  --queries_folder ../../datasets/sped/images/test/queries \
  --image_size 320 320 \
  --device cuda \
  --batch_size 16 \
  --num_workers 4 \
  --positive_dist_threshold 25 \
  --recall_values 1 5 10 100 \
  --log_dir mixvpr_4096_sped_test