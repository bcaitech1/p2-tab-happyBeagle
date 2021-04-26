# !/bin/sh

python3 run.py \
--model lgbm \
--output_path /opt/ml/code/output/my_run_try_ver4.csv \
--trainset_path /opt/ml/code/my_src/data/train_data_thres_rate_3_6_12.csv \
--testset_path /opt/ml/code/my_src/data/test_data_thres_rate_3_6_12.csv \
--outputform_path /opt/ml/code/input/sample_submission.csv 