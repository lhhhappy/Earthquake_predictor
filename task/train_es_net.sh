#!/bin/bash

# Set the data path and save directory
DATA_PATH="/home/linhang/workbench/Earthquake_data/"
SAVE_DIR="/home/linhang/workbench/workbench/Earthquake_predictor/Result/"
LOG_DIR="${SAVE_DIR}logs/"

# Set model parameters as a JSON string

predict_window=14
time_resolution=14
history_window=140

input_window=$((history_window / time_resolution))
output_window=$((predict_window / time_resolution))
predict_day_class=$((predict_window + 1))

cat <<EOF > model_params.json
{
    "feature_dim": 1,
    "ext_dim": 0,
    "gnss_feature_dim": 4,
    "embed_dim": 64,
    "skip_dim": 64,
    "lape_dim": 30,
    "geo_num_heads": 2,
    "sem_num_heads": 2,
    "t_num_heads": 4,
    "mlp_ratio": 2,
    "qkv_bias": true,
    "drop": 0.0,
    "attn_drop": 0.0,
    "drop_path": 0.3,
    "s_attn_size": 3,
    "t_attn_size": 3,
    "enc_depth": 2,
    "type_ln": "pre",
    "output_dim": 1,
    "input_window": $input_window,
    "output_window": $output_window,
    "predict_day_class": $predict_day_class,
    "far_mask_delta": 0,
    "dtw_delta": 50
}
EOF

# Run the training script with specified arguments
python train.py \
    --data-path $DATA_PATH \
    --model-arch "ES_net" \
    --energy-loss "mse" \
    --day-loss "cross_entropy" \
    --batch-size 16 \
    --val-batch-size 10 \
    --max-epochs 100 \
    --device 2 \
    --lr 1e-8 \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR \
    --model_params "model_params.json" \
    --history-window $history_window \
    --forecast-window  $predict_window \
    --lape-dim 30 \
    --far-mask-delta 30 \
    --dtw-delta 10 \
    --time-resolution $time_resolution