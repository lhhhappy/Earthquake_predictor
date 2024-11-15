#!/bin/bash

# Set the data path and save directory
DATA_PATH="/home/linhang/workbench/Earthquake_data/"
SAVE_DIR="/home/linhang/workbench/workbench/Earthquake_predictor/Result/"
LOG_DIR="${SAVE_DIR}logs/"

# Set model parameters as a JSON string

predict_window=14
time_resolution=14
gnss_history_window=140
earthquake_history_window=700


output_window=$((predict_window / time_resolution))
predict_day_class=$((predict_window + 1))

cat <<EOF > model_params.json
{
{
    "earthquake_dim": 1,
    "gnss_dim": 4,
    "embed_dim": 64,
    "lape_dim": 30,
    "gnss_history_window": $gnss_history_window,
    "earthquake_history_window": $earthquake_history_window,
    "down_sampling_method": "avg",
    "down_sampling_window": 2,
    "down_sampling_layers": 3,
    "pdm_layers": 2,
    "pdm_d_model": 64,
    "pdm_d_ff": 128,
    "pdm_dropout": 0.1,
    "pdm_decomp_method": "moving_avg",
    "pdm_moving_avg_kernel": 3,
    "geo_num_heads": 2,
    "sem_num_heads": 2,
    "qkv_bias": true,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "mlp_ratio": 2.0,
    "enc_depth": 2,
    "type_ln": "pre",
    "prediction_day_head": $predict_day_class,
    "channel_independence": 0
    "predict_energy_len": $output_window,
}
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
    --lr 1e-4 \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR \
    --model_params "model_params.json" \
    --history-window $gnss_history_window \
    --forecast-window  $predict_window \
    --lape-dim 30 \
    --geo-percentage 0.3 \
    --sem-percentage 0.3 \
    --time-resolution $time_resolution \
    --earthquake-catalog-window $earthquake_history_window \    