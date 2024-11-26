#!/bin/bash

# Set the data path and save directory
Experiment_name="Pretrain_ES_net_mixer_onlyhappen"

DATA_PATH="/home/linhang/workbench/Earthquake_data/"

SAVE_DIR="/home/linhang/workbench/workbench/Earthquake_predictor/Result/Pretrain/"

LOG_DIR="${SAVE_DIR}${Experiment_name}/logs"
MODEL_DIR="${SAVE_DIR}${Experiment_name}/checkpoints"

# Set model parameters as a JSON string

predict_window=14
time_resolution=14
gnss_history_window=140
earthquake_history_window_day=1400
pdm_d_model=32
dropout=0.5

output_window=$((predict_window / time_resolution))
predict_day_class=0
pdm_d_ff=$((pdm_d_model * 2))
earthquake_history_window=$((earthquake_history_window_day / time_resolution))

cat <<EOF > model_params.json
{
    "earthquake_dim": 1,
    "gnss_dim": 4,
    "embed_dim": $pdm_d_model,
    "lape_dim": 30,
    "earthquake_history_window": $earthquake_history_window,
    "down_sampling_method": "avg",
    "down_sampling_window": 2,
    "down_sampling_layers": 3,
    "pdm_layers": 2,
    "pdm_d_model": $pdm_d_model,
    "pdm_d_ff": $pdm_d_ff,
    "pdm_dropout": $dropout,
    "pdm_decomp_method": "moving_avg",
    "pdm_moving_avg_kernel": 3,
    "geo_num_heads": 2,
    "sem_num_heads": 2,
    "qkv_bias": true,
    "attn_drop": $dropout,
    "proj_drop": $dropout,
    "mlp_ratio": 1.0,
    "enc_depth": 1,
    "type_ln": "pre",
    "prediction_day_head": $predict_day_class,
    "prediction_energy_len": $output_window
}
EOF

# 清理旧日志目录
rm -r $LOG_DIR
rm -r $MODEL_DIR
# 创建实验目录
EXPERIMENTS_DIR="${SAVE_DIR}${Experiment_name}/Hyperparameters"
mkdir -p $EXPERIMENTS_DIR

# 存储当前脚本文件和 model_params.json 两份
cp "$0" "$EXPERIMENTS_DIR/$(basename $0)"
cp model_params.json "$EXPERIMENTS_DIR/model_params.json"


# Run the training script with specified arguments
python train.py \
    --data-path $DATA_PATH \
    --model-arch "ES_net_mixer" \
    --energy-loss "tss" \
    --day-loss "None" \
    --batch-size 4 \
    --val-batch-size 4 \
    --max-epochs 100 \
    --device 1 \
    --lr 2e-5 \
    --save-dir $MODEL_DIR \
    --log-dir $LOG_DIR \
    --model_params "model_params.json" \
    --history-window $gnss_history_window \
    --forecast-window  $predict_window \
    --lape-dim 30 \
    --geo-percentage 0.3 \
    --sem-percentage 0.3 \
    --time-resolution $time_resolution \
    --earthquake-catalog-window $earthquake_history_window_day \
    --train-percentage 0.8 \
    --use-area "California (Southern)" \
    --spilt-by-earthquake-happened True \
    --use-train-loader "train_happen" \
    --use-val-loader "val_happen" 

