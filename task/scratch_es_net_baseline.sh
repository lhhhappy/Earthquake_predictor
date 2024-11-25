#!/bin/bash

# Set the data path and save directory
Experiment_name="Scratch_ES_net_mixer_california_Mlpbaseline"

DATA_PATH="/home/linhang/workbench/Earthquake_data/"

SAVE_DIR="/home/linhang/workbench/workbench/Earthquake_predictor/Result/Scratch/"

LOG_DIR="${SAVE_DIR}${Experiment_name}/logs"
MODEL_DIR="${SAVE_DIR}${Experiment_name}/checkpoints"

# Set model parameters as a JSON string

predict_window=14
time_resolution=14
gnss_history_window=140
earthquake_history_window_day=1400
pdm_d_model=32
dropout=0.3

output_window=$((predict_window / time_resolution))
predict_day_class=0
pdm_d_ff=$((pdm_d_model * 2))
earthquake_history_window=$((earthquake_history_window_day / time_resolution))

cat <<EOF > model_params.json
{
    "earthquake_dim": 1,
    "gnss_dim": 4,
    "embed_dim": 256,
    "earthquake_history_window": $earthquake_history_window,
    "gnss_history_window": $gnss_history_window,
    "gnss_station_num": 177,
    "earthquake_num": 100,
    "dropout": $dropout
}
EOF

# 清理旧日志目录
rm -r "${SAVE_DIR}${Experiment_name}/"

# 创建实验目录
EXPERIMENTS_DIR="${SAVE_DIR}${Experiment_name}/Hyperparameters"
mkdir -p $EXPERIMENTS_DIR

# 存储当前脚本文件和 model_params.json 两份
cp "$0" "$EXPERIMENTS_DIR/$(basename $0)"
cp model_params.json "$EXPERIMENTS_DIR/model_params.json"


# Run the training script with specified arguments
python train.py \
    --data-path $DATA_PATH \
    --model-arch "Mlpbaseline" \
    --energy-loss "tss" \
    --day-loss "None" \
    --batch-size 4 \
    --val-batch-size 4 \
    --max-epochs 400 \
    --device 1 \
    --lr 5e-5 \
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
    --start-date "2017-01-01" \
    --last-date "2024-01-01" \
    --val-date "2021-09-04" \
    --train-percentage 0.7 \
    --seed 0 \
    --use-area "California (Southern)"

