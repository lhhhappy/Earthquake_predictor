#!/bin/bash

# Set the data path and save directory
DATA_PATH="/data2/linhang_data/Earthquake_data/data/"
SAVE_DIR="/data2/linhang_data/Earthquake_results/ES_net/"
LOG_DIR="${SAVE_DIR}logs/"

# Set model parameters as a JSON string
cat <<EOF > model_params.json
{
    "feature_dim": 1,
    "ext_dim": 0,
    "gnss_feature_dim": 4,
    "embed_dim": 16,
    "skip_dim": 16,
    "lape_dim": 30,
    "geo_num_heads": 4,
    "sem_num_heads": 2,
    "t_num_heads": 2,
    "mlp_ratio": 1,
    "qkv_bias": true,
    "drop": 0.0,
    "attn_drop": 0.0,
    "drop_path": 0.3,
    "s_attn_size": 3,
    "t_attn_size": 3,
    "enc_depth": 1,
    "type_ln": "pre",
    "output_dim": 1,
    "input_window": 1400,
    "output_window": 1,
    "predict_day_class": 14,
    "far_mask_delta": 0,
    "dtw_delta": 50
}
EOF

# Run the training script with specified arguments
python train.py \
    --data-path $DATA_PATH \
    --model-arch "ES_net" \
    --energy-loss "nnse" \
    --day-loss "cross_entropy" \
    --batch-size 1 \
    --val-batch-size 1 \
    --max-epochs 100 \
    --device 0 \
    --lr 1e-5 \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR \
    --model_params "model_params.json" \
    --window-size 1400 \
    --forecast-horizon 14 \
    --lape-dim 30 \
    --far-mask-delta 30 \
    --dtw-delta 10