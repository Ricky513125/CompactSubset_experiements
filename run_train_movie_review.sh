#!/bin/bash

# 豆瓣影评模型训练脚本
# 用法: bash run_train_movie_review.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 配置参数
NUM_GPUS=8
MASTER_PORT=29501
CONFIG_FILE="config_MovieReview.json"
ABLATION_CONFIG="profile_and_history"  # 可选: profile_and_history, profile_only, history_only, baseline
OUTPUT_DIR="outputs/MovieReview_${ABLATION_CONFIG}_$(date +%m%d_%H%M)"

# 训练参数
MAX_EPOCHS=50
EARLY_STOPPING_PATIENCE=3
VAL_RATIO=0.15

# W&B配置
WANDB_PROJECT="MovieReview-Experiment"
WANDB_RUN_NAME="${ABLATION_CONFIG}_$(date +%m%d_%H%M)"

# DeepSpeed配置（可选）
# DEEPSPEED_CONFIG="ds_config_zero2.json"

echo "================================"
echo "豆瓣影评模型训练"
echo "================================"
echo "GPU数量: ${NUM_GPUS}"
echo "消融配置: ${ABLATION_CONFIG}"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================"

# 启动训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train_distributed_MovieReview.py \
    --config ${CONFIG_FILE} \
    --ablation_config ${ABLATION_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --max_epochs ${MAX_EPOCHS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --val_ratio ${VAL_RATIO} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --prompt_style simple
    # --deepspeed ${DEEPSPEED_CONFIG}  # 如果需要使用DeepSpeed，取消注释

echo ""
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
