#!/bin/bash

# Chameleons 快速训练测试脚本（使用 data_item 采样数据）
# 使用 train_di3.json (16,963 个训练样本，相比原始减少 76.6%)

echo "================================================================================"
echo "Chameleons 快速训练测试 (data_item 采样版本)"
echo "================================================================================"
echo "数据集: sampled_data/Chameleons/train_di3.json"
echo "训练样本数: ~16,963 (原始: 72,471)"
echo "采样方式: 每用户最多 3 个 data_item (训练样本)"
echo "预计训练时间: 原始的 ~23%"
echo "================================================================================"
echo ""

# 配置
CONFIG="config_Chameleons_30B_di3.json"
DEEPSPEED="ds_config_zero3_optimized.json"
ABLATION="context_only"
OUTPUT_DIR="outputs/Chameleons_context_30B_di3_test"
PROJECT="Qwen3_30B-Chameleons"
RUN_NAME="context_di3_seed42_test"
MASTER_PORT=29502

# 训练命令
echo "开始训练..."
torchrun \
    --nproc_per_node=8 \
    --master_port=${MASTER_PORT} \
    train_distributed_Chameleons.py \
    --config ${CONFIG} \
    --deepspeed ${DEEPSPEED} \
    --ablation_config ${ABLATION} \
    --output_dir ${OUTPUT_DIR} \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project ${PROJECT} \
    --wandb_run_name ${RUN_NAME} \
    --prompt_style simple

echo ""
echo "================================================================================"
echo "训练完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================================================================"
