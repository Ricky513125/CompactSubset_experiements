#!/bin/bash
# DMSC 影评数据集训练脚本
# 注意：DMSC 数据加载器已自动按时序生成训练样本，无需额外扩充
# 使用方法: bash run_dmsc_with_augmentation.sh

set -e

# 配置参数
CONFIG="config_DMSC.json"
DEEPSPEED="ds_config_zero2.json"
ABLATION="history_only"
OUTPUT_DIR="outputs/DMSC_history_augmented_0211"
WANDB_PROJECT="Qwen3-DMSC-Augmented"
WANDB_RUN="history_aug_0211"
NUM_GPUS=8
MASTER_PORT=29500

echo "========================================"
echo "DMSC 影评数据训练"
echo "========================================"
echo "配置文件: $CONFIG"
echo "DeepSpeed配置: $DEEPSPEED"
echo "消融配置: $ABLATION"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NUM_GPUS"
echo "========================================"
echo ""
echo "📝 注意: DMSC 影评数据加载器已自动按时序生成训练样本"
echo "  每条影评 → 1个训练样本（包含之前所有影评作为历史）"
echo "  无需额外的时序扩充"
echo "========================================"
echo ""

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# 启动训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_distributed_MovieReview.py \
    --config $CONFIG \
    --deepspeed $DEEPSPEED \
    --ablation_config $ABLATION \
    --output_dir $OUTPUT_DIR \
    --max_epochs 50 \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN \
    --prompt_style simple

echo ""
echo "========================================"
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "========================================"
