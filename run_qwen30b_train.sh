#!/bin/bash
# Qwen 30B 模型训练启动脚本
# 使用方法: bash run_qwen30b_train.sh [ablation_config] [output_suffix]
# 示例: bash run_qwen30b_train.sh profile_and_context v1

set -e  # 遇到错误立即退出

# 默认参数
ABLATION_CONFIG=${1:-"profile_and_context"}
OUTPUT_SUFFIX=${2:-"0210"}
NUM_GPUS=${3:-8}
MASTER_PORT=${4:-29500}

# 配置文件路径
CONFIG_FILE="config_RealPersonaChat_Qwen30B.json"
DEEPSPEED_CONFIG="ds_config_zero3_30b.json"

# 输出目录
OUTPUT_DIR="outputs/Qwen30B_RealPersonaChat_${ABLATION_CONFIG}_${OUTPUT_SUFFIX}"

# W&B 配置
WANDB_PROJECT="Qwen30B-RealPersonaChat"
WANDB_RUN_NAME="${ABLATION_CONFIG}_${OUTPUT_SUFFIX}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在！"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "错误: DeepSpeed配置文件 $DEEPSPEED_CONFIG 不存在！"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "train_distributed_RealPersonaChat.py" ]; then
    echo "错误: 训练脚本 train_distributed_RealPersonaChat.py 不存在！"
    exit 1
fi

# 打印配置信息
echo "========================================"
echo "Qwen 30B 模型训练配置"
echo "========================================"
echo "GPU数量: $NUM_GPUS"
echo "主端口: $MASTER_PORT"
echo "配置文件: $CONFIG_FILE"
echo "DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "消融实验: $ABLATION_CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "W&B项目: $WANDB_PROJECT"
echo "W&B运行名: $WANDB_RUN_NAME"
echo "========================================"
echo ""

# 设置环境变量优化性能
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false

# 可选：启用 NCCL 调试（如果遇到通信问题）
# export NCCL_DEBUG=INFO

echo "开始训练..."
echo ""

# 启动训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_distributed_RealPersonaChat.py \
    --config $CONFIG_FILE \
    --deepspeed $DEEPSPEED_CONFIG \
    --ablation_config $ABLATION_CONFIG \
    --output_dir $OUTPUT_DIR \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --prompt_style simple

echo ""
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"
