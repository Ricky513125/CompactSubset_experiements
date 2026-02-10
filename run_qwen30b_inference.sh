#!/bin/bash
# Qwen 30B 模型推理启动脚本
# 使用方法: bash run_qwen30b_inference.sh [checkpoint_path] [output_name]
# 示例: bash run_qwen30b_inference.sh outputs/Qwen30B_RealPersonaChat_profile_context_0210/checkpoint-best inference_0210

set -e  # 遇到错误立即退出

# 默认参数
CHECKPOINT_PATH=${1:-"outputs/Qwen30B_RealPersonaChat_profile_context_0210/checkpoint-best"}
OUTPUT_NAME=${2:-"inference_0210"}
NUM_GPUS=${3:-8}
MASTER_PORT=${4:-29501}
BATCH_SIZE=${5:-1}

# 模型路径
MODEL_PATH="/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"

# 测试数据路径
TEST_DATA_PATH="/mnt/parallel/GIDigitalTwinBench/IdealSelf/RealPersonaChat/test.json"

# 输出目录
OUTPUT_DIR="results/Qwen30B_${OUTPUT_NAME}"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径 $MODEL_PATH 不存在！"
    exit 1
fi

# 检查测试数据
if [ ! -f "$TEST_DATA_PATH" ]; then
    echo "错误: 测试数据 $TEST_DATA_PATH 不存在！"
    exit 1
fi

# 打印配置信息
echo "========================================"
echo "Qwen 30B 模型推理配置"
echo "========================================"
echo "GPU数量: $NUM_GPUS"
echo "主端口: $MASTER_PORT"
echo "模型路径: $MODEL_PATH"
echo "检查点路径: $CHECKPOINT_PATH"
echo "测试数据: $TEST_DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "========================================"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

echo "开始推理..."
echo ""

# 启动分布式推理
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    inference_distributed_30b.py \
    --model_path $MODEL_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --test_data_path $TEST_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length 8192

echo ""
echo "推理完成！"
echo "结果保存在: $OUTPUT_DIR"
