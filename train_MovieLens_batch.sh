#!/bin/bash

set -euo pipefail

# ===== 配置参数 =====
# 训练参数
CONFIG_FILE="config_MovieLens.json"
DEEPSPEED_CONFIG="ds_config_zero2.json"
ABLATION_CONFIG="history_only"
HISTORY_STRATEGY="random"
HISTORY_RATIO=0.5
OUTPUT_DIR="outputs/MovieLens_history_0221_0"
MAX_EPOCHS=50
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_THRESHOLD=0.001
VAL_RATIO=0.1
WANDB_PROJECT="Qwen3-MovieLens"
WANDB_RUN_NAME="history_0221_0"
PROMPT_STYLE="simple"

# 分批训练参数
BATCH_SIZE=10000  # 每个批次的数据量（可根据实际情况调整）
MASTER_PORT_BASE=29500  # 基础端口号，每个批次会递增

# GPU 配置
NPROC_PER_NODE=8

# ===== 设置环境变量 =====
# 设置 Triton 缓存目录（避免磁盘空间不足）
export TRITON_CACHE_DIR="/mnt/parallel/CompactSubset_experiement/.cache/triton"
export XDG_CACHE_HOME="/mnt/parallel/CompactSubset_experiement/.cache"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"

echo "===== 分批训练配置 ====="
echo "批次大小: $BATCH_SIZE"
echo "Triton 缓存目录: $TRITON_CACHE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# ===== 第一步：处理数据并确定总批次数 =====
echo "===== 步骤1: 处理数据并确定总批次数 ====="

# 运行第一个批次（会自动处理所有数据并保存）
echo "运行第一批次（处理所有数据）..."
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT_BASE \
    train_distributed_MovieLens.py \
    --config "$CONFIG_FILE" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --ablation_config "$ABLATION_CONFIG" \
    --history_strategy "$HISTORY_STRATEGY" \
    --history_ratio $HISTORY_RATIO \
    --output_dir "$OUTPUT_DIR" \
    --max_epochs $MAX_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_threshold $EARLY_STOPPING_THRESHOLD \
    --val_ratio $VAL_RATIO \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "${WANDB_RUN_NAME}_batch0" \
    --prompt_style "$PROMPT_STYLE" \
    --data_batch_size $BATCH_SIZE \
    --data_batch_index 0

if [ $? -ne 0 ]; then
    echo "错误: 第一批次训练失败"
    exit 1
fi

echo "✓ 第一批次训练完成"
echo ""

# ===== 第二步：计算总批次数 =====
echo "===== 步骤2: 计算总批次数 ====="

ALL_SAMPLES_FILE="all_samples.json"
if [ ! -f "$ALL_SAMPLES_FILE" ]; then
    echo "错误: 找不到 $ALL_SAMPLES_FILE，无法计算总批次数"
    exit 1
fi

# 使用 Python 计算总样本数和批次数
TOTAL_SAMPLES=$(python3 << EOF
import json
with open('$ALL_SAMPLES_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
EOF
)

TOTAL_BATCHES=$(( (TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "总样本数: $TOTAL_SAMPLES"
echo "批次大小: $BATCH_SIZE"
echo "总批次数: $TOTAL_BATCHES"
echo ""

# ===== 第三步：训练剩余批次 =====
if [ $TOTAL_BATCHES -gt 1 ]; then
    echo "===== 步骤3: 训练剩余批次 (1 到 $((TOTAL_BATCHES - 1))) ====="
    
    for BATCH_INDEX in $(seq 1 $((TOTAL_BATCHES - 1))); do
        MASTER_PORT=$((MASTER_PORT_BASE + BATCH_INDEX))
        
        echo ""
        echo "=========================================="
        echo "训练第 $((BATCH_INDEX + 1))/$TOTAL_BATCHES 批次 (索引: $BATCH_INDEX)"
        echo "使用端口: $MASTER_PORT"
        echo "=========================================="
        
        torchrun \
            --nproc_per_node=$NPROC_PER_NODE \
            --master_port=$MASTER_PORT \
            train_distributed_MovieLens.py \
            --config "$CONFIG_FILE" \
            --deepspeed "$DEEPSPEED_CONFIG" \
            --ablation_config "$ABLATION_CONFIG" \
            --history_strategy "$HISTORY_STRATEGY" \
            --history_ratio $HISTORY_RATIO \
            --output_dir "$OUTPUT_DIR" \
            --max_epochs $MAX_EPOCHS \
            --early_stopping_patience $EARLY_STOPPING_PATIENCE \
            --early_stopping_threshold $EARLY_STOPPING_THRESHOLD \
            --val_ratio $VAL_RATIO \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "${WANDB_RUN_NAME}_batch${BATCH_INDEX}" \
            --prompt_style "$PROMPT_STYLE" \
            --load_from_saved \
            --data_batch_size $BATCH_SIZE \
            --data_batch_index $BATCH_INDEX
        
        if [ $? -ne 0 ]; then
            echo "错误: 第 $((BATCH_INDEX + 1)) 批次训练失败"
            echo "已完成的批次: 0 到 $BATCH_INDEX"
            exit 1
        fi
        
        echo "✓ 第 $((BATCH_INDEX + 1))/$TOTAL_BATCHES 批次训练完成"
    done
else
    echo "===== 只有1个批次，无需训练剩余批次 ====="
fi

# ===== 完成 =====
echo ""
echo "=========================================="
echo "===== 所有批次训练完成 ====="
echo "=========================================="
echo "总批次数: $TOTAL_BATCHES"
echo "输出目录:"
for i in $(seq 0 $((TOTAL_BATCHES - 1))); do
    echo "  - ${OUTPUT_DIR}_batch${i}"
done
echo ""
echo "数据文件:"
echo "  - all_samples.json (所有处理后的数据)"
echo "  - train_samples_batch*.json (各批次的训练集)"
echo "  - val_samples_batch*.json (各批次的验证集)"
echo "=========================================="
