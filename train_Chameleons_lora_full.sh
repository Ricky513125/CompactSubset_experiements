#!/bin/bash

# Chameleons LoRA 训练脚本 - 完整数据集版本
# 使用完整的 72,471 个训练样本进行 LoRA 微调

echo "================================================================================"
echo "Chameleons LoRA 训练 - 完整数据集"
echo "================================================================================"
echo "配置:"
echo "  📊 训练样本: 72,471 (完整数据集)"
echo "  🎯 方法: LoRA (rank=64)"
echo "  💾 DeepSpeed: ZeRO-2"
echo "  ⚡ 预期速度: 5-8秒/batch (vs 28秒全参数)"
echo "  ⏰ 预计时间: 8-10 天 (vs 12.5 天全参数)"
echo "================================================================================"
echo ""

# 配置
CONFIG="config_Chameleons_30B_lora_full.json"
DEEPSPEED="ds_config_zero2.json"
ABLATION="context_only"
OUTPUT_DIR="outputs/Chameleons_context_30B_lora_full"
PROJECT="Qwen3_30B-Chameleons-LoRA"
RUN_NAME="context_lora_r64_full"
MASTER_PORT=29502

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$DEEPSPEED" ]; then
    echo "❌ 错误: DeepSpeed 配置文件不存在: $DEEPSPEED"
    exit 1
fi

echo "✅ 配置检查通过"
echo ""
echo "开始训练..."
echo ""

# 训练命令
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
echo "================================================================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================================================================"
