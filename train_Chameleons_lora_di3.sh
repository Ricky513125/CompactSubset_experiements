#!/bin/bash

# Chameleons LoRA 训练脚本
# 使用 LoRA 微调 30B 模型，训练速度提升 3-5 倍

echo "================================================================================"
echo "Chameleons LoRA 训练 (30B 模型)"
echo "================================================================================"
echo "优势:"
echo "  ⚡ 训练速度: 3-5x 加速 (28秒/batch → 5-8秒/batch)"
echo "  💾 显存占用: 减少 50-70%"
echo "  ⏱️  训练时间: 12.5天 → 2-3天"
echo "  📊 可训练参数: 30B → 50-200M (<1%)"
echo ""
echo "数据集: sampled_data/Chameleons/train_di3.json (~16,963 样本)"
echo "DeepSpeed: ZeRO-2 (LoRA 显存占用小，无需 ZeRO-3)"
echo "================================================================================"
echo ""

# 配置
CONFIG="config_Chameleons_30B_lora_di3.json"
DEEPSPEED="ds_config_zero2.json"
ABLATION="context_only"
OUTPUT_DIR="outputs/Chameleons_context_30B_lora_di3"
PROJECT="Qwen3_30B-Chameleons-LoRA"
RUN_NAME="context_lora_r64_di3"
MASTER_PORT=29503

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    echo "请先运行: python create_lora_config.py"
    exit 1
fi

if [ ! -f "$DEEPSPEED" ]; then
    echo "❌ 错误: DeepSpeed 配置文件不存在: $DEEPSPEED"
    exit 1
fi

if [ ! -f "sampled_data/Chameleons/train_di3.json" ]; then
    echo "❌ 错误: 采样数据集不存在"
    echo "请先运行:"
    echo "  python sample_dataset_data_item_level.py \\"
    echo "      /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \\"
    echo "      sampled_data/Chameleons/train_di3.json \\"
    echo "      --max_data_items 3 --seed 42"
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
echo ""
echo "LoRA 适配器已保存，可用于:"
echo "  1. 继续训练"
echo "  2. 合并到基础模型: python merge_lora_weights.py"
echo "  3. 推理: 加载 base model + LoRA adapter"
echo "================================================================================"
