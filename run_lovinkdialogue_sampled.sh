#!/bin/bash
# LovinkDialogue 训练 - 不扩充数据 + 采样模式

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理缓存
echo "清理 Python 缓存..."
cd /mnt/parallel/CompactSubset_experiement
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "================================"
echo "LovinkDialogue 训练"
echo "================================"
echo ""
echo "数据模式："
echo "  ✅ 不进行数据扩充（使用 data_loader.py）"
echo "  ✅ 每个 data_item → 1 个训练样本"
echo "  ✅ 每用户采样 2 个样本"
echo ""
echo "优势："
echo "  - 训练数据量减少"
echo "  - 训练时间缩短"
echo "  - 避免过拟合"
echo ""
echo "================================"

torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

echo ""
echo "================================"
echo "训练完成！"
echo "================================"
echo ""
echo "查看结果："
echo "  日志: outputs/LovinkDialogue_profile_context_sampled_seed42/training_logs/"
echo "  样本预览: outputs/LovinkDialogue_profile_context_sampled_seed42/training_samples_preview.txt"
echo "  WandB: https://wandb.ai/your-username/Qwen3_30B-LovinkDialogue"
