#!/bin/bash
# DMSC 训练 - 每用户一个样本模式（大幅缩短训练时间）

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理缓存
echo "清理 Python 缓存..."
cd /mnt/parallel/CompactSubset_experiement
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "================================"
echo "DMSC 训练 - 每用户一个样本模式"
echo "================================"
echo ""
echo "数据模式："
echo "  - 每个用户只生成 1 个训练样本"
echo "  - 使用前 n-1 条影评作为历史"
echo "  - 预测第 n 条影评"
echo ""
echo "优势："
echo "  ✅ 训练数据量大幅减少（例如：10000条 → 100条）"
echo "  ✅ 训练时间大幅缩短（例如：4小时 → 5分钟）"
echo "  ✅ 每个样本包含该用户的完整历史"
echo "  ✅ 更适合长序列训练（16K）"
echo ""
echo "================================"

torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_one_per_user_0213 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name one_per_user_0213 \
    --prompt_style simple \
    --one_sample_per_user

echo ""
echo "================================"
echo "训练完成！"
echo "================================"
echo ""
echo "查看结果："
echo "  日志: outputs/DMSC_one_per_user_0213/training_logs/"
echo "  样本预览: outputs/DMSC_one_per_user_0213/training_samples_preview.txt"
echo "  WandB: https://wandb.ai/your-username/Qwen3_30B-DMSC"
