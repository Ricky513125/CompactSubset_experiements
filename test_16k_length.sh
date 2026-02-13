#!/bin/bash
# 测试 16K 长度训练 - DMSC Movie Review

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理缓存
echo "清理 Python 缓存..."
cd /mnt/parallel/CompactSubset_experiement
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 测试 16K 长度（少量样本）
echo "================================"
echo "测试 max_length=16384"
echo "每用户 2 个样本"
echo "================================"

torchrun \
    --nproc_per_node=8 \
    --master_port=29503 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_16k_test \
    --max_epochs 1 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name 16k_length_test \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

echo ""
echo "================================"
echo "测试完成！"
echo "================================"
echo ""
echo "检查结果："
echo "1. 如果成功 → 可以使用 16K 长度"
echo "2. 如果 OOM → 需要："
echo "   - 减少 max_length 到 8192 或 4096"
echo "   - 或使用 DeepSpeed Ulysses 序列并行"
echo ""
echo "日志位置: outputs/DMSC_16k_test/training_logs/"
