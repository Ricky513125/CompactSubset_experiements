#!/bin/bash
# 测试 DeepSpeed Ulysses 序列并行 - DMSC Movie Review

# 检查 DeepSpeed 版本
echo "检查 DeepSpeed 版本..."
python3 -c "import deepspeed; print(f'DeepSpeed 版本: {deepspeed.__version__}')"

# 检查是否支持序列并行
python3 << 'EOF'
import deepspeed
version = deepspeed.__version__
major, minor = map(int, version.split('.')[:2])

if major > 0 or (major == 0 and minor >= 10):
    print(f"✅ DeepSpeed {version} 支持序列并行（需要 >= 0.10.0）")
else:
    print(f"⚠️ DeepSpeed {version} 可能不支持序列并行")
    print("建议升级: pip install --upgrade deepspeed")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "是否升级 DeepSpeed? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        pip install --upgrade deepspeed
    else
        echo "跳过升级，继续测试..."
    fi
fi

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理缓存
echo ""
echo "清理 Python 缓存..."
cd /mnt/parallel/CompactSubset_experiement
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 测试 Ulysses 序列并行（16K 长度）
echo ""
echo "================================"
echo "测试 DeepSpeed Ulysses 序列并行"
echo "max_length=16384, 8 GPUs"
echo "每用户 2 个样本"
echo "================================"

torchrun \
    --nproc_per_node=8 \
    --master_port=29504 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_ulysses.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_ulysses_test \
    --max_epochs 1 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name ulysses_test \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

echo ""
echo "================================"
echo "Ulysses 测试完成！"
echo "================================"
echo ""
echo "如果成功："
echo "  → 可以使用序列并行处理超长序列"
echo "  → 每个 GPU 处理序列的 1/8"
echo ""
echo "如果失败（可能原因）："
echo "  1. DeepSpeed 版本不支持（需要 >= 0.10.0）"
echo "  2. 模型不支持序列并行（需要 FlashAttention 2）"
echo "  3. 配置参数不兼容"
echo ""
echo "日志位置: outputs/DMSC_ulysses_test/training_logs/"
