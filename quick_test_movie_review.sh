#!/bin/bash

# 快速测试影评训练脚本（只训练几步，验证配置正确）

echo "================================"
echo "快速测试 - 豆瓣影评模型"
echo "================================"

# 测试1: 数据加载
echo ""
echo "测试1: 数据加载和配置..."
python test_movie_review_setup.py

if [ $? -ne 0 ]; then
    echo "❌ 数据加载测试失败"
    exit 1
fi

# 测试2: 单卡快速训练（只训练1个epoch，验证流程）
echo ""
echo "================================"
echo "测试2: 单卡快速训练（1个epoch）"
echo "================================"

python train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config baseline \
    --output_dir outputs/quick_test_single_gpu \
    --max_epochs 1 \
    --prompt_style simple

if [ $? -ne 0 ]; then
    echo "❌ 单卡训练测试失败"
    exit 1
fi

# 测试3: 多卡快速训练（2卡，只训练1个epoch）
echo ""
echo "================================"
echo "测试3: 多卡快速训练（2卡，1个epoch）"
echo "================================"

torchrun \
    --nproc_per_node=2 \
    --master_port=29511 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config baseline \
    --output_dir outputs/quick_test_multi_gpu \
    --max_epochs 1 \
    --prompt_style simple

if [ $? -ne 0 ]; then
    echo "❌ 多卡训练测试失败"
    exit 1
fi

echo ""
echo "================================"
echo "✅ 所有测试通过！"
echo "================================"
echo ""
echo "现在可以开始正式训练："
echo "  bash run_train_movie_review.sh"
echo ""
echo "或者运行完整的消融实验。"
