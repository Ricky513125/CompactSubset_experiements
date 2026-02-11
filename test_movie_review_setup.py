"""
测试影评训练脚本的数据加载和初始化
用于快速验证配置是否正确，无需真正训练
"""
import json
import os
from pathlib import Path

# 测试数据加载
print("=" * 80)
print("测试1: 数据加载器")
print("=" * 80)

from data_loader_movie_review import (
    load_movie_review_data,
    extract_movie_review_samples,
    split_movie_reviews_by_time
)

# 加载示例数据
data_file = "example_movie_review_data.json"
print(f"加载数据文件: {data_file}")
raw_data = load_movie_review_data(data_file)
print(f"✓ 加载成功，用户数: {len(raw_data)}")

# 提取样本
all_samples = extract_movie_review_samples(raw_data, debug=True)
print(f"✓ 提取样本成功，总样本数: {len(all_samples)}")

# 划分数据集
train_samples, val_samples, test_samples = split_movie_reviews_by_time(
    all_samples,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    debug=True
)
print(f"✓ 数据划分成功")

# 测试配置文件
print("\n" + "=" * 80)
print("测试2: 配置文件")
print("=" * 80)

config_file = "config_MovieReview.json"
print(f"加载配置文件: {config_file}")
with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

print(f"✓ 配置加载成功")
print(f"  模型路径: {config['model']['path']}")
print(f"  训练数据: {config['data']['train_path']}")
print(f"  消融配置: {list(config['ablation_configs'].keys())}")

# 测试每个消融配置
for ablation_name, ablation_config in config['ablation_configs'].items():
    print(f"  - {ablation_name}: profile={ablation_config['use_profile']}, history={ablation_config['use_history']}")

# 测试tokenizer加载
print("\n" + "=" * 80)
print("测试3: Tokenizer和数据集")
print("=" * 80)

try:
    from transformers import AutoTokenizer
    from train_distributed_MovieReview import MovieReviewDataset
    
    model_path = config['model']['path']
    if os.path.exists(model_path):
        print(f"加载tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer加载成功")
        
        # 创建数据集
        print("创建数据集...")
        dataset = MovieReviewDataset(
            samples=train_samples[:5],  # 只用5个样本测试
            tokenizer=tokenizer,
            max_length=4096,
            use_profile=True,
            use_history=True,
            use_context=False,
            verbose=True
        )
        print(f"✓ 数据集创建成功，样本数: {len(dataset)}")
        
        # 测试第一个样本
        print("\n测试编码第一个样本...")
        sample = dataset[0]
        print(f"  Input IDs: {sample['input_ids'].shape}")
        print(f"  Labels: {sample['labels'].shape}")
        print(f"  有效标签数: {(sample['labels'] != -100).sum().item()}")
        print(f"✓ 样本编码成功")
    else:
        print(f"⚠️  模型路径不存在: {model_path}")
        print(f"   跳过tokenizer和数据集测试")
except Exception as e:
    print(f"⚠️  Tokenizer/数据集测试失败: {e}")

# 测试CUDA
print("\n" + "=" * 80)
print("测试4: CUDA环境")
print("=" * 80)

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            compute_cap = torch.cuda.get_device_capability(i)
            print(f"    计算能力: {compute_cap[0]}.{compute_cap[1]}")
    print("✓ CUDA检查完成")
except Exception as e:
    print(f"⚠️  CUDA检查失败: {e}")

# 测试FlashAttention
print("\n" + "=" * 80)
print("测试5: FlashAttention支持")
print("=" * 80)

try:
    import flash_attn
    flash_version = getattr(flash_attn, '__version__', 'unknown')
    print(f"✓ FlashAttention已安装，版本: {flash_version}")
except ImportError:
    print("⚠️  FlashAttention未安装")
    print("   可以使用 pip install flash-attn --no-build-isolation 安装")

print("\n" + "=" * 80)
print("✅ 所有测试完成！")
print("=" * 80)
print("\n如果以上测试都通过，可以开始训练：")
print("  bash run_train_movie_review.sh")
print("或者：")
print("  torchrun --nproc_per_node=8 train_distributed_MovieReview.py \\")
print("    --config config_MovieReview.json \\")
print("    --ablation_config profile_and_history \\")
print("    --output_dir outputs/test_run")
