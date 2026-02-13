#!/usr/bin/env python3
"""
测试 data_loader.py 是否只预测 continuation
"""
import sys
sys.path.insert(0, '/mnt/parallel/CompactSubset_experiement')

from data_loader import load_train_data, extract_training_samples

# 加载数据
train_path = "/mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json"
train_data = load_train_data(train_path)

print(f"原始数据项数: {len(train_data)}")

# 提取样本
samples = extract_training_samples(train_data, debug=True)

print(f"\n" + "="*50)
print(f"提取的样本数: {len(samples)}")
print(f"扩充倍数: {len(samples) / len(train_data):.2f}x")
print(f"\n预期结果（只预测 continuation）:")
print(f"  - 扩充倍数应该 ≈ 1.0x")
print(f"  - 样本数应该 ≈ {len(train_data)}")
print(f"\n实际结果:")
if len(samples) / len(train_data) < 1.5:
    print(f"  ✅ 正确！使用的是 data_loader.py（只预测 continuation）")
else:
    print(f"  ❌ 错误！使用的是 data_loader_more_data.py（数据扩充版本）")
print("="*50)

# 显示第一个样本
if samples:
    print(f"\n第一个样本示例:")
    sample = samples[0]
    print(f"  user_hash: {sample.get('user_hash', 'N/A')[:20]}...")
    print(f"  context 轮数: {len(sample.get('context', []))}")
    print(f"  next_question 长度: {len(sample.get('next_question', ''))}")
