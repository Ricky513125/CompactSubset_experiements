#!/bin/bash
# 快速测试：每用户一个样本模式

cd /mnt/parallel/CompactSubset_experiement

echo "========================================"
echo "测试：每用户一个样本模式"
echo "========================================"
echo ""

python3 << 'EOF'
import json
from data_loader_movie_review import load_movie_review_data, extract_movie_review_samples

# 加载数据
data_file = "/mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train.json"
print(f"加载数据: {data_file}")
raw_data = load_movie_review_data(data_file)
print(f"用户数: {len(raw_data)}\n")

# 模式 1: 原始模式（每条影评一个样本）
print("=" * 80)
print("模式 1: 原始模式（默认）")
print("=" * 80)
samples_original = extract_movie_review_samples(raw_data, one_sample_per_user=False, debug=False)
print(f"总样本数: {len(samples_original)}")

# 统计每个用户的样本数
user_samples = {}
for s in samples_original:
    user = s['user_hash']
    user_samples[user] = user_samples.get(user, 0) + 1

print(f"用户数: {len(user_samples)}")
print(f"平均每用户样本数: {len(samples_original)/len(user_samples):.1f}")

# 显示前3个样本的历史长度
print("\n前3个样本:")
for i in range(min(3, len(samples_original))):
    s = samples_original[i]
    print(f"  样本 {i+1}: {len(s['history'])}条历史 → {s['movie_name']}")

# 模式 2: 每用户一个样本
print("\n" + "=" * 80)
print("模式 2: 每用户一个样本（新）")
print("=" * 80)
samples_one_per_user = extract_movie_review_samples(raw_data, one_sample_per_user=True, debug=False)
print(f"总样本数: {len(samples_one_per_user)}")

# 统计每个用户的样本数
user_samples_new = {}
for s in samples_one_per_user:
    user = s['user_hash']
    user_samples_new[user] = user_samples_new.get(user, 0) + 1

print(f"用户数: {len(user_samples_new)}")
print(f"每用户样本数: {list(user_samples_new.values())[0] if user_samples_new else 0}")

# 显示前3个样本
print("\n前3个样本:")
for i in range(min(3, len(samples_one_per_user))):
    s = samples_one_per_user[i]
    total = s.get('total_reviews', 0)
    hist_count = len(s['history'])
    print(f"  样本 {i+1}: {hist_count}条历史 → {s['movie_name']} (用户总影评: {total})")
    print(f"           历史覆盖: {hist_count}/{total} = {hist_count/total*100:.1f}%")

# 对比
print("\n" + "=" * 80)
print("对比总结")
print("=" * 80)
print(f"样本数减少: {len(samples_original)} → {len(samples_one_per_user)} (减少 {len(samples_original)/len(samples_one_per_user):.1f}x)")
print(f"预估训练时间缩短: ~{len(samples_original)/len(samples_one_per_user):.0f}x")

# 检查历史长度分布
print("\n历史长度分布（每用户一个样本模式）:")
history_lengths = [len(s['history']) for s in samples_one_per_user]
if history_lengths:
    import numpy as np
    print(f"  最小: {min(history_lengths)}")
    print(f"  最大: {max(history_lengths)}")
    print(f"  平均: {np.mean(history_lengths):.1f}")
    print(f"  中位数: {np.median(history_lengths):.0f}")
    
    # 预估序列长度（假设每条影评平均80 tokens）
    avg_tokens = np.mean(history_lengths) * 80
    max_tokens = max(history_lengths) * 80
    print(f"\n预估序列长度:")
    print(f"  平均: ~{avg_tokens:.0f} tokens")
    print(f"  最长: ~{max_tokens:.0f} tokens")
    
    if max_tokens > 16384:
        print(f"\n⚠️ 最长样本可能超过 16K，建议:")
        print(f"   1. 使用 Ulysses 序列并行")
        print(f"   2. 或在 prompt 构建时截断历史")
    elif max_tokens > 8192:
        print(f"\n✅ 可以使用 16K max_length + CPU checkpointing")
    else:
        print(f"\n✅ 可以使用 8K max_length，无需额外优化")

print("\n" + "=" * 80)
print("✅ 测试完成！")
print("=" * 80)
EOF

echo ""
echo "下一步："
echo "  ./run_dmsc_one_per_user.sh"
