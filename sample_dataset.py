#!/usr/bin/env python3
"""
数据集采样脚本
对每个用户随机保留最多 N 个样本，生成新的数据集文件

用法:
    python sample_dataset.py <input_json> <output_json> --max_samples <N> --seed <seed>
    
示例:
    python sample_dataset.py /path/to/train.json /path/to/train_3.json --max_samples 3 --seed 42
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def sample_dataset(input_path, output_path, max_samples_per_user=3, seed=42, user_id_field='user_hash'):
    """
    对数据集进行采样，每个用户最多保留 max_samples_per_user 个样本
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        max_samples_per_user: 每个用户最多保留的样本数
        seed: 随机种子
        user_id_field: 用户ID字段名（可以是 'user_hash', 'user_id', 'userId' 等）
    """
    random.seed(seed)
    
    print(f"=" * 80)
    print(f"数据集采样工具")
    print(f"=" * 80)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"每用户最多样本数: {max_samples_per_user}")
    print(f"随机种子: {seed}")
    print(f"用户ID字段: {user_id_field}")
    print()
    
    # 读取数据集
    print("📖 读取数据集...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    print(f"✅ 原始样本数: {original_count}")
    
    # 自动检测用户ID字段
    if data:
        first_sample = data[0]
        possible_fields = ['user_hash', 'user_id', 'userId', 'target_user_id', 'author']
        detected_field = None
        for field in possible_fields:
            if field in first_sample:
                detected_field = field
                break
        
        if detected_field and detected_field != user_id_field:
            print(f"⚠️  检测到用户ID字段: {detected_field} (覆盖默认值 {user_id_field})")
            user_id_field = detected_field
    
    # 按用户分组
    print(f"📊 按用户分组 (使用字段: {user_id_field})...")
    user_samples = defaultdict(list)
    
    for sample in data:
        user_id = sample.get(user_id_field)
        if user_id is None:
            print(f"⚠️  警告: 样本缺少 {user_id_field} 字段，跳过")
            continue
        user_samples[user_id].append(sample)
    
    num_users = len(user_samples)
    print(f"✅ 唯一用户数: {num_users}")
    print(f"✅ 平均每用户样本数: {original_count / num_users:.2f}")
    
    # 统计样本分布
    from collections import Counter
    sample_counts = [len(samples) for samples in user_samples.values()]
    count_dist = Counter(sample_counts)
    print(f"\n样本数分布 (前10个):")
    for count, num_users_with_count in sorted(count_dist.items())[:10]:
        print(f"  {count:3d} 个样本: {num_users_with_count:5d} 个用户")
    
    if len(count_dist) > 10:
        print(f"  ... (共 {len(count_dist)} 种不同的样本数)")
    
    # 采样
    print(f"\n🎲 开始采样 (每用户最多 {max_samples_per_user} 个样本)...")
    sampled_data = []
    affected_users = 0
    removed_samples = 0
    
    for user_id, samples in user_samples.items():
        original_sample_count = len(samples)
        
        if original_sample_count > max_samples_per_user:
            # 需要采样
            sampled = random.sample(samples, max_samples_per_user)
            affected_users += 1
            removed_samples += (original_sample_count - max_samples_per_user)
        else:
            # 保留全部
            sampled = samples
        
        sampled_data.extend(sampled)
    
    # 打乱顺序
    random.shuffle(sampled_data)
    
    new_count = len(sampled_data)
    print(f"✅ 采样完成")
    print(f"   - 新样本数: {new_count}")
    print(f"   - 移除样本数: {removed_samples} ({removed_samples/original_count*100:.2f}%)")
    print(f"   - 受影响用户数: {affected_users} ({affected_users/num_users*100:.2f}%)")
    print(f"   - 保留比例: {new_count/original_count*100:.2f}%")
    
    # 保存
    print(f"\n💾 保存到: {output_path}")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成！")
    print(f"=" * 80)
    
    return {
        'original_count': original_count,
        'new_count': new_count,
        'num_users': num_users,
        'affected_users': affected_users,
        'removed_samples': removed_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description='对数据集进行采样，每个用户最多保留N个样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Chameleons 数据集，每用户最多3个样本
  python sample_dataset.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train_3.json \\
      --max_samples 3 --seed 42
  
  # REALTALK 数据集，每用户最多5个样本
  python sample_dataset.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/REALTALK/train.json \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/REALTALK/train_5.json \\
      --max_samples 5 --seed 42
        """
    )
    
    parser.add_argument('input', type=str, help='输入JSON文件路径')
    parser.add_argument('output', type=str, help='输出JSON文件路径')
    parser.add_argument('--max_samples', type=int, default=3,
                        help='每个用户最多保留的样本数 (默认: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--user_id_field', type=str, default='user_hash',
                        help='用户ID字段名 (默认: user_hash，会自动检测)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return 1
    
    # 执行采样
    sample_dataset(
        input_path=args.input,
        output_path=args.output,
        max_samples_per_user=args.max_samples,
        seed=args.seed,
        user_id_field=args.user_id_field
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
