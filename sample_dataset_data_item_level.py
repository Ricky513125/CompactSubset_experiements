#!/usr/bin/env python3
"""
Chameleons 数据集 data_item 级别采样脚本
在 data_item 级别（训练样本级别）进行采样，而不是在 user_hash 级别

用法:
    python sample_dataset_data_item_level.py <input_json> <output_json> --max_data_items <N> --seed <seed>
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def sample_at_data_item_level(input_path, output_path, max_data_items_per_user=10, seed=42):
    """
    在 data_item 级别对 Chameleons 数据集进行采样
    每个用户最多保留 max_data_items_per_user 个 data_item (训练样本)
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        max_data_items_per_user: 每个用户最多保留的 data_item 数量
        seed: 随机种子
    """
    random.seed(seed)
    
    print(f"=" * 80)
    print(f"Chameleons 数据集 data_item 级别采样工具")
    print(f"=" * 80)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"每用户最多 data_item 数: {max_data_items_per_user}")
    print(f"随机种子: {seed}")
    print()
    
    # 读取数据集
    print("📖 读取数据集...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_user_count = len(data)
    print(f"✅ 原始用户数: {original_user_count}")
    
    # 统计原始 data_item 数量
    original_data_item_count = 0
    for item in data:
        collections = item.get('task', {}).get('task_behavior_collections', [])
        for coll in collections:
            original_data_item_count += len(coll.get('data', []))
    
    print(f"✅ 原始 data_item 数: {original_data_item_count}")
    print(f"✅ 平均每用户: {original_data_item_count / original_user_count:.1f} 个 data_item")
    
    # 对每个用户的 data_item 进行采样
    print(f"\n🎲 开始采样 (每用户最多 {max_data_items_per_user} 个 data_item)...")
    
    sampled_data = []
    affected_users = 0
    removed_data_items = 0
    new_data_item_count = 0
    
    for item in data:
        user_hash = item.get('user_hash')
        collections = item.get('task', {}).get('task_behavior_collections', [])
        
        # 收集所有 data_item
        all_data_items = []
        for coll in collections:
            data_items = coll.get('data', [])
            all_data_items.extend(data_items)
        
        original_count = len(all_data_items)
        
        # 采样
        if original_count > max_data_items_per_user:
            sampled_data_items = random.sample(all_data_items, max_data_items_per_user)
            affected_users += 1
            removed_data_items += (original_count - max_data_items_per_user)
        else:
            sampled_data_items = all_data_items
        
        new_data_item_count += len(sampled_data_items)
        
        # 重构数据结构
        new_item = {
            'user_hash': item.get('user_hash'),
            'user': item.get('user'),
            'task': {
                'description': item.get('task', {}).get('description', ''),
                'task_behavior_collections': [
                    {
                        'data': sampled_data_items
                    }
                ]
            }
        }
        
        sampled_data.append(new_item)
    
    print(f"✅ 采样完成")
    print(f"   - 用户数: {len(sampled_data)} (不变)")
    print(f"   - 原始 data_item 数: {original_data_item_count}")
    print(f"   - 新 data_item 数: {new_data_item_count}")
    print(f"   - 移除 data_item 数: {removed_data_items} ({removed_data_items/original_data_item_count*100:.2f}%)")
    print(f"   - 受影响用户数: {affected_users} ({affected_users/original_user_count*100:.2f}%)")
    print(f"   - 保留比例: {new_data_item_count/original_data_item_count*100:.2f}%")
    print(f"   - 平均每用户 data_item 数: {new_data_item_count / len(sampled_data):.1f}")
    
    # 保存
    print(f"\n💾 保存到: {output_path}")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成！")
    print(f"=" * 80)
    
    return {
        'original_user_count': original_user_count,
        'original_data_item_count': original_data_item_count,
        'new_data_item_count': new_data_item_count,
        'affected_users': affected_users,
        'removed_data_items': removed_data_items,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Chameleons 数据集 data_item 级别采样',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 每用户最多10个 data_item
  python sample_dataset_data_item_level.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \\
      sampled_data/Chameleons/train_di10.json \\
      --max_data_items 10 --seed 42
  
  # 每用户最多20个 data_item
  python sample_dataset_data_item_level.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \\
      sampled_data/Chameleons/train_di20.json \\
      --max_data_items 20 --seed 42
        """
    )
    
    parser.add_argument('input', type=str, help='输入JSON文件路径')
    parser.add_argument('output', type=str, help='输出JSON文件路径')
    parser.add_argument('--max_data_items', type=int, default=10,
                        help='每个用户最多保留的 data_item 数 (默认: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return 1
    
    # 执行采样
    sample_at_data_item_level(
        input_path=args.input,
        output_path=args.output,
        max_data_items_per_user=args.max_data_items,
        seed=args.seed
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
