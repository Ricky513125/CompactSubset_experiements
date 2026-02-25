#!/usr/bin/env python3
"""
DMSC 数据集采样脚本
对每个用户按时间戳排序，保留后 X% 的评论

用法:
    python sample_dataset_DMSC.py <input_json> <output_json> --keep_ratio <0.0-1.0> [--user_id_field <field>]
    
示例:
    # 保留每个用户的后50%评论
    python sample_dataset_DMSC.py /path/to/train.json /path/to/train_50pct.json --keep_ratio 0.5
    
    # 保留每个用户的后30%评论
    python sample_dataset_DMSC.py /path/to/train.json /path/to/train_30pct.json --keep_ratio 0.3
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    解析时间戳字符串
    
    支持格式:
    - "2012-12-24"
    - "2012-12-24 10:30:00"
    - "2012-12-24T10:30:00"
    """
    timestamp_str = timestamp_str.strip()
    
    # 尝试不同的日期格式
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    # 如果都失败，返回一个很旧的日期（这样会排在前面）
    print(f"⚠️  警告: 无法解析时间戳 '{timestamp_str}'，使用默认值 1970-01-01")
    return datetime(1970, 1, 1)


def get_user_id(user_data: Dict[str, Any], user_id_field: str = 'name') -> Optional[str]:
    """
    从用户数据中提取用户ID
    
    Args:
        user_data: 用户数据对象（包含 user 和 task 字段）
        user_id_field: 用户ID字段名（默认从 user.profile.name 获取）
    
    Returns:
        用户ID字符串，如果找不到则返回 None
    """
    if user_id_field == 'name':
        # 默认从 user.profile.name 获取
        profile = user_data.get('user', {}).get('profile', {})
        return profile.get('name')
    else:
        # 尝试从不同位置获取
        if user_id_field in user_data:
            return user_data[user_id_field]
        if 'user' in user_data and user_id_field in user_data['user']:
            return user_data['user'][user_id_field]
        if 'user' in user_data and 'profile' in user_data['user']:
            return user_data['user']['profile'].get(user_id_field)
    return None


def collect_all_reviews(user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    收集用户的所有评论（跨所有 collection）
    
    Args:
        user_data: 用户数据对象
    
    Returns:
        所有评论的列表，每个评论包含原始数据和元数据
    """
    all_reviews = []
    
    task = user_data.get('task', {})
    collections = task.get('task_behavior_collections', [])
    
    for collection_idx, collection in enumerate(collections):
        data_items = collection.get('data', [])
        
        for data_idx, data_item in enumerate(data_items):
            # 为每个评论添加元数据，以便后续重新分配
            review_with_meta = {
                'data_item': data_item,
                'collection_idx': collection_idx,
                'data_idx': data_idx,
            }
            all_reviews.append(review_with_meta)
    
    return all_reviews


def sample_dmsc_dataset(
    input_path: str,
    output_path: str,
    keep_ratio: float = 0.5,
    user_id_field: str = 'name'
):
    """
    对DMSC数据集进行采样，每个用户保留后 keep_ratio 比例的评论
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        keep_ratio: 保留比例（0.0-1.0），例如 0.5 表示保留后50%
        user_id_field: 用户ID字段名（默认从 user.profile.name 获取）
    """
    print("=" * 80)
    print("DMSC 数据集采样工具（按时间戳保留后X%）")
    print("=" * 80)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"保留比例: {keep_ratio * 100:.1f}% (后{keep_ratio * 100:.1f}%)")
    print(f"用户ID字段: {user_id_field}")
    print()
    
    if not (0.0 < keep_ratio <= 1.0):
        print(f"❌ 错误: keep_ratio 必须在 (0.0, 1.0] 范围内，当前值: {keep_ratio}")
        return None
    
    # 读取数据集
    print("📖 读取数据集...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_user_count = len(data)
    print(f"✅ 原始用户数: {original_user_count}")
    
    # 统计原始数据
    total_reviews_before = 0
    user_review_counts = []
    
    for user_data in data:
        reviews = collect_all_reviews(user_data)
        review_count = len(reviews)
        total_reviews_before += review_count
        user_review_counts.append(review_count)
    
    print(f"✅ 原始评论总数: {total_reviews_before}")
    print(f"✅ 平均每用户评论数: {total_reviews_before / original_user_count:.2f}")
    
    # 统计评论数分布
    from collections import Counter
    count_dist = Counter(user_review_counts)
    print(f"\n评论数分布 (前10个):")
    for count, num_users in sorted(count_dist.items(), reverse=True)[:10]:
        print(f"  {count:3d} 条评论: {num_users:5d} 个用户")
    
    # 采样
    print(f"\n🎲 开始采样 (保留每个用户的后 {keep_ratio * 100:.1f}% 评论)...")
    sampled_data = []
    total_reviews_after = 0
    affected_users = 0
    
    for user_idx, user_data in enumerate(data):
        user_id = get_user_id(user_data, user_id_field)
        if user_id is None:
            user_id = f"user_{user_idx}"
        
        # 收集所有评论
        all_reviews = collect_all_reviews(user_data)
        original_review_count = len(all_reviews)
        
        if original_review_count == 0:
            # 没有评论，跳过
            continue
        
        # 按时间戳排序
        def get_timestamp(review):
            data_item = review['data_item']
            timestamp_str = data_item.get('timestamp', '1970-01-01')
            return parse_timestamp(timestamp_str)
        
        sorted_reviews = sorted(all_reviews, key=get_timestamp)
        
        # 计算保留数量
        keep_count = max(1, int(original_review_count * keep_ratio))  # 至少保留1条
        kept_reviews = sorted_reviews[-keep_count:]  # 保留后N条
        
        if keep_count < original_review_count:
            affected_users += 1
        
        total_reviews_after += len(kept_reviews)
        
        # 重新构建用户数据结构
        # 需要将保留的评论重新分配到原来的 collection 结构中
        new_user_data = {
            'user': user_data.get('user', {}).copy(),
            'task': {
                'description': user_data.get('task', {}).get('description', ''),
                'task_behavior_collections': []
            }
        }
        
        # 按 collection 分组保留的评论
        kept_by_collection = defaultdict(list)
        for review in kept_reviews:
            collection_idx = review['collection_idx']
            kept_by_collection[collection_idx].append(review)
        
        # 重建 collections
        original_collections = user_data.get('task', {}).get('task_behavior_collections', [])
        for collection_idx, collection in enumerate(original_collections):
            if collection_idx in kept_by_collection:
                # 这个 collection 有保留的评论
                kept_reviews_for_collection = kept_by_collection[collection_idx]
                # 按原始顺序排序（保持 data_idx）
                kept_reviews_for_collection.sort(key=lambda r: r['data_idx'])
                
                new_collection = collection.copy()
                new_collection['data'] = [r['data_item'] for r in kept_reviews_for_collection]
                new_user_data['task']['task_behavior_collections'].append(new_collection)
            # 如果没有保留的评论，跳过这个 collection
        
        sampled_data.append(new_user_data)
    
    # 统计
    removed_reviews = total_reviews_before - total_reviews_after
    print(f"✅ 采样完成")
    print(f"   - 保留用户数: {len(sampled_data)}")
    print(f"   - 保留评论数: {total_reviews_after}")
    print(f"   - 移除评论数: {removed_reviews} ({removed_reviews/total_reviews_before*100:.2f}%)")
    print(f"   - 受影响用户数: {affected_users} ({affected_users/original_user_count*100:.2f}%)")
    print(f"   - 保留比例: {total_reviews_after/total_reviews_before*100:.2f}%")
    
    # 保存
    print(f"\n💾 保存到: {output_path}")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成！")
    print("=" * 80)
    
    return {
        'original_user_count': original_user_count,
        'sampled_user_count': len(sampled_data),
        'original_review_count': total_reviews_before,
        'sampled_review_count': total_reviews_after,
        'removed_reviews': removed_reviews,
        'affected_users': affected_users,
        'keep_ratio': keep_ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description='对DMSC数据集进行采样，每个用户保留后X%的评论（按时间戳排序）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 保留每个用户的后50%评论
  python sample_dataset_DMSC.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train.json \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train_50pct.json \\
      --keep_ratio 0.5
  
  # 保留每个用户的后30%评论
  python sample_dataset_DMSC.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train.json \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train_30pct.json \\
      --keep_ratio 0.3
  
  # 保留每个用户的后80%评论
  python sample_dataset_DMSC.py \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train.json \\
      /mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC/train_80pct.json \\
      --keep_ratio 0.8
        """
    )
    
    parser.add_argument('input', type=str, help='输入JSON文件路径')
    parser.add_argument('output', type=str, help='输出JSON文件路径')
    parser.add_argument('--keep_ratio', type=float, default=0.5,
                        help='保留比例 (0.0-1.0)，例如 0.5 表示保留后50%% (默认: 0.5)')
    parser.add_argument('--user_id_field', type=str, default='name',
                        help='用户ID字段名 (默认: name，从 user.profile.name 获取)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return 1
    
    # 检查 keep_ratio 范围
    if not (0.0 < args.keep_ratio <= 1.0):
        print(f"❌ 错误: --keep_ratio 必须在 (0.0, 1.0] 范围内，当前值: {args.keep_ratio}")
        return 1
    
    # 执行采样
    result = sample_dmsc_dataset(
        input_path=args.input,
        output_path=args.output,
        keep_ratio=args.keep_ratio,
        user_id_field=args.user_id_field
    )
    
    if result is None:
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
