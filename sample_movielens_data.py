#!/usr/bin/env python3
"""
MovieLens 数据集采样脚本

功能：
1. 读取原始数据集文件
2. 对每个用户随机保留50%的数据
3. 将处理后的数据保存回原文件
4. 将原文件重命名为 _origin
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse


def sample_user_data(user_data: Dict[str, Any], sample_ratio: float = 0.5, seed: int = 42) -> Dict[str, Any]:
    """
    对单个用户的数据进行采样
    
    Args:
        user_data: 用户数据字典
        sample_ratio: 采样比例（0.0-1.0）
        seed: 随机种子
    
    Returns:
        采样后的用户数据
    """
    # 创建用户数据的副本
    sampled_user_data = json.loads(json.dumps(user_data))  # 深拷贝
    
    # 获取任务集合
    task = sampled_user_data.get('task', {})
    task_collections = task.get('task_behavior_collections', [])
    
    # 对每个集合进行采样
    for collection in task_collections:
        collection_type = collection.get('type', '')
        if collection_type not in ['movie_rating', 'movie_review']:
            continue
        
        ratings = collection.get('data', [])
        if not ratings:
            continue
        
        # 随机采样
        num_to_keep = max(1, int(len(ratings) * sample_ratio))
        sampled_ratings = random.sample(ratings, num_to_keep)
        
        # 保持时间顺序（按timestamp排序）
        sampled_ratings.sort(key=lambda x: x.get('timestamp', ''))
        
        # 更新集合中的数据
        collection['data'] = sampled_ratings
    
    return sampled_user_data


def sample_movielens_dataset(
    input_file: str,
    output_file: str = None,
    sample_ratio: float = 0.5,
    seed: int = 42,
    backup_original: bool = True
) -> None:
    """
    对MovieLens数据集进行采样
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为None，则覆盖原文件）
        sample_ratio: 采样比例（0.0-1.0）
        seed: 随机种子
        backup_original: 是否备份原文件（重命名为_origin）
    """
    # 设置随机种子
    random.seed(seed)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    # 读取原始数据
    print(f"读取数据文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保是列表格式
    if isinstance(data, dict):
        data = [data]
    
    total_users = len(data)
    print(f"总用户数: {total_users}")
    
    # 统计原始数据量
    total_ratings_before = 0
    for user_data in data:
        task = user_data.get('task', {})
        task_collections = task.get('task_behavior_collections', [])
        for collection in task_collections:
            if collection.get('type', '') in ['movie_rating', 'movie_review']:
                total_ratings_before += len(collection.get('data', []))
    
    print(f"原始总评分数: {total_ratings_before}")
    
    # 对每个用户进行采样
    print(f"\n开始采样（保留比例: {sample_ratio*100:.1f}%，随机种子: {seed}）...")
    sampled_data = []
    total_ratings_after = 0
    
    for idx, user_data in enumerate(data):
        user_name = user_data.get('user', {}).get('profile', {}).get('name', f'user_{idx}')
        
        # 采样用户数据
        sampled_user = sample_user_data(user_data, sample_ratio, seed)
        sampled_data.append(sampled_user)
        
        # 统计采样后的评分数
        task = sampled_user.get('task', {})
        task_collections = task.get('task_behavior_collections', [])
        for collection in task_collections:
            if collection.get('type', '') in ['movie_rating', 'movie_review']:
                total_ratings_after += len(collection.get('data', []))
        
        # 每处理1000个用户打印一次进度
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{total_users} 个用户 ({(idx + 1) / total_users * 100:.1f}%)")
    
    print(f"\n采样完成:")
    print(f"  原始评分数: {total_ratings_before}")
    print(f"  采样后评分数: {total_ratings_after}")
    print(f"  保留比例: {total_ratings_after / total_ratings_before * 100:.2f}%")
    
    # 备份原文件
    if backup_original:
        backup_file = input_file.replace('.json', '_origin.json')
        if os.path.exists(backup_file):
            print(f"\n警告: 备份文件已存在: {backup_file}")
            response = input("是否覆盖? (y/n): ")
            if response.lower() != 'y':
                print("取消操作")
                sys.exit(0)
        
        print(f"\n备份原文件: {input_file} -> {backup_file}")
        os.rename(input_file, backup_file)
        print(f"✓ 备份完成")
    
    # 确定输出文件路径
    if output_file is None:
        output_file = input_file
    
    # 保存采样后的数据
    print(f"\n保存采样后的数据: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    # 获取文件大小
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"✓ 保存完成 (文件大小: {file_size:.2f} MB)")
    print(f"\n✓ 所有操作完成！")


def main():
    parser = argparse.ArgumentParser(description='MovieLens数据集采样脚本')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件路径（如果为None，则覆盖原文件）')
    parser.add_argument('--sample_ratio', type=float, default=0.5,
                       help='采样比例（0.0-1.0，默认0.5即50%）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--no_backup', action='store_true',
                       help='不备份原文件（默认会备份）')
    
    args = parser.parse_args()
    
    # 验证采样比例
    if not 0.0 < args.sample_ratio <= 1.0:
        print(f"错误: 采样比例必须在 (0.0, 1.0] 范围内，当前值: {args.sample_ratio}")
        sys.exit(1)
    
    # 执行采样
    sample_movielens_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        backup_original=not args.no_backup
    )


if __name__ == '__main__':
    main()
