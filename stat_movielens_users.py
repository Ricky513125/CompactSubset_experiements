#!/usr/bin/env python3
"""
统计 MovieLens 数据集中每个用户的记录数
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def load_movielens_data(file_path: str):
    """加载 MovieLens 数据"""
    print(f"正在加载数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def count_user_records(data):
    """统计每个用户的记录数"""
    user_record_counts = []
    
    for user_data in data:
        user_profile = user_data.get('user', {}).get('profile', {})
        user_name = user_profile.get('name', 'unknown')
        
        # 获取评分数据
        task_collections = user_data.get('task', {}).get('task_behavior_collections', [])
        
        total_records = 0
        for collection in task_collections:
            collection_type = collection.get('type', '')
            if collection_type in ['movie_rating', 'movie_review']:
                ratings = collection.get('data', [])
                total_records += len(ratings)
        
        if total_records > 0:
            user_record_counts.append({
                'user_name': user_name,
                'record_count': total_records
            })
    
    return user_record_counts

def main():
    # 数据路径
    data_path = "/mnt/parallel/GIDigitalTwinBench/RealSelf/MovieLens/train.json"
    
    if not Path(data_path).exists():
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    
    # 加载数据
    data = load_movielens_data(data_path)
    print(f"✓ 加载成功，总用户数: {len(data)}")
    
    # 统计每个用户的记录数
    user_record_counts = count_user_records(data)
    
    # 计算统计信息
    record_counts = [item['record_count'] for item in user_record_counts]
    
    if not record_counts:
        print("错误: 没有找到任何记录")
        sys.exit(1)
    
    total_users = len(record_counts)
    total_records = sum(record_counts)
    avg_records = total_records / total_users
    min_records = min(record_counts)
    max_records = max(record_counts)
    
    # 计算中位数
    sorted_counts = sorted(record_counts)
    median_records = sorted_counts[len(sorted_counts) // 2]
    
    # 计算分位数
    p25 = sorted_counts[len(sorted_counts) // 4]
    p75 = sorted_counts[len(sorted_counts) * 3 // 4]
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("MovieLens 数据集用户记录统计")
    print("=" * 80)
    print(f"总用户数: {total_users:,}")
    print(f"总记录数: {total_records:,}")
    print(f"\n记录数统计:")
    print(f"  平均每个用户: {avg_records:.2f} 条记录")
    print(f"  中位数: {median_records} 条记录")
    print(f"  最小值: {min_records} 条记录")
    print(f"  最大值: {max_records} 条记录")
    print(f"  25分位数: {p25} 条记录")
    print(f"  75分位数: {p75} 条记录")
    
    # 统计分布
    print(f"\n记录数分布:")
    ranges = [
        (1, 10, "1-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, float('inf'), "1001+")
    ]
    
    for min_val, max_val, label in ranges:
        count = sum(1 for c in record_counts if min_val <= c <= max_val)
        percentage = count / total_users * 100
        print(f"  {label:10s}: {count:6,} 用户 ({percentage:5.2f}%)")
    
    # 显示前10个记录数最多的用户
    print(f"\n记录数最多的前10个用户:")
    top_users = sorted(user_record_counts, key=lambda x: x['record_count'], reverse=True)[:10]
    for i, user_info in enumerate(top_users, 1):
        print(f"  {i:2d}. {user_info['user_name']:20s}: {user_info['record_count']:5d} 条记录")
    
    # 显示前10个记录数最少的用户
    print(f"\n记录数最少的前10个用户:")
    bottom_users = sorted(user_record_counts, key=lambda x: x['record_count'])[:10]
    for i, user_info in enumerate(bottom_users, 1):
        print(f"  {i:2d}. {user_info['user_name']:20s}: {user_info['record_count']:5d} 条记录")
    
    print("=" * 80)
    
    # 输出一个用户的50条数据示例
    print("\n" + "=" * 80)
    print("用户数据示例（前50条记录）")
    print("=" * 80)
    
    # 选择一个记录数较多的用户（至少50条记录）
    example_user = None
    for user_info in sorted(user_record_counts, key=lambda x: x['record_count'], reverse=True):
        if user_info['record_count'] >= 50:
            example_user = user_info['user_name']
            break
    
    if example_user:
        # 找到该用户的数据
        for user_data in data:
            user_profile = user_data.get('user', {}).get('profile', {})
            if user_profile.get('name') == example_user:
                print(f"用户: {example_user}")
                print(f"总记录数: {user_info['record_count']} 条")
                print(f"\n前50条记录:")
                print("-" * 80)
                
                task_collections = user_data.get('task', {}).get('task_behavior_collections', [])
                count = 0
                for collection in task_collections:
                    collection_type = collection.get('type', '')
                    if collection_type in ['movie_rating', 'movie_review']:
                        ratings = collection.get('data', [])
                        for i, rating in enumerate(ratings[:50], 1):
                            count += 1
                            movie_info = rating.get('continuation_prefix', '').rstrip(': ')
                            continuation = rating.get('continuation', '')
                            timestamp = rating.get('timestamp', '')
                            
                            print(f"\n记录 {i}:")
                            print(f"  电影: {movie_info}")
                            print(f"  评分/评论: {continuation[:100]}{'...' if len(continuation) > 100 else ''}")
                            print(f"  时间: {timestamp}")
                            if count >= 50:
                                break
                    if count >= 50:
                        break
                break
    else:
        print("未找到有50条以上记录的用户")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
