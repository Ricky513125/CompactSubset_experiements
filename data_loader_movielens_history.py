"""
MovieLens 数据处理函数 - 支持历史策略
"""

import random
from typing import List, Dict, Any, Optional


def extract_movielens_samples(train_data: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    """
    从 MovieLens 原始数据中提取训练样本
    
    MovieLens 数据格式:
    {
        "user": {"profile": {"name": "user_xxx"}},
        "user_hash": "xxx",
        "task": {
            "description": "...",
            "task_behavior_collections": [
                {
                    "type": "movie_review",
                    "data": [
                        {
                            "continuation": "5.0",
                            "continuation_prefix": "Movie Title (Year) (Genres): "
                        },
                        ...
                    ]
                }
            ]
        }
    }
    
    Args:
        train_data: 原始训练数据
        debug: 是否打印调试信息
    
    Returns:
        提取的样本列表，每个样本包含:
        - user_hash: 用户标识
        - user_profile: 用户档案
        - task_text: 任务描述
        - continuation_prefix: 电影信息（前缀）
        - next_question: 评分（答案）
    """
    all_samples = []
    
    if debug:
        print(f"\n开始提取 MovieLens 样本，总数据项数: {len(train_data)}")
        print("=" * 50)
    
    for data_item in train_data:
        user_hash = data_item.get('user_hash', 'unknown')
        user_profile = data_item.get('user', {}).get('profile', {})
        task = data_item.get('task', {})
        task_description = task.get('description', '')
        
        # 获取电影评分数据
        task_behavior_collections = task.get('task_behavior_collections', [])
        
        for collection in task_behavior_collections:
            if collection.get('type') == 'movie_review':
                movie_reviews = collection.get('data', [])
                
                # if debug and len(movie_reviews) > 0:
                #     print(f"用户 {user_hash[:8]}... 有 {len(movie_reviews)} 个电影评分")
                
                # 为每个电影评分创建一个训练样本
                for review_item in movie_reviews:
                    continuation_prefix = review_item.get('continuation_prefix', '')
                    rating = review_item.get('continuation', '')
                    
                    if not continuation_prefix or not rating:
                        if debug:
                            print(f"  ⚠️  跳过：缺少电影信息或评分")
                        continue
                    
                    sample = {
                        'user_hash': user_hash,
                        'user_profile': user_profile,
                        'task_text': task_description,
                        'continuation_prefix': continuation_prefix,
                        'next_question': rating,
                        'history': []  # 稍后添加
                    }
                    
                    all_samples.append(sample)
    
    if debug:
        print("=" * 50)
        print(f"提取完成！有效样本总数: {len(all_samples)}")
        print("=" * 50)
    
    return all_samples


def add_history_to_samples_movielens(
    all_samples: List[Dict[str, Any]],
    history_strategy: str = 'all_previous',
    history_ratio: float = 0.5,
    fixed_history_count: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    为 MovieLens 样本添加历史信息，支持多种历史划分策略
    
    MovieLens 特点：
    - 每个用户有多个电影评分（按时间顺序）
    - 为每个评分创建训练样本
    - 历史 = 之前的评分记录
    
    Args:
        all_samples: 所有训练样本（已按时间排序）
        history_strategy: 历史划分策略
            - 'all_previous': 所有之前的评分作为历史（默认）
            - 'fixed_ratio': 固定比例的之前评分作为历史
            - 'fixed_count': 固定数量的之前评分作为历史
            - 'random': 随机选择一定比例的之前评分
            - 'none': 不使用历史
        history_ratio: 历史比例（用于 fixed_ratio 和 random）
        fixed_history_count: 固定历史数量（用于 fixed_count）
    
    Returns:
        添加了历史信息的样本列表
    """
    # 按用户分组
    user_samples = {}
    for sample in all_samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    # 为每个用户的样本添加历史
    samples_with_history = []
    
    for user_hash, user_sample_list in user_samples.items():
        for idx, sample in enumerate(user_sample_list):
            # 获取该样本之前的所有评分
            previous_samples = user_sample_list[:idx]
            
            # 根据策略选择历史
            if history_strategy == 'none' or not previous_samples:
                history = []
            
            elif history_strategy == 'all_previous':
                # 所有之前的评分
                history = [
                    f"{s.get('continuation_prefix', '未知电影: ')}{s.get('next_question', '')}"
                    for s in previous_samples
                ]
            
            elif history_strategy == 'fixed_ratio':
                # 固定比例
                num_history = max(1, int(len(previous_samples) * history_ratio))
                selected = previous_samples[-num_history:]  # 取最近的N个
                history = [
                    f"{s.get('continuation_prefix', '未知电影: ')}{s.get('next_question', '')}"
                    for s in selected
                ]
            
            elif history_strategy == 'fixed_count':
                # 固定数量
                if fixed_history_count is None:
                    num_history = min(10, len(previous_samples))
                else:
                    num_history = min(fixed_history_count, len(previous_samples))
                selected = previous_samples[-num_history:]  # 取最近的N个
                history = [
                    f"{s.get('continuation_prefix', '未知电影: ')}{s.get('next_question', '')}"
                    for s in selected
                ]
            
            elif history_strategy == 'random':
                # 随机选择
                num_history = max(1, int(len(previous_samples) * history_ratio))
                num_history = min(num_history, len(previous_samples))
                selected = random.sample(previous_samples, num_history)
                # 按原始顺序排序（保持时间顺序）
                selected = sorted(selected, key=lambda x: previous_samples.index(x))
                history = [
                    f"{s.get('continuation_prefix', '未知电影: ')}{s.get('next_question', '')}"
                    for s in selected
                ]
            
            else:
                raise ValueError(f"Unknown history_strategy: {history_strategy}")
            
            # 添加历史到样本
            sample['history'] = history
            samples_with_history.append(sample)
    
    return samples_with_history


if __name__ == "__main__":
    # 测试示例
    test_samples = [
        {
            'user_hash': 'user1',
            'continuation_prefix': 'Movie A (2020) (Action): ',
            'next_question': '5.0',
        },
        {
            'user_hash': 'user1',
            'continuation_prefix': 'Movie B (2021) (Comedy): ',
            'next_question': '4.0',
        },
        {
            'user_hash': 'user1',
            'continuation_prefix': 'Movie C (2022) (Drama): ',
            'next_question': '3.5',
        },
        {
            'user_hash': 'user1',
            'continuation_prefix': 'Movie D (2023) (Thriller): ',
            'next_question': '4.5',
        },
    ]
    
    print("测试 all_previous 策略:")
    result = add_history_to_samples_movielens(test_samples, 'all_previous')
    for i, s in enumerate(result):
        print(f"样本{i+1}: history={len(s['history'])} items")
        if s['history']:
            print(f"  最后一条历史: {s['history'][-1]}")
    
    print("\n测试 random 策略 (ratio=0.5):")
    result = add_history_to_samples_movielens(test_samples, 'random', history_ratio=0.5)
    for i, s in enumerate(result):
        print(f"样本{i+1}: history={len(s['history'])} items")
        for h in s['history']:
            print(f"  - {h}")
