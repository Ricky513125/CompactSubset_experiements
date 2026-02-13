"""
豆瓣影评数据加载器
专门用于处理电影评论数据，按时间顺序划分训练/验证/测试集
"""
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime


def load_movie_review_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载豆瓣影评数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持单个用户或多用户数据
    if isinstance(data, dict):
        data = [data]
    
    return data


def extract_movie_review_samples(
    raw_data: List[Dict[str, Any]], 
    one_sample_per_user: bool = False,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    将原始影评数据转换为训练样本格式
    
    两种模式：
    1. one_sample_per_user=False（默认）：每条影评转换为一个样本
       - 用户有100条影评 → 生成100个样本
       - 样本1: [] → r1, 样本2: [r1] → r2, ..., 样本100: [r1..r99] → r100
    
    2. one_sample_per_user=True：每个用户只生成1个样本
       - 用户有100条影评 → 生成1个样本
       - 样本: [r1..r99] → r100（用前n-1条预测第n条）
       - **大幅减少训练数据量，缩短训练时间**
    
    Args:
        raw_data: 原始数据
        one_sample_per_user: 是否每个用户只生成一个样本（默认False）
        debug: 是否输出调试信息
        
    Returns:
        训练样本列表（按时间顺序）
    """
    all_samples = []
    
    for user_data in raw_data:
        user_profile = user_data.get('user', {}).get('profile', {})
        task_desc = user_data.get('task', {}).get('description', '')
        
        # 获取影评数据（已按时间排序）
        task_collections = user_data.get('task', {}).get('task_behavior_collections', [])
        
        for collection in task_collections:
            if collection.get('type') != 'movie_review':
                continue
            
            reviews = collection.get('data', [])
            
            if debug:
                print(f"处理用户: {user_profile.get('name', 'Unknown')}")
                print(f"任务描述: {task_desc}")
                print(f"影评总数: {len(reviews)}")
            
            if one_sample_per_user:
                # 🔥 新模式：每个用户只生成1个样本
                # 使用前 n-1 条作为历史，预测第 n 条
                if len(reviews) < 2:
                    if debug:
                        print(f"  ⚠️ 跳过该用户（影评数 < 2）")
                    continue
                
                # 所有影评除最后一条作为历史
                history_reviews = reviews[:-1]
                last_review = reviews[-1]
                
                sample = {
                    'user_profile': user_profile,
                    'user_hash': user_profile.get('name', 'unknown'),
                    'task_description': task_desc,
                    
                    # 历史影评（前 n-1 条）
                    'history': [
                        {
                            'movie': h.get('continuation_prefix', '').rstrip(': '),
                            'review': h.get('continuation', ''),
                            'timestamp': h.get('timestamp', '')
                        }
                        for h in history_reviews
                    ],
                    
                    # 当前电影信息（第 n 条）
                    'movie_name': last_review.get('continuation_prefix', '').rstrip(': '),
                    'timestamp': last_review.get('timestamp', ''),
                    
                    # 目标：要预测的影评（第 n 条）
                    'next_question': last_review.get('continuation', ''),
                    
                    # context保持空列表（兼容现有框架）
                    'context': last_review.get('context', []),
                    
                    # 元数据
                    'total_reviews': len(reviews),
                    'history_count': len(history_reviews),
                    'raw_review': last_review
                }
                
                all_samples.append(sample)
                
                if debug:
                    print(f"  生成1个样本: {len(history_reviews)}条历史 → 预测第{len(reviews)}条")
            
            else:
                # 原模式：为每条影评创建一个训练样本
                for i, review in enumerate(reviews):
                    # 之前的所有影评作为历史上下文
                    history_reviews = reviews[:i] if i > 0 else []
                    
                    sample = {
                        'user_profile': user_profile,
                        'user_hash': user_profile.get('name', 'unknown'),
                        'task_description': task_desc,
                        
                        # 历史影评（作为上下文）
                        'history': [
                            {
                                'movie': h.get('continuation_prefix', '').rstrip(': '),
                                'review': h.get('continuation', ''),
                                'timestamp': h.get('timestamp', '')
                            }
                            for h in history_reviews
                        ],
                        
                        # 当前电影信息
                        'movie_name': review.get('continuation_prefix', '').rstrip(': '),
                        'timestamp': review.get('timestamp', ''),
                        
                        # 目标：要预测的影评
                        'next_question': review.get('continuation', ''),
                        
                        # context保持空列表（兼容现有框架）
                        'context': review.get('context', []),
                        
                        # 原始数据（用于调试）
                        'raw_review': review
                    }
                    
                    all_samples.append(sample)
            
            if debug:
                print(f"生成样本数: {len(all_samples)}")
    
    return all_samples


def split_movie_reviews_by_time(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    debug: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按时间顺序划分训练/验证/测试集
    
    重要：保持时间顺序，用早期数据训练，后期数据测试
    
    Args:
        samples: 样本列表（已按时间排序）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        debug: 是否输出调试信息
        
    Returns:
        (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1.0，当前为 {train_ratio + val_ratio + test_ratio}"
    
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    if debug:
        print("=" * 80)
        print("按时间顺序划分数据集:")
        print(f"  总样本数: {total}")
        print(f"  训练集: {len(train_samples)} ({len(train_samples)/total*100:.1f}%)")
        if train_samples:
            print(f"    时间范围: {train_samples[0].get('timestamp', 'N/A')} -> {train_samples[-1].get('timestamp', 'N/A')}")
        
        print(f"  验证集: {len(val_samples)} ({len(val_samples)/total*100:.1f}%)")
        if val_samples:
            print(f"    时间范围: {val_samples[0].get('timestamp', 'N/A')} -> {val_samples[-1].get('timestamp', 'N/A')}")
        
        print(f"  测试集: {len(test_samples)} ({len(test_samples)/total*100:.1f}%)")
        if test_samples:
            print(f"    时间范围: {test_samples[0].get('timestamp', 'N/A')} -> {test_samples[-1].get('timestamp', 'N/A')}")
        print("=" * 80)
    
    return train_samples, val_samples, test_samples


def add_cumulative_history_to_samples(
    samples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    为每个样本添加累积的历史信息
    
    这个函数确保每个样本的history字段包含了之前所有的影评
    （数据加载时已经处理，这里只是保持接口兼容）
    
    Args:
        samples: 样本列表
        
    Returns:
        处理后的样本列表
    """
    # 影评数据在extract时已经添加了history，这里直接返回
    return samples


def format_movie_review_prompt(
    sample: Dict[str, Any],
    use_profile: bool = True,
    use_history: bool = True,
    style: str = 'simple'
) -> str:
    """
    格式化影评样本为训练prompt
    
    Args:
        sample: 样本数据
        use_profile: 是否使用用户profile
        use_history: 是否使用历史影评
        style: prompt风格 ('simple' 或 'detailed')
        
    Returns:
        格式化后的prompt字符串
    """
    parts = []
    
    # 1. 用户Profile
    if use_profile and sample.get('user_profile'):
        profile = sample['user_profile']
        if style == 'simple':
            parts.append(f"[USER_PROFILE] 用户: {profile.get('name', 'Unknown')}")
        else:
            parts.append("=== 用户信息 ===")
            parts.append(f"用户名: {profile.get('name', 'Unknown')}")
            if sample.get('task_description'):
                parts.append(f"任务: {sample['task_description']}")
        parts.append("")
    
    # 2. 历史影评
    if use_history and sample.get('history'):
        history = sample['history']
        if style == 'simple':
            parts.append(f"[HISTORY] 历史影评 ({len(history)}条):")
            for h in history[-10:]:  # 只显示最近10条
                parts.append(f"  {h['movie']}: {h['review']}")
        else:
            parts.append("=== 历史影评 ===")
            for i, h in enumerate(history[-10:], 1):
                parts.append(f"{i}. {h['movie']} ({h['timestamp']})")
                parts.append(f"   评论: {h['review']}")
        parts.append("")
    
    # 3. 当前电影
    movie_name = sample.get('movie_name', '')
    if style == 'simple':
        parts.append(f"[MOVIE] {movie_name}:")
    else:
        parts.append("=== 当前电影 ===")
        parts.append(f"电影: {movie_name}")
        if sample.get('timestamp'):
            parts.append(f"时间: {sample['timestamp']}")
        parts.append("\n请写出这部电影的影评：")
    
    return "\n".join(parts)


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("用法: python data_loader_movie_review.py <json_file>")
        sys.exit(1)
    
    print("加载数据...")
    data = load_movie_review_data(test_file)
    
    print("提取样本...")
    samples = extract_movie_review_samples(data, debug=True)
    
    print("\n划分数据集...")
    train, val, test = split_movie_reviews_by_time(samples, debug=True)
    
    print("\n示例样本:")
    if train:
        print("\n训练集第1个样本:")
        print(format_movie_review_prompt(train[0], style='detailed'))
        print(f"\n目标输出: {train[0]['next_question']}")
    
    if test:
        print("\n" + "="*80)
        print("测试集第1个样本:")
        print(format_movie_review_prompt(test[0], style='detailed'))
        print(f"\n目标输出: {test[0]['next_question']}")
