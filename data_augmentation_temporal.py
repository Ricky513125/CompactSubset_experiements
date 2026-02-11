"""
时序数据扩充模块
将每个用户的时间序列历史转换为多个训练样本，实现数据扩充

使用方法:
from data_augmentation_temporal import expand_samples_with_temporal_history
expanded_samples = expand_samples_with_temporal_history(samples, min_history_length=1)
"""

from typing import List, Dict, Any, Optional
import copy


def expand_samples_with_temporal_history(
    samples: List[Dict[str, Any]],
    min_history_length: int = 1,
    max_samples_per_user: Optional[int] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    对每个用户的时间序列历史进行扩充，生成多个训练样本
    
    原理：
    假设用户有5个影评 [r1, r2, r3, r4, r5]，按时间顺序排列
    扩充后会生成:
      - 样本1: 历史[] -> 预测r1 (如果 min_history_length=0)
      - 样本2: 历史[r1] -> 预测r2
      - 样本3: 历史[r1, r2] -> 预测r3
      - 样本4: 历史[r1, r2, r3] -> 预测r4
      - 样本5: 历史[r1, r2, r3, r4] -> 预测r5
    
    Args:
        samples: 原始样本列表
        min_history_length: 最小历史长度（设为0则第一个也当样本，设为1则需要至少1个历史）
        max_samples_per_user: 每个用户最多生成多少个样本（None表示不限制）
        verbose: 是否打印详细信息
    
    Returns:
        扩充后的样本列表
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"时序数据扩充")
        print(f"{'='*80}")
        print(f"原始样本数: {len(samples)}")
        print(f"最小历史长度: {min_history_length}")
        print(f"每用户最大样本数: {max_samples_per_user if max_samples_per_user else '不限制'}")
        print(f"{'='*80}\n")
    
    # 1. 按用户分组并按时间排序
    user_samples = {}
    for sample in samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    # 2. 对每个用户的样本进行扩充
    expanded_samples = []
    
    for user_hash, user_sample_list in user_samples.items():
        # 假设samples已经按时间排序（如果没有，需要根据时间戳排序）
        # 这里直接使用原有顺序
        
        num_samples = len(user_sample_list)
        
        # 决定实际生成多少个样本
        if max_samples_per_user:
            num_to_generate = min(num_samples, max_samples_per_user)
            # 如果限制了数量，从后往前取（保留最近的）
            start_idx = num_samples - num_to_generate
        else:
            num_to_generate = num_samples
            start_idx = 0
        
        if verbose and num_samples > 1:
            print(f"用户 {user_hash[:8]}... : {num_samples} 个原始样本 -> 生成 {num_to_generate} 个扩充样本")
        
        # 3. 生成扩充样本
        for i in range(start_idx, num_samples):
            current_sample = user_sample_list[i]
            
            # 历史长度检查
            history_length = i
            if history_length < min_history_length:
                continue  # 跳过历史长度不足的样本
            
            # 创建新样本（深拷贝）
            new_sample = copy.deepcopy(current_sample)
            
            # 构建历史：从该用户的前i个样本中提取next_question作为历史
            historical_reviews = []
            for j in range(i):
                prev_sample = user_sample_list[j]
                prev_review = prev_sample.get('next_question', '').strip()
                if prev_review:
                    historical_reviews.append(prev_review)
            
            # 更新样本的history字段
            new_sample['history'] = historical_reviews
            
            # 添加扩充元数据（可选，用于调试）
            new_sample['_augmentation_meta'] = {
                'original_index': i,
                'history_length': len(historical_reviews),
                'user_total_samples': num_samples
            }
            
            expanded_samples.append(new_sample)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"扩充完成")
        print(f"{'='*80}")
        print(f"扩充后样本数: {len(expanded_samples)}")
        if len(samples) > 0:
            print(f"扩充倍数: {len(expanded_samples) / len(samples):.2f}x")
        else:
            print(f"⚠️  警告: 原始样本数为0，无法计算扩充倍数")
        print(f"{'='*80}\n")
    
    return expanded_samples


def expand_samples_with_sliding_window(
    samples: List[Dict[str, Any]],
    window_size: int = 5,
    stride: int = 1,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    使用滑动窗口策略扩充样本
    
    与完整历史不同，这个方法只保留固定窗口大小的历史
    适合历史很长但不想全部使用的情况
    
    Args:
        samples: 原始样本列表
        window_size: 滑动窗口大小
        stride: 滑动步长
        verbose: 是否打印详细信息
    
    Returns:
        扩充后的样本列表
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"滑动窗口数据扩充")
        print(f"{'='*80}")
        print(f"原始样本数: {len(samples)}")
        print(f"窗口大小: {window_size}")
        print(f"滑动步长: {stride}")
        print(f"{'='*80}\n")
    
    # 按用户分组
    user_samples = {}
    for sample in samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    expanded_samples = []
    
    for user_hash, user_sample_list in user_samples.items():
        num_samples = len(user_sample_list)
        
        # 使用滑动窗口
        for i in range(0, num_samples, stride):
            if i < window_size - 1:
                continue  # 跳过历史不足window_size的位置
            
            current_sample = user_sample_list[i]
            new_sample = copy.deepcopy(current_sample)
            
            # 只保留window_size个历史
            start_idx = max(0, i - window_size + 1)
            historical_reviews = []
            for j in range(start_idx, i):
                prev_review = user_sample_list[j].get('next_question', '').strip()
                if prev_review:
                    historical_reviews.append(prev_review)
            
            new_sample['history'] = historical_reviews
            new_sample['_augmentation_meta'] = {
                'window_start': start_idx,
                'window_end': i,
                'window_size': len(historical_reviews)
            }
            
            expanded_samples.append(new_sample)
    
    if verbose:
        print(f"\n扩充后样本数: {len(expanded_samples)}")
        print(f"扩充倍数: {len(expanded_samples) / len(samples):.2f}x\n")
    
    return expanded_samples


def print_augmentation_stats(samples: List[Dict[str, Any]]):
    """打印扩充数据的统计信息"""
    print(f"\n{'='*80}")
    print("数据扩充统计")
    print(f"{'='*80}")
    
    total = len(samples)
    print(f"总样本数: {total}")
    
    # 统计history长度分布
    history_lengths = []
    for sample in samples:
        history = sample.get('history', [])
        if isinstance(history, list):
            history_lengths.append(len(history))
        else:
            history_lengths.append(0)
    
    if history_lengths:
        print(f"\n历史长度分布:")
        print(f"  最小: {min(history_lengths)}")
        print(f"  最大: {max(history_lengths)}")
        print(f"  平均: {sum(history_lengths) / len(history_lengths):.2f}")
        print(f"  中位数: {sorted(history_lengths)[len(history_lengths)//2]}")
        
        # 分布直方图
        from collections import Counter
        hist_counter = Counter(history_lengths)
        print(f"\n详细分布:")
        for length in sorted(hist_counter.keys())[:10]:  # 只显示前10个
            count = hist_counter[length]
            percentage = count / total * 100
            bar = '█' * int(percentage / 2)
            print(f"  长度{length:2d}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        if len(hist_counter) > 10:
            print(f"  ... (还有 {len(hist_counter) - 10} 个长度)")
    
    # 统计用户分布
    user_counts = {}
    for sample in samples:
        user_hash = sample.get('user_hash', 'unknown')
        user_counts[user_hash] = user_counts.get(user_hash, 0) + 1
    
    print(f"\n用户统计:")
    print(f"  用户总数: {len(user_counts)}")
    user_sample_counts = list(user_counts.values())
    print(f"  每用户样本数:")
    print(f"    最小: {min(user_sample_counts)}")
    print(f"    最大: {max(user_sample_counts)}")
    print(f"    平均: {sum(user_sample_counts) / len(user_sample_counts):.2f}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # 测试示例
    test_samples = [
        {'user_hash': 'user1', 'next_question': 'review1', 'context': []},
        {'user_hash': 'user1', 'next_question': 'review2', 'context': []},
        {'user_hash': 'user1', 'next_question': 'review3', 'context': []},
        {'user_hash': 'user1', 'next_question': 'review4', 'context': []},
        {'user_hash': 'user2', 'next_question': 'review1', 'context': []},
        {'user_hash': 'user2', 'next_question': 'review2', 'context': []},
    ]
    
    print("测试：完整历史扩充")
    expanded = expand_samples_with_temporal_history(test_samples, min_history_length=0, verbose=True)
    for i, sample in enumerate(expanded):
        print(f"样本{i+1}: 历史={sample['history']} -> 预测={sample['next_question']}")
    
    print_augmentation_stats(expanded)
