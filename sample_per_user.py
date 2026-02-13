"""
对提取的样本进行随机采样，减少训练数据量
每个用户（角色）随机保留 N 个样本
"""
import random
from typing import List, Dict, Any


def sample_per_user(
    all_samples: List[Dict[str, Any]],
    max_samples_per_user: int = 2,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    对每个用户的样本进行随机采样
    
    Args:
        all_samples: 所有训练样本
        max_samples_per_user: 每个用户最多保留多少个样本
        random_seed: 随机种子（保证可复现）
    
    Returns:
        采样后的样本列表
    """
    random.seed(random_seed)
    
    # 按用户分组
    user_samples = {}
    for sample in all_samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    # 对每个用户的样本进行采样
    sampled_samples = []
    for user_hash, samples in user_samples.items():
        if len(samples) <= max_samples_per_user:
            # 样本数不超过限制，全部保留
            sampled_samples.extend(samples)
        else:
            # 随机采样
            sampled = random.sample(samples, max_samples_per_user)
            sampled_samples.extend(sampled)
    
    # 打印统计信息
    print(f"\n{'='*50}")
    print(f"样本采样统计:")
    print(f"  原始样本数: {len(all_samples)}")
    print(f"  用户数: {len(user_samples)}")
    print(f"  每用户最大样本数: {max_samples_per_user}")
    print(f"  采样后样本数: {len(sampled_samples)}")
    print(f"  采样比例: {len(sampled_samples) / len(all_samples) * 100:.1f}%")
    print(f"  预期训练时间缩短: {len(all_samples) / len(sampled_samples):.1f}x")
    print(f"{'='*50}\n")
    
    return sampled_samples


if __name__ == "__main__":
    # 测试
    test_samples = [
        {'user_hash': 'user1', 'text': 'sample1'},
        {'user_hash': 'user1', 'text': 'sample2'},
        {'user_hash': 'user1', 'text': 'sample3'},
        {'user_hash': 'user2', 'text': 'sample4'},
        {'user_hash': 'user2', 'text': 'sample5'},
    ]
    
    result = sample_per_user(test_samples, max_samples_per_user=2)
    print(f"原始: {len(test_samples)} 个样本")
    print(f"采样后: {len(result)} 个样本")
    for s in result:
        print(f"  {s}")
