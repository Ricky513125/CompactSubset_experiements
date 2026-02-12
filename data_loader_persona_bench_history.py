"""
PERSONA_Bench 数据处理函数 - 支持时序历史
"""

from typing import List, Dict, Any


def add_history_to_samples_persona_bench(
    all_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    为 PERSONA_Bench 样本添加时序历史信息
    
    PERSONA_Bench 特点：
    - 每个用户有多个对话回复（按时间顺序排列）
    - 数据已按照主用户对评论回复的时间升序排序
    - 为每个回复创建训练样本
    - 历史 = 该用户之前的所有回复（不包括当前）
    
    数据格式:
    {
        "context": [
            {"source": "user_name", "content": "...", "timestamp": "..."},
            ...
        ],
        "continuation": "用户的回复",
        "timestamp": "2014-03-07 00:03:58"
    }
    
    Args:
        all_samples: 所有训练样本（已按时间排序）
    
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
            # 获取该样本之前的所有回复作为历史
            previous_samples = user_sample_list[:idx]
            
            # 从之前的样本中提取用户的回复内容
            history = []
            for prev_sample in previous_samples:
                # PERSONA_Bench 中，continuation 就是用户的回复
                continuation = prev_sample.get('next_question', '').strip()
                if not continuation:
                    continuation = prev_sample.get('continuation', '').strip()
                
                if continuation:
                    # 可以选择包含时间戳信息
                    timestamp = prev_sample.get('timestamp', '')
                    if timestamp:
                        history.append(f"[{timestamp}] {continuation}")
                    else:
                        history.append(continuation)
            
            # 添加历史到样本
            sample['history'] = history
            samples_with_history.append(sample)
    
    return samples_with_history


if __name__ == "__main__":
    # 测试示例
    test_samples = [
        {
            'user_hash': 'user1',
            'next_question': 'First post',
            'timestamp': '2014-01-01 10:00:00',
            'context': []
        },
        {
            'user_hash': 'user1',
            'next_question': 'Second reply',
            'timestamp': '2014-01-01 11:00:00',
            'context': [{'source': 'other_user', 'content': 'Comment'}]
        },
        {
            'user_hash': 'user1',
            'next_question': 'Third reply',
            'timestamp': '2014-01-01 12:00:00',
            'context': [{'source': 'other_user', 'content': 'Another comment'}]
        },
        {
            'user_hash': 'user2',
            'next_question': 'User2 first post',
            'timestamp': '2014-01-01 10:30:00',
            'context': []
        },
    ]
    
    print("测试 PERSONA_Bench 时序历史:")
    result = add_history_to_samples_persona_bench(test_samples)
    for i, s in enumerate(result):
        print(f"\n样本{i+1} (user: {s['user_hash']}):")
        print(f"  回复: {s['next_question']}")
        print(f"  历史条数: {len(s['history'])}")
        if s['history']:
            print(f"  历史内容:")
            for h in s['history']:
                print(f"    - {h}")
