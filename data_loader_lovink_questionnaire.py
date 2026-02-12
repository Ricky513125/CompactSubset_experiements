"""
Lovink问卷数据加载器
专门用于处理问卷问答数据
"""
import json
from typing import List, Dict, Any, Tuple
import random


def load_questionnaire_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载Lovink问卷数据
    
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


def extract_questionnaire_samples(
    raw_data: List[Dict[str, Any]], 
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    将原始问卷数据转换为训练样本格式
    
    每个问答对转换为一个样本：
    - user_profile: 用户信息
    - context: 问卷问题（作为对话上下文）
    - next_question: 用户的回答（要预测的内容）
    
    Args:
        raw_data: 原始数据
        debug: 是否输出调试信息
        
    Returns:
        训练样本列表
    """
    all_samples = []
    
    for user_data in raw_data:
        user_profile = user_data.get('user', {}).get('profile', {})
        task_desc = user_data.get('task', {}).get('description', '')
        
        # 获取问卷数据
        task_collections = user_data.get('task', {}).get('task_behavior_collections', [])
        
        for collection in task_collections:
            if collection.get('type') != 'dialogue':
                continue
            
            qa_pairs = collection.get('data', [])
            
            if debug:
                print(f"处理用户: {user_profile.get('name', 'Unknown')}")
                print(f"任务描述: {task_desc}")
                print(f"问答对总数: {len(qa_pairs)}")
            
            # 为每个问答对创建一个训练样本
            for i, qa in enumerate(qa_pairs):
                context_items = qa.get('context', [])
                answer = qa.get('continuation', '')
                
                # 构建context（问卷问题）
                context = []
                for ctx_item in context_items:
                    # 问卷问题作为user角色
                    context.append({
                        'role': 'user',
                        'content': ctx_item.get('content', ''),
                        'source': ctx_item.get('source', 'questionnaire')
                    })
                
                sample = {
                    'user_profile': user_profile,
                    'user_hash': user_profile.get('name', 'unknown'),
                    'task_description': task_desc,
                    
                    # 问卷问题作为context
                    'context': context,
                    
                    # 用户的回答作为要预测的内容
                    'next_question': answer,
                    
                    # 问答对索引
                    'qa_index': i,
                    
                    # 原始数据（用于调试）
                    'raw_qa': qa
                }
                
                all_samples.append(sample)
            
            if debug:
                print(f"生成样本数: {len(all_samples)}")
    
    return all_samples


def add_questionnaire_history_to_samples(
    samples: List[Dict[str, Any]],
    history_strategy: str = 'all_previous',
    history_ratio: float = 0.5,
    fixed_history_count: int = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    为问卷样本添加历史回答信息（支持多种策略）
    
    Args:
        samples: 样本列表
        history_strategy: 历史划分策略
            - 'all_previous': 使用所有之前的问答（默认）
            - 'fixed_ratio': 使用前N%的问答作为历史
            - 'fixed_count': 使用固定数量的问答作为历史
            - 'random': 随机选择一部分问答作为历史
            - 'none': 不使用历史（每个问题独立）
        history_ratio: 当strategy='fixed_ratio'时，历史所占比例
        fixed_history_count: 当strategy='fixed_count'时，历史问答数量
        seed: 随机种子
        
    Returns:
        添加了历史信息的样本列表
    """
    random.seed(seed)
    
    # 按用户分组
    user_samples = {}
    for sample in samples:
        user_hash = sample['user_hash']
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    # 为每个用户的样本添加历史
    for user_hash, user_sample_list in user_samples.items():
        # 按qa_index排序（确保顺序正确）
        user_sample_list.sort(key=lambda x: x.get('qa_index', 0))
        
        total_qa = len(user_sample_list)
        
        for i, sample in enumerate(user_sample_list):
            history = []
            
            if history_strategy == 'none':
                # 不使用历史
                pass
            
            elif history_strategy == 'all_previous':
                # 使用所有之前的问答
                for j in range(i):
                    prev_sample = user_sample_list[j]
                    question = ''
                    if prev_sample['context']:
                        question = prev_sample['context'][0]['content']
                    answer = prev_sample['next_question']
                    history.append(f"问题：{question}\n回答：{answer}")
            
            elif history_strategy == 'fixed_ratio':
                # 前N%的问答作为历史，后面的作为要预测的
                history_count = int(total_qa * history_ratio)
                if i < history_count:
                    # 当前样本在历史范围内，使用之前的所有
                    for j in range(i):
                        prev_sample = user_sample_list[j]
                        question = ''
                        if prev_sample['context']:
                            question = prev_sample['context'][0]['content']
                        answer = prev_sample['next_question']
                        history.append(f"问题：{question}\n回答：{answer}")
                else:
                    # 当前样本在预测范围，使用前history_count个作为历史
                    for j in range(history_count):
                        prev_sample = user_sample_list[j]
                        question = ''
                        if prev_sample['context']:
                            question = prev_sample['context'][0]['content']
                        answer = prev_sample['next_question']
                        history.append(f"问题：{question}\n回答：{answer}")
            
            elif history_strategy == 'fixed_count':
                # 使用固定数量的问答作为历史
                if fixed_history_count is None:
                    fixed_history_count = min(5, total_qa // 2)
                
                start_idx = max(0, i - fixed_history_count)
                for j in range(start_idx, i):
                    prev_sample = user_sample_list[j]
                    question = ''
                    if prev_sample['context']:
                        question = prev_sample['context'][0]['content']
                    answer = prev_sample['next_question']
                    history.append(f"问题：{question}\n回答：{answer}")
            
            elif history_strategy == 'random':
                # 从之前的问答中随机选择一部分作为历史
                if i > 0:
                    available_indices = list(range(i))
                    num_history = min(len(available_indices), max(1, i // 2))
                    selected_indices = random.sample(available_indices, num_history)
                    selected_indices.sort()
                    
                    for j in selected_indices:
                        prev_sample = user_sample_list[j]
                        question = ''
                        if prev_sample['context']:
                            question = prev_sample['context'][0]['content']
                        answer = prev_sample['next_question']
                        history.append(f"问题：{question}\n回答：{answer}")
            
            sample['history'] = history
            sample['history_strategy'] = history_strategy
    
    return samples


def create_train_test_split_by_question_position(
    samples: List[Dict[str, Any]],
    train_question_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    基于问题位置划分训练/测试集
    
    策略：前N%的问题用于训练（作为先验知识），后面的问题用于测试
    这样可以模拟"学习前面的问答风格，预测后面的回答"
    
    Args:
        samples: 样本列表
        train_question_ratio: 训练集问题比例
        seed: 随机种子
        
    Returns:
        (train_samples, test_samples)
    """
    # 按用户分组
    user_samples = {}
    for sample in samples:
        user_hash = sample['user_hash']
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    train_samples = []
    test_samples = []
    
    # 为每个用户划分
    for user_hash, user_sample_list in user_samples.items():
        # 按qa_index排序
        user_sample_list.sort(key=lambda x: x.get('qa_index', 0))
        
        # 前N%用于训练
        split_idx = int(len(user_sample_list) * train_question_ratio)
        split_idx = max(1, split_idx)  # 至少保留1个训练样本
        
        train_samples.extend(user_sample_list[:split_idx])
        test_samples.extend(user_sample_list[split_idx:])
    
    return train_samples, test_samples


def split_train_val(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    划分训练集和验证集
    
    采用随机划分（问卷数据通常没有时间顺序要求）
    
    Args:
        samples: 样本列表
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_samples, val_samples)
    """
    random.seed(seed)
    
    # 随机打乱
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    # 划分
    split_idx = int(len(shuffled_samples) * (1 - val_ratio))
    train_samples = shuffled_samples[:split_idx]
    val_samples = shuffled_samples[split_idx:]
    
    return train_samples, val_samples


def format_questionnaire_prompt(
    sample: Dict[str, Any],
    use_profile: bool = True,
    use_history: bool = True,
    style: str = 'simple'
) -> str:
    """
    格式化问卷样本为训练prompt
    
    Args:
        sample: 样本数据
        use_profile: 是否使用用户profile
        use_history: 是否使用历史回答
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
            parts.append(f"用户ID: {profile.get('name', 'Unknown')}")
            if sample.get('task_description'):
                parts.append(f"任务: {sample['task_description']}")
        parts.append("")
    
    # 2. 历史回答
    if use_history and sample.get('history'):
        history = sample['history']
        if style == 'simple':
            parts.append(f"[HISTORY] 历史回答 ({len(history)}条):")
            for h in history[-5:]:  # 只显示最近5条
                parts.append(f"  {h}")
        else:
            parts.append("=== 历史回答 ===")
            for i, h in enumerate(history[-5:], 1):
                parts.append(f"{i}. {h}")
        parts.append("")
    
    # 3. 当前问题
    if sample.get('context'):
        question = sample['context'][0]['content']
        if style == 'simple':
            parts.append(f"[QUESTION] {question}")
        else:
            parts.append("=== 当前问题 ===")
            parts.append(question)
            parts.append("\n请给出你的回答：")
    
    return "\n".join(parts)


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("用法: python data_loader_lovink_questionnaire.py <json_file>")
        sys.exit(1)
    
    print("加载数据...")
    data = load_questionnaire_data(test_file)
    
    print("提取样本...")
    samples = extract_questionnaire_samples(data, debug=True)
    
    print("\n添加历史信息...")
    samples = add_questionnaire_history_to_samples(samples)
    
    print("\n划分数据集...")
    train, val = split_train_val(samples, val_ratio=0.1, seed=42)
    print(f"训练集: {len(train)} 个样本")
    print(f"验证集: {len(val)} 个样本")
    
    print("\n示例样本（训练集第1个）:")
    if train:
        print(format_questionnaire_prompt(train[0], style='detailed'))
        print(f"\n目标输出: {train[0]['next_question']}")
    
    print("\n示例样本（带历史，训练集最后1个）:")
    if train:
        print(format_questionnaire_prompt(train[-1], style='detailed'))
        print(f"\n目标输出: {train[-1]['next_question']}")
