"""
分布式训练脚本 - 豆瓣影评模型（FlashAttention 2 + 动态Batch Padding）

用于训练用户影评风格模拟模型

使用方法：
# 单卡训练
python train_distributed_MovieReview.py \
    --data_file movie_review_data.json \
    --output_dir outputs/movie_review_model

# 多卡训练
torchrun --nproc_per_node=4 train_distributed_MovieReview.py \
    --data_file movie_review_data.json \
    --output_dir outputs/movie_review_model_4gpu
"""
import json
import argparse
import os
import sys
from pathlib import Path
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)
from typing import List, Dict, Any, Optional, Tuple
import torch.nn as nn
from torch.utils.data import Dataset

"""
豆瓣影评数据加载器
专门用于处理电影评论数据，按时间顺序划分训练/验证/测试集
"""
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


"""
消融实验训练脚本（带早停机制 + 动态Batch Padding优化）
关键优化：不再将batch内所有样本padding到固定max_length，
而是动态padding到batch内最长样本的实际长度，大幅节省显存。
"""


def split_train_val(samples, val_ratio=0.15, seed=42):
    """
    划分训练集和验证集（用户内划分，保证每个用户在训练和验证集都有样本）
    
    策略：对每个用户的样本进行随机打乱后按比例划分
    - 适用场景：测试集中的用户也出现在训练集中
    - 目标：学习基于用户已有对话预测新对话（用户内泛化）
    
    Args:
        samples: 所有训练样本
        val_ratio: 验证集比例（默认0.15，即15%）
        seed: 随机种子
    
    Returns:
        (train_samples, val_samples)
    """
    random.seed(seed)
    
    # 按用户分组
    user_samples = {}
    for sample in samples:
        user_hash = sample['user_hash']
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    train_samples = []
    val_samples = []
    
    # 对每个用户的样本进行划分
    for user_hash, user_data in user_samples.items():
        # 随机打乱该用户的样本
        random.shuffle(user_data)
        
        # 计算划分点：(1 - val_ratio) 的样本用于训练
        split_idx = int(len(user_data) * (1 - val_ratio))
        
        # 确保至少有1个样本在训练集（如果该用户只有1个样本，全部给训练集）
        if split_idx == 0 and len(user_data) > 0:
            split_idx = 1
        
        # 划分
        train_samples.extend(user_data[:split_idx])
        val_samples.extend(user_data[split_idx:])
    
    return train_samples, val_samples


def add_history_to_samples(train_samples, all_samples):
    """为每个样本添加历史信息（只包含用户的问题，不包含assistant内容）"""
    samples_with_history = []
    for sample in train_samples:
        user_hash = sample['user_hash']
        history = get_user_only_history(
            all_samples, 
            user_hash,
            current_sample=sample,
            current_context=sample.get('context'),
            max_history=15,
            use_cache=True
        )
        sample['history'] = history
        samples_with_history.append(sample)
    return samples_with_history


class DynamicPaddingDataset(Dataset):
    """
    优化版数据集：不做padding，返回原始长度的tensor
    padding将在collate_fn中动态进行
    
    注意：MovieReviewDataset 会覆盖 format_prompt 方法
    """
    def __init__(self, samples, tokenizer, max_length=32768, use_profile=True, use_history=True, use_context=True, verbose=False, use_detailed_template=True, max_context_turns=15, template_filename=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        self.use_detailed_template = use_detailed_template
        self.max_context_turns = max_context_turns
        self.template_filename = template_filename
        self.verbose = verbose
        
        # 截断统计
        self.truncation_stats = {
            'total_samples': 0,
            'truncated_samples': 0,
            'truncated_turns': 0,
            'total_history_items': 0,
            'truncated_history_items': 0,
            'samples_with_history': 0,
            'samples_with_history_truncated': 0
        }
        
        self.first_truncation_logged = False
    
    def build_movie_review_prompt(self, sample: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
        """
        构建影评训练prompt（简洁格式）
        
        返回:
            messages: 聊天消息列表
            target_answer: 目标答案（要预测的影评）
        """
        parts = []
        
        # 1. 用户Profile
        if self.use_profile and sample.get('user_profile'):
            profile = sample['user_profile']
            parts.append(f"用户: {profile.get('name', 'Unknown')}")
        
        # 2. 历史影评
        if self.use_history and sample.get('history'):
            history = sample['history']
            if history:
                parts.append(f"\n历史影评记录 ({len(history)}条):")
                for h in history:
                    parts.append(f"  电影《{h['movie']}》: {h['review']}")
        
        # 3. 当前电影
        movie_name = sample.get('movie_name', '')
        parts.append(f"\n模仿用户风格为电影《{movie_name}》写一条影评：")
        
        system_content = "\n".join(parts)
        
        messages = [
            {'role': 'system', 'content': system_content}
        ]
        
        target_answer = sample.get('next_question', '')
        
        return messages, target_answer

    def __len__(self):
        return len(self.samples)
    
    def get_truncation_stats(self):
        """获取截断统计信息"""
        if self.truncation_stats['total_samples'] == 0:
            return {
                'truncation_rate': 0.0,
                'avg_truncated_turns': 0.0,
                'total_samples': 0,
                'truncated_samples': 0,
                # 历史记录统计
                'history_truncation_rate': 0.0,
                'total_history_items': 0,
                'truncated_history_items': 0,
                'samples_with_history': 0,
                'samples_with_history_truncated': 0
            }
        
        truncation_rate = self.truncation_stats['truncated_samples'] / self.truncation_stats['total_samples']
        avg_truncated_turns = (self.truncation_stats['truncated_turns'] / self.truncation_stats['truncated_samples'] 
                               if self.truncation_stats['truncated_samples'] > 0 else 0)
        
        # 计算历史记录截断率
        history_truncation_rate = 0.0
        if self.truncation_stats['total_history_items'] > 0:
            history_truncation_rate = self.truncation_stats['truncated_history_items'] / self.truncation_stats['total_history_items']
        
        return {
            'truncation_rate': truncation_rate,
            'avg_truncated_turns': avg_truncated_turns,
            'total_samples': self.truncation_stats['total_samples'],
            'truncated_samples': self.truncation_stats['truncated_samples'],
            # 历史记录统计
            'history_truncation_rate': history_truncation_rate,
            'total_history_items': self.truncation_stats['total_history_items'],
            'truncated_history_items': self.truncation_stats['truncated_history_items'],
            'samples_with_history': self.truncation_stats['samples_with_history'],
            'samples_with_history_truncated': self.truncation_stats['samples_with_history_truncated']
        }

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        格式化样本为训练prompt（应该被子类覆盖）
        
        Returns:
            格式化后的prompt字符串
        """
        raise NotImplementedError("Subclass must implement format_prompt()")
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 格式化 prompt（由子类实现）
        prompt_text = self.format_prompt(sample)
        
        # 2. 获取目标答案
        target_answer = sample.get('next_question', '')
        
        # 3. 构建完整文本
        # 使用 Qwen 的对话格式
        messages = [
            {"role": "system", "content": prompt_text}
        ]
        
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        generation_suffix = "<|im_start|>assistant\n"
        full_prompt = full_prompt.strip() + generation_suffix
        im_end_token = "<|im_end|>"
        full_text = full_prompt + target_answer + im_end_token
        
        # 4. 编码 - 关键：不做padding！
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # 5. 计算labels
        target_ids = self.tokenizer.encode(target_answer, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        actual_prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        safe_prompt_len = min(actual_prompt_len, len(input_ids) - 1)
        labels[:safe_prompt_len] = -100
        
        # 屏蔽padding token
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'actual_length': len(input_ids)
        }


def dynamic_padding_collate_fn(examples, tokenizer):
    """
    动态Padding的collate函数
    关键优化：只padding到batch内最长样本的长度，而不是固定的max_length
    """
    # 找到batch中最长的序列长度
    max_length_in_batch = max(ex['input_ids'].shape[0] for ex in examples)
    
    # 打印batch信息（用于调试）
    lengths = [ex['input_ids'].shape[0] for ex in examples]
    if random.random() < 0.05:  # 5%的概率打印，避免刷屏
        print(f"[Batch Info] Lengths: {lengths}, Max: {max_length_in_batch}, Avg: {sum(lengths)/len(lengths):.0f}")
    
    batch = {}
    
    # 动态padding每个字段
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ex in examples:
        seq_len = ex['input_ids'].shape[0]
        pad_len = max_length_in_batch - seq_len
        
        # Padding input_ids
        padded_input_ids.append(
            torch.cat([
                ex['input_ids'],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
        )
        
        # Padding attention_mask
        padded_attention_mask.append(
            torch.cat([
                ex['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        
        # Padding labels
        padded_labels.append(
            torch.cat([
                ex['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )
    
    batch['input_ids'] = torch.stack(padded_input_ids)
    batch['attention_mask'] = torch.stack(padded_attention_mask)
    batch['labels'] = torch.stack(padded_labels)
    
    # 添加其他元信息（如果有）
    if 'actual_length' in examples[0]:
        batch['actual_length'] = [ex['actual_length'] for ex in examples]
    
    return batch


class AblationTrainerWithDynamicPadding(AblationTrainer):
    """带早停 + 动态Padding的训练器"""
    
    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: Optional[List[Dict[str, Any]]] = None,
        max_epochs: int = 10,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.00001
    ):
        """训练模型（带早停 + 动态Padding）"""
        train_config = self.config.get('training', {})
        
        # 创建数据集（使用动态Padding版本）
        print("创建训练数据集（动态Padding模式）...")
        train_dataset = DynamicPaddingDataset(
            samples=train_samples,
            tokenizer=self.tokenizer,
            max_length=train_config.get('max_length', 4096),
            use_profile=self.use_profile,
            use_history=self.use_history,
            use_context=self.use_context,
            max_context_turns=train_config.get('max_context_turns', 15)  # 新增：从 config 读取
        )
        
        val_dataset = None
        if val_samples:
            print("创建验证数据集（动态Padding模式）...")
            val_dataset = DynamicPaddingDataset(
                samples=val_samples,
                tokenizer=self.tokenizer,
                max_length=train_config.get('max_length', 4096),
                use_profile=self.use_profile,
                use_history=self.use_history,
                use_context=self.use_context,
                max_context_turns=train_config.get('max_context_turns', 15)  # 新增
            )
        
        # 数据整理器（动态Padding）
        def collate_fn(examples):
            return dynamic_padding_collate_fn(examples, self.tokenizer)
        
        # 计算每个epoch的步数和评估步数
        steps_per_epoch = len(train_dataset) // (train_config.get('batch_size', 1) * train_config.get('gradient_accumulation_steps', 16))
        eval_steps_value = max(1, steps_per_epoch // 2) if val_dataset else None
        
        # 调整 save_steps
        save_steps_value = train_config.get('save_steps', 500)
        if val_dataset and eval_steps_value and save_steps_value % eval_steps_value != 0:
            save_steps_value = ((save_steps_value + eval_steps_value - 1) // eval_steps_value) * eval_steps_value
            print(f"调整 save_steps 为 {save_steps_value}（eval_steps={eval_steps_value} 的整数倍）")
        
        # 学习率检查
        learning_rate = train_config.get('learning_rate', 1e-5)
        if learning_rate > 1e-5:
            print(f"警告: 学习率 {learning_rate} 可能过大")
        print(f"使用学习率: {learning_rate}")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=max_epochs,
            per_device_train_batch_size=train_config.get('batch_size', 2),
            per_device_eval_batch_size=train_config.get('eval_batch_size', 2),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 8),
            learning_rate=learning_rate,
            weight_decay=train_config.get('weight_decay', 0.01),
            warmup_steps=train_config.get('warmup_steps', 100),
            logging_steps=train_config.get('logging_steps', 10),
            save_steps=save_steps_value,
            eval_steps=eval_steps_value,
            eval_strategy="steps" if val_dataset else "no",
            save_total_limit=train_config.get('save_total_limit', 3),
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            bf16=True,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=0.5,
            report_to="wandb" if os.environ.get('WANDB_PROJECT') else "none",
            ddp_find_unused_parameters=False,
        )
        
        # 自定义 Trainer
        class CustomTrainer(Trainer):
            def __init__(self, *args, verbose_logging=False, log_dir=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.verbose_logging = verbose_logging
                self.log_dir = log_dir
                self.io_log_file = None
                
                # 创建输入输出日志文件
                if self.log_dir:
                    try:
                        os.makedirs(self.log_dir, exist_ok=True)
                        self.io_log_file = open(os.path.join(self.log_dir, 'io_logs.jsonl'), 'w', encoding='utf-8')
                        print(f"输入输出日志将保存到: {os.path.join(self.log_dir, 'io_logs.jsonl')}")
                    except Exception as e:
                        print(f"警告: 无法创建输入输出日志文件: {e}")
                        self.io_log_file = None
                else:
                    self.io_log_file = None
            
            def __del__(self):
                """关闭日志文件"""
                if hasattr(self, 'io_log_file') and self.io_log_file:
                    try:
                        self.io_log_file.close()
                    except:
                        pass
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """计算损失（带数值稳定性检查）"""
                # 移除actual_length字段（如果存在），避免传给模型
                actual_lengths = inputs.pop('actual_length', None)
                
                outputs = model(**inputs)
                logits = outputs.get("logits")
                labels = inputs.get("labels")
                input_ids = inputs.get("input_ids")
                
                # 检查并清理logits中的nan/inf
                if logits is not None:
                    has_nan = False
                    has_inf = False
                    
                    # 只检查部分数据，提高效率
                    if logits.numel() > 0:
                        check_size = min(1000, logits.numel() // 2)
                        if logits.numel() > check_size * 2:
                            head_values = logits.view(-1)[:check_size]
                            tail_values = logits.view(-1)[-check_size:]
                            if torch.isnan(head_values).any() or torch.isnan(tail_values).any():
                                has_nan = True
                            if torch.isinf(head_values).any() or torch.isinf(tail_values).any():
                                has_inf = True
                        else:
                            if torch.isnan(logits).any():
                                has_nan = True
                            if torch.isinf(logits).any():
                                has_inf = True
                    
                    # 如果发现问题，进行清理
                    if has_nan or has_inf:
                        nan_count = torch.isnan(logits).sum().item()
                        inf_count = torch.isinf(logits).sum().item()
                        
                        if nan_count > 0 or inf_count > 0:
                            print(f"警告: Step {self.state.global_step} logits中有 {nan_count} 个nan, {inf_count} 个inf")
                            logits = torch.where(
                                torch.isnan(logits) | torch.isinf(logits),
                                torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                                logits
                            )
                            logits = torch.clamp(logits, min=-50.0, max=50.0)
                
                # 计算损失
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                elif labels is not None:
                    valid_labels_count = (labels != -100).sum().item()
                    
                    if valid_labels_count == 0:
                        print(f"错误: Step {self.state.global_step} 没有有效的labels")
                        loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                    else:
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                else:
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                
                # 检查损失值
                if loss is not None and torch.is_tensor(loss):
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    loss_value = loss.item()
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"错误: Step {self.state.global_step} loss为nan/inf")
                        loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                        loss_value = 2.0
                    elif loss_value > 1e6:
                        print(f"警告: Step {self.state.global_step} loss过大 ({loss_value:.2f})")
                        loss = torch.clamp(loss, max=100.0)
                        loss_value = 100.0
                
                # 定期清理CUDA缓存
                if self.state.global_step % 10 == 0:
                    torch.cuda.empty_cache()
                
                if return_outputs:
                    return loss, outputs
                return loss
        
        # 创建早停回调
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        
        # 设置日志目录
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建 Trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,  # 使用动态padding的collate_fn
            processing_class=self.tokenizer,
            callbacks=[early_stopping] if val_dataset else [],
            verbose_logging=True,
            log_dir=log_dir,
        )
        
        # 开始训练
        print("="*80)
        print("🚀 开始训练（动态Batch Padding优化版）")
        print("="*80)
        print(f"训练样本数: {len(train_dataset)}")
        if val_dataset:
            print(f"验证样本数: {len(val_dataset)}")
        print(f"使用配置: profile={self.use_profile}, history={self.use_history}, context={self.use_context}")
        print(f"最大序列长度: {train_config.get('max_length', 4096)} (动态padding)")
        print(f"最大轮次: {max_epochs}")
        print(f"早停耐心值: {early_stopping_patience}")
        print("="*80)
        
        trainer.train()
        
        # 保存最终模型
        print(f"保存最终模型到 {self.output_dir}")
        try:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            print("✓ 模型保存成功")
        except Exception as e:
            print(f"警告: 保存模型时出错: {e}")
        
        # 输出截断统计
        if hasattr(train_dataset, 'get_truncation_stats'):
            stats = train_dataset.get_truncation_stats()
            print("\n" + "="*80)
            print("📊 训练数据截断统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  被截断样本数: {stats['truncated_samples']}")
            print(f"  Context 截断率: {stats['truncation_rate']:.2%}")
            print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
            
            # 如果使用了历史记录，输出历史记录统计
            if stats['samples_with_history'] > 0:
                print("\n  📚 历史记录统计:")
                print(f"    包含历史记录的样本数: {stats['samples_with_history']}")
                print(f"    历史记录总条目数: {stats['total_history_items']}")
                print(f"    被截断的历史条目数: {stats['truncated_history_items']}")
                print(f"    历史记录截断率: {stats['history_truncation_rate']:.2%}")
                print(f"    包含被截断历史的样本数: {stats['samples_with_history_truncated']}")
                if stats['samples_with_history'] > 0:
                    history_sample_rate = stats['samples_with_history_truncated'] / stats['samples_with_history']
                    print(f"    样本级历史截断率: {history_sample_rate:.2%}")
            print("="*80)
        
        print("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='消融实验训练（动态Padding优化版）')
    parser.add_argument('--config', type=str,
                       default='/data/lingyu.li/parallel-post-train/ablation/config.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--gpu', type=int, default=1,
                       help='使用的GPU编号（默认：1）')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='最大训练轮次（默认：50）')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值（默认：3）')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='早停阈值（默认：0.001）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Weights & Biases项目名称')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases运行名称')
    
    args = parser.parse_args()
    
    # 配置 Weights & Biases
    if args.wandb_project:
        try:
            import wandb
            os.environ['WANDB_PROJECT'] = args.wandb_project
            if args.wandb_run_name:
                os.environ['WANDB_NAME'] = args.wandb_run_name
            print(f"✓ 已启用 Weights & Biases 监控")
            print(f"  项目: {args.wandb_project}")
            if args.wandb_run_name:
                print(f"  运行名称: {args.wandb_run_name}")
        except ImportError:
            print("警告: wandb 未安装")
            args.wandb_project = None
    
    # 设置GPU
    physical_gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
    print(f"=" * 60)
    print(f"GPU 设置: 物理GPU {physical_gpu_id}")
    print(f"=" * 60)
    
    # 验证GPU
    if torch.cuda.is_available():
        print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 名称: {gpu_name}")
        print(f"GPU 总内存: {gpu_memory:.2f} GB")
    else:
        print("警告: CUDA 不可用")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取消融配置
    ablation_config = config['ablation_configs'][args.ablation_config]
    use_profile = ablation_config.get('use_profile', True)
    use_history = ablation_config.get('use_history', True)
    use_context = ablation_config.get('use_context', True)
    config_name = ablation_config['name']
    
    print("=" * 60)
    print(f"消融实验（动态Padding优化版）: {config_name}")
    print(f"使用配置: profile={use_profile}, history={use_history}, context={use_context}")
    print("=" * 60)
    
    # 加载训练数据
    print("加载训练数据...")
    train_path = config['data']['train_path']
    train_data = load_train_data(train_path)
    
    if not train_data:
        print(f"错误: 无法加载训练数据")
        return
    
    # 提取训练样本
    all_samples = extract_training_samples(train_data, debug=True)
    print(f"提取了 {len(all_samples)} 个训练样本")
    
    # 添加历史信息
    if use_history:
        print("添加历史信息...")
        all_samples = add_history_to_samples(all_samples, all_samples)
    
    # 划分训练集和验证集
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"验证集: {len(val_samples)} 个样本")
    
    # 获取模型配置
    model_config = config['model']
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用指定的输出目录: {output_dir}")
    else:
        checkpoint_dir = model_config['checkpoint_dir']
        dataset_name = os.path.basename(os.path.dirname(train_path))
        output_dir = os.path.join(checkpoint_dir, f"{dataset_name}_ablation_{config_name}_dynamic_padding")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
        except (OSError, IOError) as e:
            print(f"警告: 无法创建目录: {e}")
            local_checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
            output_dir = os.path.join(local_checkpoint_dir, f"{dataset_name}_ablation_{config_name}_dynamic_padding")
            os.makedirs(output_dir, exist_ok=True)
            print(f"使用本地目录: {output_dir}")
    
    # 创建训练器
    model_path = model_config['path']
    trainer = AblationTrainerWithDynamicPadding(
        model_path=model_path,
        output_dir=output_dir,
        config=config,
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context
    )
    
    # 开始训练
    trainer.train(
        train_samples, 
        val_samples,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    print(f"\n✅ 训练完成！模型保存在: {output_dir}")


if __name__ == '__main__':
    main()


def check_flash_attention_support():
    """检查系统是否支持 FlashAttention 2"""
    try:
        import flash_attn
        flash_version = getattr(flash_attn, '__version__', 'unknown')
        print(f"FlashAttention 已安装，版本: {flash_version}")
        return True
    except ImportError:
        print("警告: FlashAttention 未安装")
        return False


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print('未检测到分布式训练环境，使用单卡训练')
        rank = 0
        world_size = 1
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


class MovieReviewDataset(DynamicPaddingDataset):
    """
    影评数据集（继承自DynamicPaddingDataset）
    """
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        格式化影评样本为训练prompt
        
        覆盖父类方法，使用影评专用格式
        """
        parts = []
        
        # 1. 用户Profile
        if self.use_profile and sample.get('user_profile'):
            profile = sample['user_profile']
            parts.append(f"用户: {profile.get('name', 'Unknown')}")
            if sample.get('task_description'):
                parts.append(f"任务: {sample['task_description']}")
            parts.append("")
        
        # 2. 历史影评（如果启用）
        if self.use_history and sample.get('history'):
            history = sample['history']
            parts.append(f"历史影评记录 ({len(history)}条):")
            
            # 只使用最近的N条历史
            # max_history = 15
            for h in history:
                parts.append(f"  电影《{h['movie']}》: {h['review']}")
            
            # if len(history) > max_history:
            #     parts.append(f"  ...（还有{len(history) - max_history}条更早的评论）")
            parts.append("")
        
        # 3. 当前电影
        movie_name = sample.get('movie_name', '')
        parts.append(f"模仿用户风格为电影《{movie_name}》写一条影评：")
        
        return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description='豆瓣影评模型 - 分布式训练')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       default='config_MovieReview.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history', 'profile_only', 'history_only', 'baseline'],
                       help='消融实验配置')
    
    # 数据相关（可选，覆盖配置文件）
    parser.add_argument('--data_file', type=str, default=None,
                       help='影评数据JSON文件路径（覆盖配置文件）')
    parser.add_argument('--val_ratio', type=float, default=None,
                       help='验证集比例（覆盖配置文件）')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='最大训练轮次（默认50）')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值（默认3）')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='早停阈值（默认0.001）')
    
    # DeepSpeed和其他
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed配置文件路径')
    parser.add_argument('--disable_flash_attn', action='store_true',
                       help='禁用FlashAttention 2')
    
    # 新增：每用户采样参数
    parser.add_argument('--max_samples_per_user', type=int, default=None,
                       help='每个用户最多保留多少个样本（用于减少训练数据量）')
    parser.add_argument('--sample_seed', type=int, default=42,
                       help='采样随机种子（默认：42，保证可复现）')
    
    # 新增：每用户一个样本模式
    parser.add_argument('--one_sample_per_user', action='store_true',
                       help='每个用户只生成1个样本（用前n-1条历史预测第n条，大幅减少训练时间）')
    
    parser.add_argument('--wandb_project', type=str, default='MovieReview',
                       help='Weights & Biases项目名称')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases运行名称')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='本地进程rank')
    parser.add_argument('--prompt_style', type=str, default='simple',
                       choices=['simple', 'detailed'],
                       help='Prompt风格：simple=简洁，detailed=详细')
    
    args = parser.parse_args()
    
    # 加载配置文件
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        local_config = Path(__file__).parent / args.config
        if local_config.exists():
            config_path = str(local_config)
        else:
            config_path = args.config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取消融配置
    ablation_config = config['ablation_configs'][args.ablation_config]
    use_profile = ablation_config.get('use_profile', True)
    use_history = ablation_config.get('use_history', True)
    config_name = ablation_config['name']
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    # 配置W&B
    if args.wandb_project and is_main_process:
        try:
            import wandb
            os.environ['WANDB_PROJECT'] = args.wandb_project
            if args.wandb_run_name:
                os.environ['WANDB_NAME'] = args.wandb_run_name
            print(f"✓ 已启用 W&B 监控: {args.wandb_project}")
        except ImportError:
            print("警告: wandb 未安装")
            args.wandb_project = None
    
    if is_main_process:
        print("=" * 80)
        print("豆瓣影评模型 - 分布式训练")
        print("=" * 80)
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        print(f"消融实验: {config_name}")
        print(f"使用配置:")
        print(f"  Profile: {use_profile}")
        print(f"  History: {use_history}")
        print(f"  Prompt Style: {args.prompt_style}")
        if args.deepspeed:
            print(f"  DeepSpeed: {args.deepspeed}")
        print("=" * 80)
    
    # 检查FlashAttention（所有进程独立检查，避免CUDA broadcast问题）
    use_flash_attn = False
    if not args.disable_flash_attn:
        # 所有进程都检查FlashAttention支持
        try:
            import flash_attn
            use_flash_attn = True
            if is_main_process:
                flash_version = getattr(flash_attn, '__version__', 'unknown')
                print(f"FlashAttention 已安装，版本: {flash_version}")
        except ImportError:
            if is_main_process:
                print("FlashAttention 未安装，使用标准attention")
    
    # 验证GPU是否可用
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] 错误: CUDA 不可用")
        cleanup_distributed()
        return
    
    if is_main_process:
        print(f"CUDA 可用，GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {local_rank} - {torch.cuda.get_device_name(local_rank)}")
        compute_cap = torch.cuda.get_device_capability(local_rank)
        print(f"计算能力: {compute_cap[0]}.{compute_cap[1]}")
        print(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}")
    
    # 加载数据
    if is_main_process:
        print("\n" + "=" * 80)
        print("加载影评数据...")
    
    # 数据路径（优先使用命令行参数，否则使用配置文件）
    data_file = args.data_file if args.data_file else config['data']['train_path']
    if not os.path.isabs(data_file):
        data_file = str(Path(__file__).parent / data_file)
    
    raw_data = load_movie_review_data(data_file)
    all_samples = extract_movie_review_samples(
        raw_data, 
        one_sample_per_user=args.one_sample_per_user,  # 🔥 新增：启用每用户一个样本模式
        debug=is_main_process
    )
    
    if is_main_process:
        print(f"数据文件: {data_file}")
        print(f"提取了 {len(all_samples)} 个样本")
        if args.one_sample_per_user:
            print(f"  ✅ 每用户一个样本模式：用前n-1条历史预测第n条")
    
    # 新增：每用户采样（如果指定了 max_samples_per_user 且未启用 one_sample_per_user）
    if args.max_samples_per_user is not None and not args.one_sample_per_user:
        if is_main_process:
            print(f"\n对每个用户进行采样（每用户最多 {args.max_samples_per_user} 个样本）...")
        all_samples = sample_per_user(
            all_samples,
            max_samples_per_user=args.max_samples_per_user,
            random_seed=args.sample_seed
        )
    
    # 获取数据划分比例
    data_split = config.get('data_split', {})
    train_ratio = data_split.get('train_ratio', 0.7)
    val_ratio_config = data_split.get('val_ratio', 0.15)
    test_ratio = data_split.get('test_ratio', 0.15)
    
    # 如果指定了val_ratio，需要重新计算test_ratio
    if args.val_ratio is not None:
        val_ratio_config = args.val_ratio
        test_ratio = 1.0 - train_ratio - val_ratio_config
    
    # 按时间顺序划分数据集
    train_samples, val_samples, test_samples = split_movie_reviews_by_time(
        all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio_config,
        test_ratio=test_ratio,
        debug=is_main_process
    )
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_dir = config['model']['checkpoint_dir']
        flash_suffix = "flashattn2" if use_flash_attn else "standard"
        output_dir = os.path.join(checkpoint_dir, f"MovieReview_{config_name}_{flash_suffix}")
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
        
        # 保存测试集（用于后续评估）
        test_file = os.path.join(output_dir, 'test_samples.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        print(f"测试集已保存到: {test_file}")
    
    if world_size > 1:
        dist.barrier()
    
    # 加载tokenizer和模型
    model_path = config['model']['path']
    if is_main_process:
        print(f"\n加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model_kwargs = {
        'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        'trust_remote_code': True,
    }
    
    if use_flash_attn:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if is_main_process:
            print(f"✓ 模型已加载 ({'FlashAttention 2' if use_flash_attn else '标准Attention'})")
    except Exception as e:
        if is_main_process:
            print(f"加载失败: {e}")
            print("回退到标准attention...")
        model_kwargs.pop('attn_implementation', None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        use_flash_attn = False
    
    # 启用梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if is_main_process:
            print("✓ 梯度检查点已启用")
    
    model = model.to(local_rank)
    
    # 创建数据集
    train_config = config.get('training', {})
    if is_main_process:
        print("\n创建训练数据集...")
    
    train_dataset = MovieReviewDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=train_config.get('max_length', 4096),
        use_profile=use_profile,
        use_history=use_history,
        use_context=False,  # 影评数据不使用context
        verbose=is_main_process,
        use_detailed_template=False  # 🔥 明确指定使用简单格式（虽然会被 format_prompt 覆盖）
    )
    
    val_dataset = None
    if val_samples:
        val_dataset = MovieReviewDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            max_length=train_config.get('max_length', 4096),
            use_profile=use_profile,
            use_history=use_history,
            use_context=False,
            verbose=False,
            use_detailed_template=False  # 🔥 明确指定使用简单格式
        )
    
    # 🔥 新增：统计所有样本的 token 长度
    if is_main_process:
        print("\n" + "=" * 80)
        print("📊 Token 长度统计（训练集）")
        print("=" * 80)
        
        # 收集所有样本的 token 长度
        token_lengths = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            input_len = len(sample['input_ids'])
            token_lengths.append(input_len)
        
        if token_lengths:
            import numpy as np
            max_length_config = train_config.get('max_length', 4096)
            
            print(f"样本总数: {len(token_lengths)}")
            print(f"配置的 max_length: {max_length_config}")
            print(f"\nToken 长度分布:")
            print(f"  最小长度: {min(token_lengths)} tokens")
            print(f"  最大长度: {max(token_lengths)} tokens")
            print(f"  平均长度: {np.mean(token_lengths):.1f} tokens")
            print(f"  中位数: {np.median(token_lengths):.0f} tokens")
            print(f"\n分位数:")
            print(f"  25%: {np.percentile(token_lengths, 25):.0f} tokens")
            print(f"  50%: {np.percentile(token_lengths, 50):.0f} tokens")
            print(f"  75%: {np.percentile(token_lengths, 75):.0f} tokens")
            print(f"  90%: {np.percentile(token_lengths, 90):.0f} tokens")
            print(f"  95%: {np.percentile(token_lengths, 95):.0f} tokens")
            print(f"  99%: {np.percentile(token_lengths, 99):.0f} tokens")
            
            # 检查是否有样本超过 max_length
            exceed_count = sum(1 for l in token_lengths if l > max_length_config)
            if exceed_count > 0:
                print(f"\n⚠️ 警告: {exceed_count} 个样本 ({exceed_count/len(token_lengths)*100:.1f}%) 超过 max_length={max_length_config}")
                print(f"   这些样本会被截断，可能影响训练效果")
                print(f"   建议:")
                max_needed = max(token_lengths)
                if max_needed <= 8192:
                    print(f"     - 增加 max_length 到 8192")
                elif max_needed <= 16384:
                    print(f"     - 增加 max_length 到 16384")
                    print(f"     - 或启用 CPU checkpointing")
                else:
                    print(f"     - 增加 max_length 到 {max_needed}")
                    print(f"     - 需要使用序列并行（Ulysses）")
            else:
                print(f"\n✅ 所有样本都在 max_length={max_length_config} 范围内")
        
        print("=" * 80)
    
    # 数据整理器
    def collate_fn(examples):
        return dynamic_padding_collate_fn(examples, tokenizer)
    
    # 打印样本示例
    if is_main_process:
        print("\n" + "=" * 80)
        print("📝 训练样本示例（前3个）")
        print("=" * 80)
        
        sample_log_file = os.path.join(output_dir, "training_samples_preview.txt")
        with open(sample_log_file, 'w', encoding='utf-8') as log_file:
            for i in range(min(3, len(train_samples))):
                sample = train_samples[i]
                
                print(f"\n--- 样本 {i+1} ---")
                log_file.write(f"\n{'='*80}\n样本 {i+1}\n{'='*80}\n\n")
                
                # 电影信息
                movie_name = sample.get('movie_name', 'N/A')
                timestamp = sample.get('timestamp', 'N/A')
                print(f"电影: {movie_name}")
                print(f"时间: {timestamp}")
                log_file.write(f"电影: {movie_name}\n")
                log_file.write(f"时间: {timestamp}\n\n")
                
                # 历史影评数量
                history_count = len(sample.get('history', []))
                print(f"历史影评: {history_count}条")
                log_file.write(f"历史影评: {history_count}条\n")
                
                if history_count > 0:
                    log_file.write("最近3条:\n")
                    for h in sample['history'][-3:]:
                        log_file.write(f"  - {h['movie']}: {h['review']}\n")
                log_file.write("\n")
                
                # 目标影评
                target = sample.get('next_question', '')
                print(f"目标影评: {target[:80]}...")
                log_file.write(f"目标影评:\n{target}\n\n")
                
                # 编码信息
                encoded = train_dataset[i]
                input_len = len(encoded['input_ids'])
                valid_labels = (encoded['labels'] != -100).sum().item()
                print(f"编码长度: {input_len} tokens, 有效标签: {valid_labels}")
                log_file.write(f"编码长度: {input_len} tokens\n")
                log_file.write(f"有效标签: {valid_labels} tokens\n")
        
        print(f"\n✓ 样本详情已保存到: {sample_log_file}")
        print("=" * 80)
    
    # 计算训练步数
    batch_size = train_config.get('batch_size', 2)
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 8)
    steps_per_epoch = len(train_dataset) // (world_size * batch_size * gradient_accumulation_steps)
    eval_steps = max(1, steps_per_epoch // 2) if val_dataset else None
    save_steps = train_config.get('save_steps', 500)
    
    # 调整save_steps为eval_steps的整数倍
    if val_dataset and eval_steps and save_steps % eval_steps != 0:
        save_steps = ((save_steps + eval_steps - 1) // eval_steps) * eval_steps
        if is_main_process:
            print(f"调整 save_steps 为 {save_steps}（eval_steps={eval_steps} 的整数倍）")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_config.get('eval_batch_size', 2),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=train_config.get('learning_rate', 1e-5),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 100),
        logging_steps=train_config.get('logging_steps', 10),
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=0.5,
        report_to="wandb" if args.wandb_project else "none",
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        dataloader_num_workers=2,
        save_on_each_node=False,
        logging_first_step=True,
        deepspeed=args.deepspeed,
    )
    
    # 早停回调
    callbacks = []
    if val_dataset:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping)
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # 开始训练
    if is_main_process:
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80)
        print(f"训练样本: {len(train_dataset)}")
        print(f"验证样本: {len(val_dataset) if val_dataset else 0}")
        print(f"测试样本: {len(test_samples)}")
        print(f"每个GPU batch size: {batch_size}")
        print(f"梯度累积: {gradient_accumulation_steps}")
        print(f"有效batch size: {batch_size * gradient_accumulation_steps * world_size}")
        print(f"预计每epoch步数: {steps_per_epoch}")
        print(f"Max Length: {train_config.get('max_length', 4096)}")
        print(f"Learning Rate: {train_config.get('learning_rate', 1e-5)}")
        if args.deepspeed:
            print(f"DeepSpeed: {args.deepspeed}")
        print("=" * 80)
    
    trainer.train()
    
    # 保存模型
    if is_main_process:
        print(f"\n保存模型到 {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        config_info = {
            'data_file': data_file,
            'ablation_config': args.ablation_config,
            'config_name': config_name,
            'use_profile': use_profile,
            'use_history': use_history,
            'flash_attention_2': use_flash_attn,
            'max_length': train_config.get('max_length', 4096),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'prompt_style': args.prompt_style,
        }
        with open(os.path.join(output_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        print("✓ 训练完成！")
        print(f"✓ 模型已保存到: {output_dir}")
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
