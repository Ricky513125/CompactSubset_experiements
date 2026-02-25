"""
分布式训练脚本 - 豆瓣影评模型（FlashAttention 2 + 动态Batch Padding）
自包含版本 - 无需外部依赖

用于训练用户影评风格模拟模型

使用方法:
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_one_per_user_0213 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name one_per_user_0213 \
    --prompt_style simple \
    --one_sample_per_user
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
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)
from typing import List, Dict, Any, Optional, Tuple
import torch.nn as nn
from datetime import datetime

# LoRA 支持
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ================================
# 数据加载函数
# ================================

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
    max_samples_per_user: Optional[int] = None,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    将原始影评数据转换为训练样本格式
    
    两种模式：
    1. max_samples_per_user=None（默认）：每条影评转换为一个样本
       - 用户有100条影评 → 生成100个样本
       - 样本1: [] → r1, 样本2: [r1] → r2, ..., 样本100: [r1..r99] → r100
    
    2. max_samples_per_user=N：每个用户只生成最后N个样本
       - 用户有100条影评，max_samples_per_user=2 → 生成2个样本
       - 样本1: [r1..r98] → r99（用前98条预测第99条）
       - 样本2: [r1..r99] → r100（用前99条预测第100条）
       - **大幅减少训练数据量，同时最大化历史信息利用**
    
    Args:
        raw_data: 原始数据
        max_samples_per_user: 每个用户最多生成的样本数（None表示全部，默认None）
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
            
            if max_samples_per_user is not None and max_samples_per_user > 0:
                # 🔥 新模式：每个用户选择最后N条作为预测目标
                # 使用前 n-N 条作为历史，预测最后 N 条
                if len(reviews) < max_samples_per_user + 1:
                    if debug:
                        print(f"  ⚠️ 跳过该用户（影评数 < {max_samples_per_user + 1}）")
                    continue
                
                # 前 n-N 条作为共享历史
                history_reviews = reviews[:-max_samples_per_user]
                # 最后N条作为预测目标
                last_n_reviews = reviews[-max_samples_per_user:]
                
                # 为最后N条影评分别创建样本，使用累积的历史
                for idx, target_review in enumerate(last_n_reviews):
                    # 历史包括：前 n-N 条 + 当前目标之前的所有最后N条中的影评
                    current_history = history_reviews + last_n_reviews[:idx]
                    
                    sample = {
                        'user_profile': user_profile,
                        'user_hash': user_profile.get('name', 'unknown'),
                        'task_description': task_desc,
                        
                        # 历史影评
                        'history': [
                            {
                                'movie': h.get('continuation_prefix', '').rstrip(': '),
                                'review': h.get('continuation', ''),
                                'timestamp': h.get('timestamp', '')
                            }
                            for h in current_history
                        ],
                        
                        # 当前电影信息
                        'movie_name': target_review.get('continuation_prefix', '').rstrip(': '),
                        'timestamp': target_review.get('timestamp', ''),
                        
                        # 目标：要预测的影评
                        'next_question': target_review.get('continuation', ''),
                        
                        # context保持空列表（兼容现有框架）
                        'context': target_review.get('context', []),
                        
                        # 元数据
                        'total_reviews': len(reviews),
                        'history_count': len(current_history),
                        'target_index': len(reviews) - max_samples_per_user + idx,
                        'raw_review': target_review
                    }
                    
                    all_samples.append(sample)
                
                if debug:
                    print(f"  生成{max_samples_per_user}个样本: {len(history_reviews)}条共享历史 → 预测最后{max_samples_per_user}条")
            
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


# ================================
# 数据集类
# ================================

class MovieReviewDataset(Dataset):
    """
    影评数据集（动态Padding版本）
    """
    def __init__(self, samples, tokenizer, max_length=4096, use_profile=True, use_history=True, verbose=False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_profile = use_profile
        self.use_history = use_history
        self.verbose = verbose
        
        # 截断统计
        self.truncation_stats = {
            'total_samples': 0,
            'truncated_samples': 0,
            'truncated_history': 0,
        }
        self.first_truncation_logged = False
    
    def build_prompt(self, sample: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
        """
        构建影评训练prompt
        
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
        
        # 3. 当前电影（中文提示）
        movie_name = sample.get('movie_name', '')
        parts.append(f"\n预测用户对该电影的评价：")
        parts.append("注意：请直接给出用户对该电影的评价，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
        
        system_content = "\n".join(parts)
        
        messages = [
            {'role': 'system', 'content': system_content}
        ]
        
        # target_answer 用 [ANSWER] 和 [/ANSWER] 包裹 next_question（与训练时保持一致）
        next_question = sample.get('next_question', '')
        target_answer = f"[ANSWER]\n{next_question}\n[/ANSWER]"
        
        return messages, target_answer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 构建消息
        messages, target_answer = self.build_prompt(sample)
        
        # 2. 生成完整文本
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        generation_suffix = "<|im_start|>assistant\n"
        full_prompt = full_prompt.strip() + generation_suffix
        im_end_token = "<|im_end|>"
        full_text = full_prompt + target_answer + im_end_token
        
        # 3. 处理超长文本：删除历史记录
        full_length = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        is_truncated = False
        removed_history = 0
        
        if full_length > self.max_length:
            is_truncated = True
            history = sample.get('history', [])
            
            if history and len(history) > 0:
                reduced_history = history[:]
                while full_length > self.max_length and len(reduced_history) > 0:
                    reduced_history.pop(0)  # 删除最旧的历史记录
                    removed_history += 1
                    
                    # 重建样本
                    temp_sample = sample.copy()
                    temp_sample['history'] = reduced_history
                    messages, target_answer = self.build_prompt(temp_sample)
                    
                    # 重新生成
                    full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    full_prompt = full_prompt.strip() + generation_suffix
                    full_text = full_prompt + target_answer + im_end_token
                    full_length = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        
        # 更新统计
        self.truncation_stats['total_samples'] += 1
        if is_truncated:
            self.truncation_stats['truncated_samples'] += 1
            self.truncation_stats['truncated_history'] += removed_history
            
            if not self.first_truncation_logged and self.verbose:
                self.first_truncation_logged = True
                print(f"\n⚠️  样本#{idx} 超长，删除了 {removed_history} 条历史记录")
                print(f"  调整后长度: {full_length} tokens (max: {self.max_length})\n")
        
        # 4. 编码（不做padding）
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
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        actual_prompt_len = len(prompt_ids)
        
        labels = input_ids.clone()
        safe_prompt_len = min(actual_prompt_len, len(input_ids) - 1)
        labels[:safe_prompt_len] = -100
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
    只padding到batch内最长样本的长度
    """
    # 找到batch中最长的序列长度
    max_length_in_batch = max(ex['input_ids'].shape[0] for ex in examples)
    
    # 5%概率打印batch信息
    if random.random() < 0.05:
        lengths = [ex['input_ids'].shape[0] for ex in examples]
        print(f"[Batch] Lengths: min={min(lengths)}, max={max_length_in_batch}, avg={sum(lengths)/len(lengths):.0f}")
    
    batch = {}
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ex in examples:
        seq_len = ex['input_ids'].shape[0]
        pad_len = max_length_in_batch - seq_len
        
        # Padding
        padded_input_ids.append(
            torch.cat([
                ex['input_ids'],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
        )
        
        padded_attention_mask.append(
            torch.cat([
                ex['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        
        padded_labels.append(
            torch.cat([
                ex['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )
    
    batch['input_ids'] = torch.stack(padded_input_ids)
    batch['attention_mask'] = torch.stack(padded_attention_mask)
    batch['labels'] = torch.stack(padded_labels)
    
    return batch


# ================================
# 分布式训练辅助函数
# ================================

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


# ================================
# 主函数
# ================================

def main():
    parser = argparse.ArgumentParser(description='豆瓣影评模型 - 分布式训练（自包含版本）')
    
    # 配置文件
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history', 'profile_only', 'history_only', 'baseline'],
                       help='消融实验配置')
    
    # 数据相关
    parser.add_argument('--data_file', type=str, default=None, help='影评数据JSON文件路径（覆盖配置文件）')
    parser.add_argument('--val_ratio', type=float, default=None, help='验证集比例（覆盖配置文件）')
    parser.add_argument('--max_samples_per_user', type=int, default=None, 
                       help='每个用户最多生成的样本数（None表示全部，默认从配置文件读取）')
    parser.add_argument('--one_sample_per_user', action='store_true',
                       help='[已废弃] 使用 --max_samples_per_user=2 替代')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None, help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=50, help='最大训练轮次')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001, help='早停阈值')
    
    # DeepSpeed
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed配置文件路径')
    parser.add_argument('--disable_flash_attn', action='store_true', help='禁用FlashAttention 2')
    
    # W&B
    parser.add_argument('--wandb_project', type=str, default='MovieReview', help='Weights & Biases项目名称')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases运行名称')
    
    # 其他
    parser.add_argument('--local_rank', type=int, default=-1, help='本地进程rank')
    parser.add_argument('--prompt_style', type=str, default='simple', choices=['simple', 'detailed'],
                       help='Prompt风格（本版本只支持simple）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
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
        print("豆瓣影评模型 - 分布式训练（自包含版本）")
        print("=" * 80)
        print(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")
        print(f"消融实验: {config_name}")
        print(f"使用配置: Profile={use_profile}, History={use_history}")
        # 处理 max_samples_per_user
        max_samples = args.max_samples_per_user
        if max_samples is None:
            # 从配置文件读取
            max_samples = config.get('training', {}).get('max_samples_per_user', None)
        # 兼容旧的 one_sample_per_user 参数
        if args.one_sample_per_user:
            max_samples = 2
            if is_main_process:
                print(f"⚠️  --one_sample_per_user 已废弃，请使用 --max_samples_per_user=2")
        
        if max_samples is not None:
            print(f"采样模式: 每用户最多 {max_samples} 个样本")
        else:
            print(f"采样模式: 每条影评一个样本（全部）")
        if args.deepspeed:
            print(f"DeepSpeed: {args.deepspeed}")
        print("=" * 80)
    
    # 检查FlashAttention
    use_flash_attn = False
    if not args.disable_flash_attn:
        try:
            import flash_attn
            use_flash_attn = True
            if is_main_process:
                print(f"FlashAttention 已安装")
        except ImportError:
            if is_main_process:
                print("FlashAttention 未安装，使用标准attention")
    
    # 验证GPU
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] 错误: CUDA 不可用")
        cleanup_distributed()
        return
    
    if is_main_process:
        print(f"CUDA 可用，GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {local_rank} - {torch.cuda.get_device_name(local_rank)}")
    
    # 加载数据
    if is_main_process:
        print("\n" + "=" * 80)
        print("加载影评数据...")
    
    data_file = args.data_file if args.data_file else config['data']['train_path']
    raw_data = load_movie_review_data(data_file)
    
    # 确定 max_samples_per_user
    max_samples = args.max_samples_per_user
    if max_samples is None:
        # 从配置文件读取
        max_samples = config.get('training', {}).get('max_samples_per_user', None)
    # 兼容旧的 one_sample_per_user 参数
    if args.one_sample_per_user:
        max_samples = 2
        if is_main_process:
            print(f"⚠️  --one_sample_per_user 已废弃，请使用 --max_samples_per_user=2")
    
    all_samples = extract_movie_review_samples(
        raw_data, 
        max_samples_per_user=max_samples,
        debug=is_main_process
    )
    
    if is_main_process:
        print(f"数据文件: {data_file}")
        print(f"提取了 {len(all_samples)} 个样本")
    
    # 获取数据划分比例
    data_split = config.get('data_split', {})
    train_ratio = data_split.get('train_ratio', 0.7)
    val_ratio_config = data_split.get('val_ratio', 0.15)
    test_ratio = data_split.get('test_ratio', 0.15)
    
    if args.val_ratio is not None:
        val_ratio_config = args.val_ratio
        test_ratio = 1.0 - train_ratio - val_ratio_config
    
    # 按时间划分数据集
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
        output_dir = os.path.join(checkpoint_dir, f"MovieReview_{config_name}")
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
        
        # 保存测试集
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
    
    # 获取训练配置和模型配置
    train_config = config.get('training', {})
    model_config = config.get('model', {})
    
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
            if use_flash_attn:
                print("✓ 模型已加载（FlashAttention 2）")
            else:
                print("✓ 模型已加载（标准Attention）")
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
    
    # LoRA 配置（如果配置文件中启用）
    use_lora = model_config.get('use_lora', False)
    if use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 已启用但 peft 库未安装。请运行: pip install peft")
        
        lora_config_dict = model_config.get('lora_config', {})
        if is_main_process:
            print("\n" + "="*80)
            print("⚡ LoRA 配置:")
            print(f"   - rank (r): {lora_config_dict.get('r', 64)}")
            print(f"   - alpha: {lora_config_dict.get('lora_alpha', 128)}")
            print(f"   - dropout: {lora_config_dict.get('lora_dropout', 0.05)}")
            print(f"   - target modules: {lora_config_dict.get('target_modules', [])}")
            print("="*80 + "\n")
        
        # 创建 LoRA 配置
        lora_config = LoraConfig(
            r=lora_config_dict.get('r', 64),
            lora_alpha=lora_config_dict.get('lora_alpha', 128),
            lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
            target_modules=lora_config_dict.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias=lora_config_dict.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        
        if is_main_process:
            print("✓ LoRA 已应用")
            model.print_trainable_parameters()
    
    model = model.to(local_rank)
    
    # 创建数据集
    if is_main_process:
        print("\n创建训练数据集...")
    
    train_dataset = MovieReviewDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=train_config.get('max_length', 4096),
        use_profile=use_profile,
        use_history=use_history,
        verbose=is_main_process
    )
    
    val_dataset = None
    if val_samples:
        val_dataset = MovieReviewDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            max_length=train_config.get('max_length', 4096),
            use_profile=use_profile,
            use_history=use_history,
            verbose=False
        )
    
    # Token长度统计
    if is_main_process:
        print("\n" + "=" * 80)
        print("📊 Token 长度统计（训练集）")
        print("=" * 80)
        
        token_lengths = []
        for i in range(min(len(train_dataset), 1000)):  # 只统计前1000个
            sample = train_dataset[i]
            token_lengths.append(len(sample['input_ids']))
        
        if token_lengths:
            import numpy as np
            max_length_config = train_config.get('max_length', 4096)
            
            print(f"样本总数: {len(train_dataset)}")
            print(f"配置的 max_length: {max_length_config}")
            print(f"\nToken 长度分布（前1000个样本）:")
            print(f"  最小: {min(token_lengths)}, 最大: {max(token_lengths)}, 平均: {np.mean(token_lengths):.1f}")
            print(f"  中位数: {np.median(token_lengths):.0f}")
            print(f"  75%: {np.percentile(token_lengths, 75):.0f}")
            print(f"  95%: {np.percentile(token_lengths, 95):.0f}")
            
            exceed_count = sum(1 for l in token_lengths if l > max_length_config)
            if exceed_count > 0:
                print(f"\n⚠️ {exceed_count} 个样本超过 max_length")
            else:
                print(f"\n✅ 所有样本都在 max_length 范围内")
        
        print("=" * 80)
    
    # 数据整理器
    def collate_fn(examples):
        return dynamic_padding_collate_fn(examples, tokenizer)
    
    # 计算训练步数
    batch_size = train_config.get('batch_size', 2)
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 8)
    steps_per_epoch = len(train_dataset) // (world_size * batch_size * gradient_accumulation_steps)
    eval_steps = max(1, steps_per_epoch // 2) if val_dataset else None
    save_steps = train_config.get('save_steps', 500)
    
    # 调整save_steps
    if val_dataset and eval_steps and save_steps % eval_steps != 0:
        save_steps = ((save_steps + eval_steps - 1) // eval_steps) * eval_steps
        if is_main_process:
            print(f"调整 save_steps 为 {save_steps}")
    
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
        dataloader_num_workers=train_config.get('dataloader_num_workers', 4),
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
    
    # 创建自定义Trainer（带权重处理）
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 保存 tokenizer 引用（用于损失权重计算）
            self.tokenizer = tokenizer
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """计算损失（对 [ANSWER] 和 [/ANSWER] token 增加权重）"""
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            elif labels is not None:
                valid_labels_count = (labels != -100).sum().item()
                
                if valid_labels_count == 0:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} 没有有效的labels")
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # 创建损失权重：对 [ANSWER] 和 [/ANSWER] token 增加权重
                    # 获取 tokenizer 中的 [ANSWER] 和 [/ANSWER] 的所有 token IDs
                    answer_start_token_ids = set()
                    answer_end_token_ids = set()
                    
                    try:
                        # 尝试获取 [ANSWER] 和 [/ANSWER] 的所有 token IDs
                        if hasattr(self.tokenizer, 'encode'):
                            # 编码标签（可能被编码为多个 token）
                            answer_start_tokens = self.tokenizer.encode("[ANSWER]", add_special_tokens=False)
                            answer_end_tokens = self.tokenizer.encode("[/ANSWER]", add_special_tokens=False)
                            
                            # 保存所有相关的 token IDs（不仅仅是第一个）
                            if answer_start_tokens:
                                answer_start_token_ids = set(answer_start_tokens)
                            if answer_end_tokens:
                                answer_end_token_ids = set(answer_end_tokens)
                    except:
                        pass
                    
                    # 创建权重张量（默认权重为 1.0）
                    batch_size, seq_len = shift_labels.shape
                    loss_weights = torch.ones_like(shift_labels, dtype=torch.float32)
                    
                    # 对 [ANSWER] 和 [/ANSWER] 的所有 token 增加权重（权重设为 3.0）
                    if answer_start_token_ids:
                        for token_id in answer_start_token_ids:
                            loss_weights[shift_labels == token_id] = 3.0
                    if answer_end_token_ids:
                        for token_id in answer_end_token_ids:
                            loss_weights[shift_labels == token_id] = 3.0
                    
                    # 使用加权损失
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                    per_token_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # 应用权重并计算平均损失
                    per_token_loss = per_token_loss.view(batch_size, seq_len)
                    valid_mask = (shift_labels != -100)
                    weighted_loss = (per_token_loss * loss_weights * valid_mask.float()).sum()
                    valid_count = (valid_mask.float() * loss_weights).sum()
                    
                    if valid_count > 0:
                        loss = weighted_loss / valid_count
                    else:
                        loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
            else:
                loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
            
            # 检查损失值
            if loss is not None and torch.is_tensor(loss):
                if loss.dim() > 0:
                    loss = loss.mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} loss为nan/inf")
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                elif loss.item() > 1e6:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} loss过大")
                    loss = torch.clamp(loss, max=100.0)
            
            if return_outputs:
                return loss, outputs
            return loss
    
    # 创建Trainer
    trainer = CustomTrainer(
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
        print(f"有效batch size: {batch_size * gradient_accumulation_steps * world_size}")
        print(f"预计每epoch步数: {steps_per_epoch}")
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
            'max_samples_per_user': max_samples,
            'one_sample_per_user': args.one_sample_per_user,  # 保留以兼容
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
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
