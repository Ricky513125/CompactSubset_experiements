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
    one_sample_per_user: bool = False,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    将原始影评数据转换为训练样本格式
    
    两种模式：
    1. one_sample_per_user=False（默认）：每条影评转换为一个样本
       - 用户有100条影评 → 生成100个样本
       - 样本1: [] → r1, 样本2: [r1] → r2, ..., 样本100: [r1..r99] → r100
    
    2. one_sample_per_user=True：每个用户只生成2个样本（最后2条）
       - 用户有100条影评 → 生成2个样本
       - 样本1: [r1..r98] → r99（用前98条预测第99条）
       - 样本2: [r1..r99] → r100（用前99条预测第100条）
       - **大幅减少训练数据量，同时最大化历史信息利用**
    
    Args:
        raw_data: 原始数据
        one_sample_per_user: 是否每个用户只生成最后2个样本（默认False）
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
                # 🔥 新模式：每个用户选择最后2条作为预测目标
                # 使用前 n-2 条作为历史，预测最后 2 条
                if len(reviews) < 3:
                    if debug:
                        print(f"  ⚠️ 跳过该用户（影评数 < 3）")
                    continue
                
                # 前 n-2 条作为共享历史
                history_reviews = reviews[:-2]
                # 最后2条作为预测目标
                last_two_reviews = reviews[-2:]
                
                # 为最后2条影评分别创建样本，但都使用相同的历史
                for idx, target_review in enumerate(last_two_reviews):
                    # 对于第二个样本（reviews[-1]），可以额外包含reviews[-2]作为历史
                    if idx == 0:
                        # 第一个样本：只用前 n-2 条作为历史
                        current_history = history_reviews
                    else:
                        # 第二个样本：用前 n-2 条 + reviews[-2] 作为历史
                        current_history = history_reviews + [last_two_reviews[0]]
                    
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
                        'target_index': len(reviews) - 2 + idx,  # 倒数第2个或最后1个
                        'raw_review': target_review
                    }
                    
                    all_samples.append(sample)
                
                if debug:
                    print(f"  生成2个样本: {len(history_reviews)}条共享历史 → 预测最后2条")
            
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
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None, help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=50, help='最大训练轮次')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001, help='早停阈值')
    
    # DeepSpeed
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed配置文件路径')
    parser.add_argument('--disable_flash_attn', action='store_true', help='禁用FlashAttention 2')
    
    # 采样参数
    parser.add_argument('--one_sample_per_user', action='store_true',
                       help='每个用户只生成1个样本（用前n-1条历史预测第n条）')
    
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
        if args.one_sample_per_user:
            print(f"采样模式: 每用户一个样本")
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
    all_samples = extract_movie_review_samples(
        raw_data, 
        one_sample_per_user=args.one_sample_per_user,
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
    
    # 📊 数据长度分析（在加载模型之前）
    train_config = config.get('training', {})
    if is_main_process:
        print("\n" + "=" * 80)
        print("📊 分析训练数据长度分布...")
        print("=" * 80)
        
        try:
            # 采样分析（不超过500个样本）
            sample_size = min(500, len(all_samples))
            analysis_samples = random.sample(all_samples, sample_size) if len(all_samples) > sample_size else all_samples
            
            lengths = []
            failed_count = 0
            for sample in analysis_samples:
                try:
                    # 构建完整的prompt
                    messages, target_answer = build_movie_review_prompt(
                        sample=sample,
                        use_profile=use_profile,
                        use_history=use_history
                    )
                    
                    # 转换为文本
                    full_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    ) + target_answer
                    
                    # 编码获取长度
                    token_ids = tokenizer.encode(full_text, add_special_tokens=True)
                    lengths.append(len(token_ids))
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:  # 只打印前3个错误
                        print(f"  样本分析失败: {type(e).__name__}: {str(e)[:100]}")
                    continue
            
            if lengths:
                import numpy as np
                lengths_array = np.array(lengths)
                
                max_length = int(np.max(lengths_array))
                min_length = int(np.min(lengths_array))
                mean_length = float(np.mean(lengths_array))
                median_length = float(np.median(lengths_array))
                percentile_90 = float(np.percentile(lengths_array, 90))
                percentile_95 = float(np.percentile(lengths_array, 95))
                percentile_99 = float(np.percentile(lengths_array, 99))
                
                print(f"分析了 {len(lengths)}/{len(all_samples)} 个样本:")
                print(f"  最小长度: {min_length}")
                print(f"  最大长度: {max_length}")
                print(f"  平均长度: {mean_length:.0f}")
                print(f"  中位数长度: {median_length:.0f}")
                print(f"  90分位数长度: {percentile_90:.0f}")
                print(f"  95分位数长度: {percentile_95:.0f}")
                print(f"  99分位数长度: {percentile_99:.0f}")
                
                # 与配置的max_length对比
                configured_max_length = train_config.get('max_length', 4096)
                print(f"\n配置的 max_length: {configured_max_length}")
                
                exceeds_count = np.sum(lengths_array > configured_max_length)
                print(f"超过 max_length 的样本数: {exceeds_count} ({exceeds_count/len(lengths)*100:.1f}%)")
                
                # 给出建议
                print(f"\n建议:")
                if percentile_95 > configured_max_length:
                    print(f"  警告: 95%的数据超过配置的max_length，可能导致大量截断")
                    print(f"  建议调整 max_length 至少到 {int(percentile_95)}")
                elif percentile_95 < configured_max_length * 0.7:
                    print(f"  提示: 95%的数据长度远小于max_length，可以考虑降低以节省显存")
                else:
                    print(f"  ✓ max_length 设置合理")
                print("=" * 80 + "\n")
            else:
                print(f"警告: 无法分析样本长度 (成功: 0/{sample_size}, 失败: {failed_count})")
                print("=" * 80 + "\n")
        
        except Exception as e:
            print(f"数据长度分析失败: {e}")
            print("=" * 80 + "\n")
    
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
            print(f"✓ 模型已加载")
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
            'one_sample_per_user': args.one_sample_per_user,
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
