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

# 导入影评数据加载器
from data_loader_movie_review import (
    load_movie_review_data, 
    extract_movie_review_samples,
    split_movie_reviews_by_time,
    format_movie_review_prompt
)

# 复用动态Padding数据集
from train_with_dynamic_padding_Lovink import DynamicPaddingDataset, dynamic_padding_collate_fn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)
from typing import List, Dict, Any, Optional
import torch.nn as nn


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
    all_samples = extract_movie_review_samples(raw_data, debug=is_main_process)
    
    if is_main_process:
        print(f"数据文件: {data_file}")
        print(f"提取了 {len(all_samples)} 个样本")
    
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
            use_context=False,
            verbose=False
        )
    
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
