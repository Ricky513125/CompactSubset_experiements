
import json
import argparse
import os
import sys
from pathlib import Path
import random
import torch
import torch.distributed as dist
# 统一使用 data_loader_more_data（包含 [ANSWER] 标签，与推理脚本一致）
from data_loader_more_data import load_train_data, extract_training_samples, get_user_only_history
from train_with_dynamic_padding import DynamicPaddingDataset, dynamic_padding_collate_fn, split_train_val, add_history_to_samples, CustomTrainerWithAnswerWeight
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)
from typing import List, Dict, Any, Optional
import torch.nn as nn


def sample_per_user(samples: List[Dict], max_samples_per_user: Optional[int], seed: int = 42) -> List[Dict]:
    """
    对每个用户的样本进行采样
    
    Args:
        samples: 训练样本列表
        max_samples_per_user: 每个用户最多采样的样本数，None表示不采样
        seed: 随机种子
    
    Returns:
        采样后的样本列表
    """
    if max_samples_per_user is None:
        return samples
    
    # 按用户分组
    user_samples = {}
    for sample in samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    # 对每个用户采样
    random.seed(seed)
    sampled_samples = []
    for user_hash, user_sample_list in user_samples.items():
        if len(user_sample_list) <= max_samples_per_user:
            sampled_samples.extend(user_sample_list)
        else:
            sampled = random.sample(user_sample_list, max_samples_per_user)
            sampled_samples.extend(sampled)
    
    return sampled_samples


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
        print('警告: 未检测到分布式训练环境变量，使用单卡训练')
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


def main():
    parser = argparse.ArgumentParser(description='分布式消融实验训练（FlashAttention 2 + 动态Padding）- REALTALK')
    parser.add_argument('--config', type=str,
                       default='config_REALTALK.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--max_epochs', type=int, default=3,
                       help='最大训练轮次（默认：3）')
    parser.add_argument('--early_stopping_patience', type=int, default=2,
                       help='早停耐心值（默认：2）')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='早停阈值（默认：0.001）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='本地进程rank（由 torch.distributed.launch 自动设置）')
    parser.add_argument('--wandb_project', type=str, default='Qwen3-REALTALK',
                       help='Weights & Biases项目名称（默认：Qwen3-REALTALK）')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases运行名称（默认：自动生成）')
    parser.add_argument('--disable_flash_attn', action='store_true',
                       help='禁用FlashAttention 2，使用标准attention')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed配置文件路径（可选）')
    
    # 新增：Prompt 模板控制参数
    parser.add_argument('--prompt_style', type=str, default='simple',
                       choices=['simple', 'detailed', 'lovink'],
                       help='Prompt 风格：simple=简洁标签格式（默认），detailed=详细模板，lovink=Lovink风格')
    parser.add_argument('--template_filename', type=str, default=None,
                       help='指定模板文件名（仅当 prompt_style=detailed 时生效）')
    
    # 新增：用户采样参数
    parser.add_argument('--max_samples_per_user', type=int, default=None,
                       help='每个用户最多采样的样本数（None表示使用所有样本）')
    parser.add_argument('--sample_seed', type=int, default=42,
                       help='采样随机种子（默认：42）')
    
    args = parser.parse_args()
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 只在主进程打印信息
    is_main_process = (rank == 0)
    
    # 配置 Weights & Biases (只在主进程)
    if args.wandb_project:
        try:
            import wandb
            os.environ['WANDB_PROJECT'] = args.wandb_project
            if args.wandb_run_name:
                os.environ['WANDB_NAME'] = args.wandb_run_name
            if is_main_process:
                print(f"✓ 已启用 Weights & Biases 监控")
        except ImportError:
            if is_main_process:
                print("警告: wandb 未安装")
            args.wandb_project = None
    
    if is_main_process:
        print(f"=" * 80)
        print(f"分布式训练设置（FlashAttention 2 + 动态Padding）:")
        print(f"  World Size (总进程数): {world_size}")
        print(f"  Rank (进程ID): {rank}")
        print(f"  Local Rank (本地GPU ID): {local_rank}")
        print(f"  使用 {world_size} 张GPU进行并行训练")
        print(f"  优化策略: FlashAttention 2 + 动态Batch Padding")
        if args.deepspeed:
            print(f"  DeepSpeed配置: {args.deepspeed}")
        print(f"=" * 80)
    
    # 检查 FlashAttention 支持
    use_flash_attn = False
    if not args.disable_flash_attn and is_main_process:
        use_flash_attn = check_flash_attention_support()
    
    # 广播 use_flash_attn 到所有进程
    if world_size > 1:
        use_flash_attn_tensor = torch.tensor([use_flash_attn], dtype=torch.bool, device=f'cuda:{local_rank}')
        dist.broadcast(use_flash_attn_tensor, src=0)
        use_flash_attn = use_flash_attn_tensor.item()
    
    # 验证GPU是否可用
    if torch.cuda.is_available():
        if is_main_process:
            print(f"CUDA 可用，总GPU数量: {torch.cuda.device_count()}")
            print(f"当前进程使用 GPU: {local_rank}")
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
            print(f"GPU 名称: {gpu_name}")
            print(f"GPU 总内存: {gpu_memory:.2f} GB")
            
            # 检查GPU计算能力
            compute_capability = torch.cuda.get_device_capability(local_rank)
            print(f"GPU 计算能力: {compute_capability[0]}.{compute_capability[1]}")
            if compute_capability[0] >= 8:  # A100/H100
                print("✓ GPU支持FlashAttention 2优化")
            else:
                print("GPU计算能力较低，FlashAttention 2性能可能受限")
    else:
        print("错误: CUDA 不可用")
        cleanup_distributed()
        return
    
    # 加载配置（优先使用当前目录，支持绝对路径）
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # 优先查找当前目录
        local_config = Path(__file__).parent / args.config
        if local_config.exists():
            config_path = str(local_config)
        else:
            # 回退到父目录（向后兼容）
            config_path = os.path.join(Path(__file__).parent.parent, args.config)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取消融配置
    ablation_config = config['ablation_configs'][args.ablation_config]
    use_profile = ablation_config.get('use_profile', True)
    use_history = ablation_config.get('use_history', True)
    use_context = ablation_config.get('use_context', True)
    config_name = ablation_config['name']
    
    if is_main_process:
        print("=" * 80)
        print(f"消融实验（FlashAttn2 + 动态Padding）: {config_name}")
        print(f"使用配置: profile={use_profile}, history={use_history}, context={use_context}")
        print(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}")
        print("=" * 80)
    
    # 加载训练数据
    if is_main_process:
        print("加载训练数据...")
    train_path = config['data']['train_path']
    train_data = load_train_data(train_path)
    
    if not train_data:
        print(f"错误: 无法加载训练数据")
        cleanup_distributed()
        return
    
    # 提取训练样本
    all_samples = extract_training_samples(train_data, debug=is_main_process)
    if is_main_process:
        print(f"提取了 {len(all_samples)} 个训练样本")
    
    # 用户采样（如果指定）
    if args.max_samples_per_user is not None:
        if is_main_process:
            print(f"对每个用户采样最多 {args.max_samples_per_user} 个样本（种子={args.sample_seed}）...")
        all_samples = sample_per_user(all_samples, args.max_samples_per_user, args.sample_seed)
        if is_main_process:
            print(f"采样后剩余 {len(all_samples)} 个样本")
    
    # 添加历史信息
    if use_history:
        if is_main_process:
            print("添加历史信息...")
        all_samples = add_history_to_samples(all_samples, all_samples)
    
    # 划分训练集和验证集
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
    if is_main_process:
        print(f"训练集: {len(train_samples)} 个样本")
        print(f"验证集: {len(val_samples)} 个样本")
        print(f"每个GPU实际处理约 {len(train_samples) // world_size} 个训练样本")
    
    # 获取模型配置
    model_config = config['model']
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_dir = model_config['checkpoint_dir']
        dataset_name = os.path.basename(os.path.dirname(train_path))
        flash_suffix = "flashattn2" if use_flash_attn else "standard"
        output_dir = os.path.join(checkpoint_dir, f"{dataset_name}_ablation_{config_name}_{flash_suffix}_dynamic_distributed")
    
    # 只在主进程创建目录
    if is_main_process:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
        except (OSError, IOError) as e:
            print(f"警告: 无法创建输出目录: {e}")
    
    # 等待主进程创建完目录
    if world_size > 1:
        dist.barrier()
    
    # 加载模型和tokenizer
    model_path = model_config['path']
    if is_main_process:
        print(f"加载模型: {model_path}")
        if use_flash_attn:
            print("  使用 FlashAttention 2 实现...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_config = config.get('training', {})
    # 加载模型到指定GPU（使用FlashAttention 2）
    model_kwargs = {
        'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        'trust_remote_code': True,
    }
    
    # 如果支持且未禁用，则使用FlashAttention 2
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
            print(f"加载FlashAttention 2失败: {e}")
            print("   回退到标准attention...")
        # 回退到标准attention
        model_kwargs.pop('attn_implementation', None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        use_flash_attn = False
    
    # 启用梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if is_main_process:
            print("✓ 梯度检查点已启用")
    
    # 将模型移到对应的GPU
    model = model.to(local_rank)
    
    # 创建数据集（使用动态Padding版本）
    if is_main_process:
        print("创建训练数据集（动态Padding模式）...")
    
    # ✅ 根据命令行参数决定使用哪种 prompt 风格
    use_detailed_template = (args.prompt_style != 'simple')
    template_filename = args.template_filename if args.prompt_style == 'detailed' else None
    
    if is_main_process:
        print(f"Prompt 风格: {args.prompt_style}")
        if args.prompt_style == 'simple':
            print("   使用简洁标签格式（[USER_PROFILE] [DIM_XXX=score] ...）")
        elif args.prompt_style == 'detailed':
            if template_filename:
                print(f"   使用详细模板: {template_filename} (标准 {{VAR_NAME}} 格式)")
            else:
                print("   使用详细模板（默认顺序查找）")
        elif args.prompt_style == 'lovink':
            print("   使用 Lovink 风格模板")
    
    train_dataset = DynamicPaddingDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=train_config.get('max_length', 4096),
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context,
        verbose=is_main_process,  # 只在主进程输出详细日志
        use_detailed_template=use_detailed_template,
        template_filename=template_filename
    )
    
    val_dataset = None
    if val_samples:
        if is_main_process:
            print("创建验证数据集（动态Padding模式）...")
        val_dataset = DynamicPaddingDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            max_length=train_config.get('max_length', 4096),
            use_profile=use_profile,
            use_history=use_history,
            use_context=use_context,
            use_detailed_template=use_detailed_template,
            template_filename=template_filename
        )
    
    # 数据整理器（使用动态Padding版本）
    def collate_fn(examples):
        return dynamic_padding_collate_fn(examples, tokenizer)
    
    # 打印第一个样本的输入输出示例（用于调试）
    if is_main_process and len(train_samples) > 0:
        print("\n" + "=" * 80)
        print("📋 第一个训练样本示例")
        print("=" * 80)
        first_sample = train_samples[0]
        first_encoded = train_dataset[0]
        
        print(f"User Hash: {first_sample.get('user_hash', 'N/A')}")
        print(f"Context 轮数: {len(first_sample.get('context', []))}")
        print(f"Target 长度: {len(first_sample.get('next_question', ''))} 字符")
        print(f"\n编码信息:")
        print(f"  Input length: {len(first_encoded['input_ids'])} tokens")
        print(f"  Valid labels: {(first_encoded['labels'] != -100).sum().item()} tokens")
        print(f"  训练比例: {(first_encoded['labels'] != -100).sum().item() / len(first_encoded['labels']):.2%}")
        
        # 解码显示输入和输出
        print(f"\n输入文本（前500 tokens）:")
        input_text = tokenizer.decode(first_encoded['input_ids'][:500], skip_special_tokens=False)
        print(f"  {input_text[:300]}...")
        
        valid_label_indices = (first_encoded['labels'] != -100).nonzero(as_tuple=True)[0]
        if len(valid_label_indices) > 0:
            valid_labels = first_encoded['labels'][valid_label_indices]
            print(f"\n目标输出（模型要生成的部分，前200 tokens）:")
            target_text = tokenizer.decode(valid_labels[:200], skip_special_tokens=False)
            print(f"  {target_text[:300]}...")
        
        print("=" * 80 + "\n")
    
    # 计算训练步数
    steps_per_epoch = len(train_dataset) // (world_size * train_config.get('batch_size', 2) * train_config.get('gradient_accumulation_steps', 8))
    eval_steps_value = max(1, steps_per_epoch // 2) if val_dataset else None
    save_steps_value = train_config.get('save_steps', 500)
    
    if val_dataset and eval_steps_value and save_steps_value % eval_steps_value != 0:
        save_steps_value = ((save_steps_value + eval_steps_value - 1) // eval_steps_value) * eval_steps_value
        if is_main_process:
            print(f"调整 save_steps 为 {save_steps_value}（eval_steps={eval_steps_value} 的整数倍）")
    
    # 训练参数（分布式 + FlashAttention 2 + 动态Padding）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=train_config.get('batch_size', 2),
        per_device_eval_batch_size=train_config.get('eval_batch_size', 2),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 8),
        learning_rate=train_config.get('learning_rate', 1e-5),
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
        bf16=True,  # FlashAttention 2 与 BF16 配合效果更好
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=0.5,
        report_to="wandb" if args.wandb_project else "none",
        # 分布式训练关键参数
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        dataloader_num_workers=2,
        save_on_each_node=False,
        logging_first_step=True,
        # DeepSpeed配置（可选）
        deepspeed=args.deepspeed,
    )
    
    # 创建早停回调
    callbacks = []
    if val_dataset:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping)
    
    # 创建 Trainer（使用统一的 CustomTrainerWithAnswerWeight）
    trainer = CustomTrainerWithAnswerWeight(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  # 使用动态padding的collate_fn
        processing_class=tokenizer,
        callbacks=callbacks,
        tokenizer=tokenizer,
        is_main_process=is_main_process,
        rank=rank,
        debug_steps=3,
    )
    
    # 开始训练
    if is_main_process:
        print("=" * 80)
        print("开始分布式训练（FlashAttention 2 + 动态Padding）")
        print("=" * 80)
        print(f"总样本数: {len(train_dataset)}")
        print(f"每个GPU处理约: {len(train_dataset) // world_size} 个样本")
        effective_batch = train_config.get('batch_size', 2) * train_config.get('gradient_accumulation_steps', 8) * world_size
        print(f"有效 batch size: {effective_batch}")
        print(f"预计每个epoch步数: {steps_per_epoch}")
        print(f"Max Length: {train_config.get('max_length', 4096)} (动态padding)")
        print(f"Attention: {'FlashAttention 2' if use_flash_attn else '标准Attention'}")
        if args.wandb_project:
            print(f"W&B 监控: 项目={args.wandb_project}, 运行={args.wandb_run_name or 'auto'}")
        
        # 输出初始截断统计（训练前）
        if hasattr(train_dataset, 'get_truncation_stats'):
            stats = train_dataset.get_truncation_stats()
            print(f"\n数据预处理截断统计:")
            print(f"  已处理样本: {stats['total_samples']}")
            print(f"  被截断样本: {stats['truncated_samples']}")
            if stats['total_samples'] > 0:
                print(f"  截断率: {stats['truncation_rate']:.2%}")
                if stats['truncated_samples'] > 0:
                    print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
        
        print("=" * 80)
    
    trainer.train()
    
    # 保存最终模型（只在主进程保存）
    if is_main_process:
        print(f"保存最终模型到 {output_dir}")
        try:
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            print("✓ 模型保存成功")
            
            # 保存训练配置信息
            config_info = {
                'flash_attention_2': use_flash_attn,
                'dynamic_padding': True,
                'gradient_checkpointing': True,
                'ablation_config': args.ablation_config,
                'config_name': config_name
            }
            with open(os.path.join(output_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            print("✓ 训练配置已保存")
            
        except Exception as e:
            print(f"警告: 保存模型时出错: {e}")
        
        # 输出截断统计
        if hasattr(train_dataset, 'get_truncation_stats'):
            stats = train_dataset.get_truncation_stats()
            print("\n" + "="*80)
            print("训练数据截断统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  被截断样本数: {stats['truncated_samples']}")
            print(f"  截断率: {stats['truncation_rate']:.2%}")
            if stats['truncated_samples'] > 0:
                print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
            print("="*80)
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    if is_main_process:
        print(f"\n 训练完成！模型保存在: {output_dir}")
        if use_flash_attn:
            print(" 使用了 FlashAttention 2 加速训练")
    
    # 清理分布式环境
    cleanup_distributed()


if __name__ == '__main__':
    main()
