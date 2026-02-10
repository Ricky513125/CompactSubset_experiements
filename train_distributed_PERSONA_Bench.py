"""
分布式训练脚本（FlashAttention 2 + 动态Batch Padding优化版）

关键优化：
1. FlashAttention 2：更快的注意力机制，显存效率更高
2. 动态Padding：不再将batch内所有样本padding到固定max_length
3. 梯度检查点：降低显存占用
4. 分布式训练：支持多GPU并行

FlashAttention 2 优势：
- 速度提升 2-4x（相比标准attention）
- 显存占用降低 10-20%
- 支持更长的序列长度
- 完全保持数学等价性

环境要求：
- torch >= 2.0.0
- flash-attn >= 2.0.0 (需要手动安装: pip install flash-attn --no-build-isolation)
- CUDA >= 11.6
- GPU: A100/H100 等支持 FlashAttention 的显卡

使用方法：
# 8卡训练
torchrun --nproc_per_node=8 version2_flash_attn/train_distributed_flashattn2.py \
    --config config_realpersonachat.json \
    --ablation_config profile_and_context \
    --output_dir outputs/0130_RealPersonaChat_profile_and_context_flashattn2_8gpu
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

# 注释掉父目录路径，统一使用当前目录（prompt_improvement/Lovink/）下的文件
# sys.path.insert(0, str(Path(__file__).parent.parent))
# from data_loader import load_train_data, extract_training_samples, get_user_only_history # 旧版本 复杂的训练prompt 
from data_loader_more_data import load_train_data, extract_training_samples, get_user_only_history # 新版本 简短的训练prompt
from train_with_dynamic_padding_Lovink import DynamicPaddingDataset, dynamic_padding_collate_fn, split_train_val, add_history_to_samples
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
    parser = argparse.ArgumentParser(description='分布式消融实验训练（FlashAttention 2 + 动态Padding）- PERSONA-Bench')
    parser.add_argument('--config', type=str,
                       default='config_PERSONA_Bench.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='最大训练轮次（默认：50）')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值（默认：3）')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='早停阈值（默认：0.001）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='本地进程rank（由 torch.distributed.launch 自动设置）')
    parser.add_argument('--wandb_project', type=str, default='Qwen3-PERSONA-Bench',
                       help='Weights & Biases项目名称（默认：Qwen3-PERSONA-Bench）')
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
    
    # 只在主进程创建目录和日志文件
    training_log_path = None
    if is_main_process:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
            
            # 创建训练日志文件
            training_log_path = os.path.join(output_dir, "training_samples_log.txt")
            print(f"训练日志: {training_log_path}")
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
    train_config = config.get('training', {})
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
    
    # 在主进程中打印几个样本示例（用于调试和验证）
    if is_main_process and training_log_path:
        print("\n" + "=" * 80)
        print("📝 样本示例（前5个训练样本）")
        print("=" * 80)
        
        # 同时写入日志文件
        with open(training_log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write(f"训练配置: {config_name}\n")
            log_file.write(f"数据集: {train_path}\n")
            log_file.write(f"总样本数: {len(train_samples)}\n")
            log_file.write(f"Max Length: {train_config.get('max_length', 4096)}\n")
            log_file.write(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}\n")
            log_file.write("=" * 80 + "\n\n")
            
            num_samples_to_show = min(5, len(train_samples))
            for i in range(num_samples_to_show):
                sample = train_samples[i]
                
                # 控制台输出
                print(f"\n--- 样本 {i+1} ---")
                
                # 日志文件输出
                log_file.write(f"\n{'=' * 80}\n")
                log_file.write(f"样本 {i+1}\n")
                log_file.write(f"{'=' * 80}\n\n")
                
                # 显示角色映射的context
                context_info = f"Context ({len(sample['context'])}轮):"
                print(context_info)
                log_file.write(context_info + "\n")
                
                for j, turn in enumerate(sample['context']):
                    role = turn['role']
                    content = turn['content']
                    role_desc = "user(对话者)" if role == "user" else "assistant(目标用户)"
                    
                    # 控制台只显示前5轮，且截断
                    if j < 5:
                        print(f"  {j+1}. {role_desc:25s}: {content[:60]}...")
                    
                    # 日志文件显示完整内容
                    log_file.write(f"  {j+1}. {role_desc}:\n")
                    log_file.write(f"     {content}\n\n")
                
                if len(sample['context']) > 5:
                    print(f"  ... (还有 {len(sample['context']) - 5} 轮)")
                
                # 显示要预测的target
                target = sample['next_question']
                print(f"\nTarget (模型要生成的):")
                print(f"  assistant(目标用户): {target[:100]}...")
                
                log_file.write(f"\nTarget (模型要生成的):\n")
                log_file.write(f"  assistant(目标用户):\n")
                log_file.write(f"     {target}\n\n")
                
                # 显示profile信息
                if sample.get('user_profile'):
                    profile = sample['user_profile']
                    print(f"\nProfile:")
                    log_file.write(f"Profile:\n")
                    
                    for key in ['name', 'age', 'gender', 'profession', 'residence']:
                        if key in profile:
                            info = f"  {key.capitalize()}: {profile[key]}"
                            print(info)
                            log_file.write(info + "\n")
                
                # 使用dataset的__getitem__来获取编码后的信息
                try:
                    encoded_sample = train_dataset[i]
                    input_length = len(encoded_sample['input_ids'])
                    valid_labels = (encoded_sample['labels'] != -100).sum().item()
                    actual_length = encoded_sample.get('actual_length', input_length)
                    
                    encoding_info = [
                        f"\n编码信息:",
                        f"  输入长度: {input_length} tokens",
                        f"  实际长度: {actual_length} tokens",
                        f"  有效标签数: {valid_labels} tokens",
                        f"  训练比例: {valid_labels/input_length:.2%}"
                    ]
                    
                    for line in encoding_info:
                        print(line)
                        log_file.write(line + "\n")
                    
                    # 检查是否被截断
                    if hasattr(train_dataset, 'truncation_stats'):
                        stats = train_dataset.get_truncation_stats()
                        if stats['truncated_samples'] > 0:
                            truncation_info = f"  ⚠️  已有 {stats['truncated_samples']} 个样本被截断"
                            print(truncation_info)
                            log_file.write(truncation_info + "\n")
                    
                except Exception as e:
                    error_msg = f"\n编码信息: 无法获取 ({e})"
                    print(error_msg)
                    log_file.write(error_msg + "\n")
                
                log_file.write("\n")
        
        print(f"\n✓ 样本详情已保存到: {training_log_path}")
        print("=" * 80)
    
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
    
    # 创建自定义Trainer（带数值稳定性检查和详细日志）
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 创建训练进度日志文件
            if is_main_process:
                self.progress_log_file = os.path.join(output_dir, "training_logs", "training_progress.txt")
                os.makedirs(os.path.dirname(self.progress_log_file), exist_ok=True)
                with open(self.progress_log_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 100 + "\n")
                    f.write("训练进度日志\n")
                    f.write("=" * 100 + "\n\n")
            else:
                self.progress_log_file = None
        
        def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs) -> None:
            """
            重写log方法，修正梯度累积导致的train_loss显示问题，并添加详细日志
            """
            if "loss" in logs:
                # 修正train_loss：除以梯度累积步数
                logs["loss"] = logs["loss"] / self.args.gradient_accumulation_steps
            
            # 记录详细日志（前50步和每100步）
            if is_main_process and self.progress_log_file:
                step = self.state.global_step
                if step <= 50 or step % 100 == 0:
                    with open(self.progress_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'=' * 80}\n")
                        f.write(f"Step {step} | Epoch {self.state.epoch:.2f}\n")
                        f.write(f"{'=' * 80}\n")
                        for key, value in logs.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {key}: {value:.6f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                        f.write("\n")
            
            # 调用父类的log方法，传递所有额外参数
            if start_time is not None:
                super().log(logs, start_time, **kwargs)
            else:
                super().log(logs, **kwargs)
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """计算损失（带数值稳定性检查和batch日志）"""
            # 记录前3个batch的详细信息
            if is_main_process and self.state.global_step <= 3 and self.progress_log_file:
                with open(self.progress_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'=' * 100}\n")
                    f.write(f"Batch 详细信息 - Step {self.state.global_step}\n")
                    f.write(f"{'=' * 100}\n")
                    f.write(f"Batch size: {inputs['input_ids'].shape[0]}\n")
                    f.write(f"Sequence lengths: {inputs['input_ids'].shape[1]}\n")
                    
                    # 显示第一个样本的信息
                    if inputs['input_ids'].shape[0] > 0:
                        first_input_ids = inputs['input_ids'][0]
                        first_labels = inputs['labels'][0]
                        first_attention_mask = inputs['attention_mask'][0]
                        
                        f.write(f"\n第一个样本:\n")
                        f.write(f"  Input length: {len(first_input_ids)}\n")
                        f.write(f"  Valid labels: {(first_labels != -100).sum().item()}\n")
                        f.write(f"  Attention tokens: {first_attention_mask.sum().item()}\n")
                        
                        # 解码更多tokens以查看实际内容
                        try:
                            seq_len = len(first_input_ids)
                            
                            f.write(f"\n  解码的输入 (前500 tokens):\n")
                            f.write(f"  {tokenizer.decode(first_input_ids[:500], skip_special_tokens=False)}\n")
                            f.write(f"  ...\n")
                            
                            # 如果够长，打印中间部分
                            if seq_len > 1000:
                                f.write(f"\n  解码的输入 (第500-1000 tokens):\n")
                                f.write(f"  {tokenizer.decode(first_input_ids[500:1000], skip_special_tokens=False)}\n")
                                f.write(f"  ...\n")
                            
                            f.write(f"\n  解码的输入 (后500 tokens):\n")
                            f.write(f"  {tokenizer.decode(first_input_ids[-500:], skip_special_tokens=False)}\n\n")
                            
                            # 解码标签（完整显示，不截断）
                            valid_label_mask = first_labels != -100
                            if valid_label_mask.any():
                                valid_labels = first_labels[valid_label_mask]
                                f.write(f"  解码的标签 (完整有效部分，共{len(valid_labels)}个tokens):\n")
                                f.write(f"  {tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")
                        except Exception as e:
                            f.write(f"  解码失败: {e}\n")
                    
                    f.write("\n")
            
            # 移除actual_length字段（如果存在）
            actual_lengths = inputs.pop('actual_length', None)
            
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            
            # 检查并清理logits中的nan/inf
            if logits is not None and logits.numel() > 0:
                # 快速采样检查
                check_size = min(1000, logits.numel() // 2)
                if logits.numel() > check_size * 2:
                    head_values = logits.view(-1)[:check_size]
                    tail_values = logits.view(-1)[-check_size:]
                    has_issue = torch.isnan(head_values).any() or torch.isnan(tail_values).any() or \
                                torch.isinf(head_values).any() or torch.isinf(tail_values).any()
                else:
                    has_issue = torch.isnan(logits).any() or torch.isinf(logits).any()
                
                if has_issue:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} 检测到nan/inf，正在清理...")
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
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} 没有有效的labels")
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
                
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} loss为nan/inf")
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                elif loss.item() > 1e6:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} loss过大")
                    loss = torch.clamp(loss, max=100.0)
            
            # 定期清理CUDA缓存
            if self.state.global_step % 10 == 0:
                torch.cuda.empty_cache()
            
            if return_outputs:
                return loss, outputs
            return loss
    
    # 创建 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  # 使用动态padding的collate_fn
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # 创建训练日志文件（主进程）
    if is_main_process:
        log_dir = os.path.join(output_dir, "training_logs")
        os.makedirs(log_dir, exist_ok=True)
        training_log_file = os.path.join(log_dir, "detailed_training_log.txt")
        
        print(f"\n📝 创建详细训练日志: {training_log_file}")
        
        with open(training_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("详细训练日志 - 前3个训练样本\n")
            f.write("=" * 100 + "\n\n")
            
            # 记录前3个训练样本的详细信息
            num_samples_to_log = min(3, len(train_dataset))
            for idx in range(num_samples_to_log):
                raw_sample = train_samples[idx]
                encoded_sample = train_dataset[idx]
                
                f.write(f"\n{'=' * 100}\n")
                f.write(f"训练样本 #{idx + 1}\n")
                f.write(f"{'=' * 100}\n\n")
                
                # 1. 原始样本信息
                f.write("【原始样本信息】\n")
                f.write(f"User Hash: {raw_sample.get('user_hash', 'N/A')}\n")
                if raw_sample.get('user_profile'):
                    profile = raw_sample['user_profile']
                    f.write(f"User Profile: {profile.get('name', 'N/A')} (age: {profile.get('age', 'N/A')})\n")
                f.write("\n")
                
                # 2. 对话上下文
                f.write("【对话上下文 Context】\n")
                context = raw_sample.get('context', [])
                for turn_idx, turn in enumerate(context[-5:], 1):  # 只显示最后5轮
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    f.write(f"  轮次{turn_idx} [{role}]: {content}\n")
                if len(context) > 5:
                    f.write(f"  ... (还有 {len(context) - 5} 轮对话)\n")
                f.write("\n")
                
                # 3. 目标输出（模型要学习生成的内容）
                f.write("【目标输出 Next Question】\n")
                next_question = raw_sample.get('next_question', '')
                f.write(f"{next_question}\n\n")
                
                # 4. 历史信息（如果有）
                if use_history and raw_sample.get('history'):
                    f.write("【历史信息 History】\n")
                    history = raw_sample['history']
                    for hist_idx, hist_item in enumerate(history[:3], 1):  # 只显示前3条
                        f.write(f"  历史{hist_idx}: {hist_item[:100]}...\n")
                    if len(history) > 3:
                        f.write(f"  ... (还有 {len(history) - 3} 条历史)\n")
                    f.write("\n")
                
                # 5. 编码后的信息
                f.write("【编码后的数据】\n")
                input_ids = encoded_sample['input_ids']
                labels = encoded_sample['labels']
                attention_mask = encoded_sample['attention_mask']
                
                f.write(f"Input IDs 长度: {len(input_ids)}\n")
                f.write(f"Attention Mask 长度: {len(attention_mask)}\n")
                f.write(f"Labels 长度: {len(labels)}\n")
                
                valid_labels = (labels != -100).sum().item()
                f.write(f"有效标签数: {valid_labels}\n")
                f.write(f"训练比例: {valid_labels / len(labels):.2%}\n")
                
                # 解码查看实际的文本（更详细的打印）
                total_length = len(input_ids)
                
                # 如果序列不太长（< 6000 tokens），直接打印完整内容
                if total_length <= 6000:
                    f.write("\n【完整的输入文本】\n")
                    f.write("-" * 100 + "\n")
                    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
                    f.write(decoded_full + "\n")
                    f.write("-" * 100 + "\n\n")
                    f.write(f"总序列长度: {total_length} tokens (已打印完整内容)\n\n")
                else:
                    # 序列太长，分段打印
                    f.write(f"\n【序列太长 ({total_length} tokens)，分段打印】\n\n")
                    
                    # 打印前2000个tokens
                    f.write("【第1-2000 tokens】\n")
                    f.write("-" * 100 + "\n")
                    decoded_input_start = tokenizer.decode(input_ids[:2000], skip_special_tokens=False)
                    f.write(decoded_input_start + "\n")
                    f.write("-" * 100 + "\n\n")
                    
                    # 打印中间部分（第2000-4000个tokens）
                    f.write("【第2001-4000 tokens】\n")
                    f.write("-" * 100 + "\n")
                    decoded_input_middle = tokenizer.decode(input_ids[2000:4000], skip_special_tokens=False)
                    f.write(decoded_input_middle + "\n")
                    f.write("-" * 100 + "\n\n")
                    
                    # 如果还有更多，打印第4000-6000
                    if total_length > 6000:
                        f.write("【第4001-6000 tokens】\n")
                        f.write("-" * 100 + "\n")
                        decoded_input_middle2 = tokenizer.decode(input_ids[4000:6000], skip_special_tokens=False)
                        f.write(decoded_input_middle2 + "\n")
                        f.write("-" * 100 + "\n\n")
                    
                    # 打印后2000个tokens
                    f.write("【后2000 tokens】\n")
                    f.write("-" * 100 + "\n")
                    decoded_input_end = tokenizer.decode(input_ids[-2000:], skip_special_tokens=False)
                    f.write(decoded_input_end + "\n")
                    f.write("-" * 100 + "\n\n")
                    
                    f.write(f"总序列长度: {total_length} tokens\n\n")
                
                # 解码标签（只显示有效的部分）
                valid_label_indices = (labels != -100).nonzero(as_tuple=True)[0]
                if len(valid_label_indices) > 0:
                    f.write("【解码后的标签文本 (模型要学习生成的部分)】\n")
                    f.write("-" * 100 + "\n")
                    valid_labels_ids = labels[valid_label_indices]
                    decoded_labels = tokenizer.decode(valid_labels_ids, skip_special_tokens=False)
                    f.write(decoded_labels + "\n")
                    f.write("-" * 100 + "\n\n")
                
                f.write("\n")
            
            f.write("=" * 100 + "\n")
            f.write("训练样本日志记录完成\n")
            f.write("=" * 100 + "\n")
        
        print(f"✓ 训练样本日志已保存到: {training_log_file}\n")
        
        # 在控制台显示第一个样本的简要信息
        print("=" * 80)
        print("📋 第一个训练样本预览")
        print("=" * 80)
        
        first_sample = train_samples[0]
        print(f"User Hash: {first_sample.get('user_hash', 'N/A')}")
        
        context = first_sample.get('context', [])
        if context:
            print(f"\nContext 最后一轮:")
            last_turn = context[-1]
            print(f"  [{last_turn.get('role', 'unknown')}]: {last_turn.get('content', '')[:150]}...")
        
        next_question = first_sample.get('next_question', '')
        print(f"\nTarget (要学习生成的):")
        print(f"  {next_question[:150]}...")
        
        first_encoded = train_dataset[0]
        print(f"\n编码信息:")
        print(f"  Input length: {len(first_encoded['input_ids'])} tokens")
        print(f"  Valid labels: {(first_encoded['labels'] != -100).sum().item()} tokens")
        print(f"  训练比例: {(first_encoded['labels'] != -100).sum().item() / len(first_encoded['labels']):.2%}")
        
        print("=" * 80 + "\n")
    
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
    
    # 训练完成，输出日志汇总
    if is_main_process:
        print("\n" + "=" * 80)
        print("📊 训练日志汇总")
        print("=" * 80)
        
        log_dir = os.path.join(output_dir, "training_logs")
        if os.path.exists(log_dir):
            print(f"详细日志文件:")
            for log_file_name in os.listdir(log_dir):
                log_path = os.path.join(log_dir, log_file_name)
                file_size = os.path.getsize(log_path) / 1024  # KB
                print(f"  - {log_path} ({file_size:.1f} KB)")
        print("=" * 80 + "\n")
    
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
            print(" 训练数据截断统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  被截断样本数: {stats['truncated_samples']}")
            print(f"  截断率: {stats['truncation_rate']:.2%}")
            print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
            print("="*80)
            
            # 将截断统计写入日志文件
            if training_log_path:
                try:
                    with open(training_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write("\n" + "="*80 + "\n")
                        log_file.write("📊 最终训练数据截断统计\n")
                        log_file.write("="*80 + "\n")
                        log_file.write(f"总样本数: {stats['total_samples']}\n")
                        log_file.write(f"被截断样本数: {stats['truncated_samples']}\n")
                        log_file.write(f"截断率: {stats['truncation_rate']:.2%}\n")
                        log_file.write(f"平均截断轮次: {stats['avg_truncated_turns']:.2f}\n")
                        log_file.write(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}\n")
                        log_file.write("="*80 + "\n")
                    print(f"✓ 截断统计已追加到: {training_log_path}")
                except Exception as e:
                    print(f"警告: 无法写入截断统计到日志文件: {e}")
    
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
