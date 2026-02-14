"""
消融实验训练脚本（带早停机制 + 动态Batch Padding优化）
关键优化：不再将batch内所有样本padding到固定max_length，
而是动态padding到batch内最长样本的实际长度，大幅节省显存。
"""
import json
import argparse
import os
import sys
from pathlib import Path
import random
import torch

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
# from data_loader import load_train_data, extract_training_samples, get_user_history_samples, get_user_only_history # 旧版本 复杂的训练prompt  
from data_loader_more_data import load_train_data, extract_training_samples, get_user_only_history # 新版本 简短的训练prompt
from trainer_pc import AblationTrainer
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from typing import List, Dict, Any, Optional
import torch.nn as nn
from torch.utils.data import Dataset


def split_train_val(samples, val_ratio=0.1, seed=42):
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
    """
    def __init__(self, samples, tokenizer, max_length=32768, use_profile=True, use_history=True, use_context=True, verbose=False, use_detailed_template=True, max_context_turns=15, template_filename=None, require_token_type_ids=None):
        # 使用绝对路径导入，确保使用当前目录的模块
        import sys
        from pathlib import Path
        current_dir = str(Path(__file__).parent.absolute())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # ✅ 根据 use_detailed_template 选择 prompt 构建函数
        if use_detailed_template:
            # 使用详细模板（标准 markdown 格式，使用 {VAR_NAME} 占位符）
            from prompt_builder import build_training_prompt
            print("使用详细 Prompt 模板 (prompt_builder)")
            self.build_training_prompt = build_training_prompt
        else:
            # 使用简短模板
            # 优先尝试从 data_loader.py 导入（新版本，只预测 continuation）
            # 如果失败，则从 data_loader_more_data.py 导入（旧版本，数据扩充）
            try:
                from data_loader import build_simple_training_prompt as build_training_prompt
                print("✅ 使用简短 Prompt 模板 (data_loader.build_simple_training_prompt - 只预测continuation)")
                self.build_training_prompt = build_training_prompt
            except ImportError:
                from data_loader_more_data import build_simple_training_prompt as build_training_prompt
                print("✅ 使用简短 Prompt 模板 (data_loader_more_data.build_simple_training_prompt)")
                self.build_training_prompt = build_training_prompt
        
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        self.use_detailed_template = use_detailed_template  # 是否使用详细模板
        self.max_context_turns = max_context_turns  # 新增：最大保留的 context 轮次数
        self.template_filename = template_filename  # 新增：模板文件名
        self.verbose = verbose  # 是否输出详细日志
        
        # ✅ 自动检测是否需要 token_type_ids（用于 Gemma3 模型）
        if require_token_type_ids is None:
            # 根据 tokenizer 自动判断
            model_type = getattr(tokenizer, 'name_or_path', '').lower()
            self.require_token_type_ids = 'gemma' in model_type
            if self.require_token_type_ids and verbose:
                print(f"✅ 检测到 Gemma 模型，将添加 token_type_ids")
        else:
            self.require_token_type_ids = require_token_type_ids
        
        # 截断统计
        self.truncation_stats = {
            'total_samples': 0,
            'truncated_samples': 0,
            'truncated_turns': 0,
            # 历史记录统计
            'total_history_items': 0,
            'truncated_history_items': 0,
            'samples_with_history': 0,
            'samples_with_history_truncated': 0
        }
        
        # 用于记录第一次截断的样本信息（调试用）
        self.first_truncation_logged = False

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

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 统计历史记录信息
        original_history = sample.get('history', []) if self.use_history else []
        has_history = len(original_history) > 0
        original_history_count = len(original_history)
        
        if has_history:
            self.truncation_stats['samples_with_history'] += 1
            self.truncation_stats['total_history_items'] += original_history_count
        
        # 1. 构建消息
        # ✅ 根据模板类型，传递不同的参数
        if self.use_detailed_template:
            # 详细模板需要额外的参数
            messages, target_answer = self.build_training_prompt(
                context=sample['context'],
                next_question=sample['next_question'],
                user_profile=sample.get('user_profile') if self.use_profile else None,
                task_description=sample.get('task_description'),
                history=original_history,
                use_profile=self.use_profile,
                use_history=self.use_history,
                use_context=self.use_context,
                use_detailed_template=self.use_detailed_template,
                max_context_turns=self.max_context_turns,
                tokenizer=self.tokenizer,
                template_filename=self.template_filename  # ✅ 传递模板文件名
            )
        else:
            # 简短模板 - ✅ 添加 tokenizer 和 max_length 用于动态长度调整
            messages, target_answer = self.build_training_prompt(
                context=sample['context'],
                next_question=sample['next_question'],
                user_profile=sample.get('user_profile') if self.use_profile else None,
                task_description=sample.get('task_description'),
                history=original_history,
                use_profile=self.use_profile,
                use_history=self.use_history,
                use_context=self.use_context,
                tokenizer=self.tokenizer,         # ✅ 传递 tokenizer
                max_length=self.max_length,       # ✅ 传递 max_length
                min_target_tokens=64,             # ✅ 预留 64 tokens 给 target
                user_hash=sample.get('user_hash')  # ✅ 传递 user_hash（始终包含）
            )


        # 检查历史记录是否被截断（在 prompt_builder 中限制为前5个）
        if has_history and original_history_count > 5:
            truncated_history_count = original_history_count - 5
            self.truncation_stats['truncated_history_items'] += truncated_history_count
            self.truncation_stats['samples_with_history_truncated'] += 1


        # 2. 生成完整文本
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # 修正：应该生成assistant角色的回复（目标用户）
        generation_suffix = "<|im_start|>assistant\n"
        full_prompt = full_prompt.strip() + generation_suffix
        im_end_token = "<|im_end|>"
        full_text = full_prompt + target_answer + im_end_token
        
        # ✅ 第二层保护：如果仍然超长，逐步从前往后删除对话轮次
        target_with_end = target_answer + im_end_token
        target_tokens = len(self.tokenizer.encode(target_with_end, add_special_tokens=False))
        min_buffer = 64
        
        full_length = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        is_truncated = False
        removed_turns = 0
        
        if full_length > self.max_length:
            is_truncated = True
            
            # 允许的最大 prompt 长度
            max_prompt_tokens = self.max_length - target_tokens - min_buffer
            
            # 如果有 RECENT_DIALOGUE 部分，逐步从前往后删除旧对话
            if len(messages) > 0 and messages[0].get('role') == 'system':
                system_content = messages[0]['content']
                
                if '[RECENT_DIALOGUE]' in system_content:
                    # 解析 dialogue 部分
                    parts = system_content.split('[RECENT_DIALOGUE]')
                    if len(parts) > 1:
                        prefix = parts[0].strip()  # Profile + Task
                        dialogue_section = parts[1].strip()
                        
                        # 提取对话行（跳过 "Predict the user's next message:"）
                        dialogue_lines = []
                        for line in dialogue_section.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('Predict') and not line.startswith('（前面省略'):
                                if line.startswith('User:') or line.startswith('Assistant:'):
                                    dialogue_lines.append(line)
                        
                        # 从前往后逐步删除对话轮次，直到长度合适
                        while dialogue_lines and full_length > self.max_length:
                            # 删除最旧的一轮（第一个）
                            dialogue_lines.pop(0)
                            removed_turns += 1
                            
                            # 重建 system message
                            if removed_turns > 0 and dialogue_lines:
                                new_dialogue = f"\n[RECENT_DIALOGUE]\n（前面省略了 {removed_turns} 轮对话）\n" + "\n".join(dialogue_lines)
                            elif dialogue_lines:
                                new_dialogue = "\n[RECENT_DIALOGUE]\n" + "\n".join(dialogue_lines)
                            else:
                                new_dialogue = ""
                            
                            new_system = prefix + new_dialogue + "\n\nPredict the user's next message:"
                            messages[0]['content'] = new_system
                            
                            # 重新生成并测试长度
                            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                            full_prompt = full_prompt.strip() + generation_suffix
                            full_text = full_prompt + target_answer + im_end_token
                            full_length = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        
        # 更新截断统计
        self.truncation_stats['total_samples'] += 1
        if is_truncated:
            self.truncation_stats['truncated_samples'] += 1
            self.truncation_stats['truncated_turns'] += removed_turns
            
            # 第一次遇到截断时输出日志
            if not self.first_truncation_logged and self.verbose:
                self.first_truncation_logged = True
                print(f"\n⚠️  第二层保护：逐步删除旧对话 (样本#{idx}):")
                print(f"  删除了 {removed_turns} 轮对话（从最旧的开始）")
                print(f"  调整后长度: {full_length} tokens")
                print(f"  最大长度: {self.max_length} tokens")
                print(f"  Target 长度: {target_tokens} tokens (已完整保留)")
                print(f"  (后续截断将不再输出详细信息)\n")

        # 3. 编码 - 关键：不做padding！
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 关键改动：不padding
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # 4. 计算labels
        target_ids = self.tokenizer.encode(target_answer, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        actual_prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        safe_prompt_len = min(actual_prompt_len, len(input_ids) - 1)
        labels[:safe_prompt_len] = -100
        
        # 屏蔽padding token（虽然现在没有padding，但为了兼容性保留）
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'actual_length': len(input_ids)  # 记录实际长度，用于调试
        }
        
        # ✅ 只有 Gemma3 模型需要时才添加 token_type_ids
        if self.require_token_type_ids:
            token_type_ids = torch.zeros_like(input_ids)
            result['token_type_ids'] = token_type_ids
        
        return result


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
    padded_token_type_ids = []  # ✅ 添加 token_type_ids
    
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
        
        # ✅ Padding token_type_ids
        if 'token_type_ids' in ex:
            padded_token_type_ids.append(
                torch.cat([
                    ex['token_type_ids'],
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            )
    
    batch['input_ids'] = torch.stack(padded_input_ids)
    batch['attention_mask'] = torch.stack(padded_attention_mask)
    batch['labels'] = torch.stack(padded_labels)
    
    # ✅ 添加 token_type_ids 到 batch
    if padded_token_type_ids:
        batch['token_type_ids'] = torch.stack(padded_token_type_ids)
    
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
