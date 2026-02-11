"""
训练器模块 - 优化版
适配消融实验，支持严格的角色控制与日志监控
添加 Emoji 过滤功能
"""
import os
import re
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加当前目录到路径，确保能导入模块
sys.path.insert(0, str(Path(__file__).parent))

# ✅ 使用新的简短 prompt 构建函数（现在在同一目录下）
try:
    from data_loader_more_data import build_simple_training_prompt
    print("使用简短 prompt 构建函数 (data_loader_more_data)")
except ImportError as e:
    print(f"⚠️ 无法导入 data_loader_more_data: {e}")
    print("⚠️ 回退到详细 prompt 构建函数")
    from prompt_builder_LovinkDialogue import build_training_prompt as build_simple_training_prompt

# 导入 emoji 过滤模块
try:
    from emoji_filter import contains_emoji
except ImportError:
    print("警告: 无法导入 emoji_filter，将跳过 emoji 过滤")
    def contains_emoji(text):
        return False


class AblationDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=32768, use_profile=True, use_history=True, use_context=True, filter_emoji=True):
        """
        Args:
            samples: 训练样本列表
            tokenizer: tokenizer实例
            max_length: 最大序列长度
            use_profile: 是否使用 profile
            use_history: 是否使用 history
            use_context: 是否使用 context
            filter_emoji: 是否在数据集层面再次过滤 emoji（双重保险，默认True）
        """
        # Emoji 过滤（双重保险）
        if filter_emoji:
            original_count = len(samples)
            filtered_samples = []
            emoji_count = 0
            
            for sample in samples:
                target_text = sample.get('next_question', '')
                if contains_emoji(target_text):
                    emoji_count += 1
                    continue
                filtered_samples.append(sample)
            
            self.samples = filtered_samples
            
            if emoji_count > 0:
                print(f"\n{'='*80}")
                print(f"🚫 Dataset 层 Emoji 二次过滤:")
                print(f"  原始样本数: {original_count}")
                print(f"  额外过滤 emoji 样本数: {emoji_count}")
                print(f"  最终样本数: {len(self.samples)}")
                print(f"  额外过滤比例: {emoji_count / original_count * 100:.2f}%")
                print(f"{'='*80}\n")
        else:
            self.samples = samples
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        
        # 截断统计
        self.truncation_stats = {
            'total_samples': 0,
            'truncated_samples': 0,
            'truncated_turns': 0
        }
        
        # Emoji 统计
        self.emoji_stats = {
            'checked_samples': 0,
            'emoji_found': 0
        }

    def __len__(self):
        return len(self.samples)
    
    def get_truncation_stats(self):
        """获取截断统计信息"""
        if self.truncation_stats['total_samples'] == 0:
            return {
                'truncation_rate': 0.0,
                'avg_truncated_turns': 0.0,
                'total_samples': 0,
                'truncated_samples': 0
            }
        
        truncation_rate = self.truncation_stats['truncated_samples'] / self.truncation_stats['total_samples']
        avg_truncated_turns = (self.truncation_stats['truncated_turns'] / self.truncation_stats['truncated_samples'] 
                               if self.truncation_stats['truncated_samples'] > 0 else 0)
        
        return {
            'truncation_rate': truncation_rate,
            'avg_truncated_turns': avg_truncated_turns,
            'total_samples': self.truncation_stats['total_samples'],
            'truncated_samples': self.truncation_stats['truncated_samples']
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 运行时 emoji 检测（最后一道防线，理论上不应该触发）
        target_text = sample.get('next_question', '')
        self.emoji_stats['checked_samples'] += 1
        
        if contains_emoji(target_text):
            self.emoji_stats['emoji_found'] += 1
            # 如果在运行时发现 emoji，记录警告但继续训练
            # （因为已经在 __init__ 时过滤过了，这里应该不会触发）
            if self.emoji_stats['emoji_found'] <= 3:  # 只打印前3次
                print(f"⚠️  警告: 在运行时检测到 emoji（样本 #{idx}）: {target_text[:50]}...")
        
        # 1. 初始构建 - ✅ 使用简短 prompt 构建函数
        messages, target_answer = build_simple_training_prompt(
            context=sample['context'],
            next_question=sample['next_question'],
            user_profile=sample.get('user_profile') if self.use_profile else None,
            task_description=sample.get('task_description'),
            history=sample.get('history') if self.use_history else None,
            use_profile=self.use_profile,
            use_history=self.use_history,
            use_context=self.use_context
        )

        # 记录原始消息长度
        original_message_count = len(messages)
        is_truncated = False
        truncated_turns = 0
        
        # --- 核心优化：动态裁剪历史以防止截断 ---
        # 如果消息太长，循环删除 messages 中最早的对话轮次（保留 system 提示词）
        # 索引 0 是 system，1 和 2 是最早的一对 user/assistant
        while len(self.tokenizer.apply_chat_template(messages, tokenize=True)) > (self.max_length - 512):
            if len(messages) > 2:
                messages.pop(1) # 弹出最早的对话
                is_truncated = True
                truncated_turns += 1
            else:
                break
        
        # 更新截断统计
        self.truncation_stats['total_samples'] += 1
        if is_truncated:
            self.truncation_stats['truncated_samples'] += 1
            self.truncation_stats['truncated_turns'] += truncated_turns

        # 2. 生成 Prompt (手动添加引导符)
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # 修正：应该生成assistant角色的回复（目标用户）
        generation_suffix = "<|im_start|>assistant\n"

        # 3. 组合成真正的 Prompt
        full_prompt = full_prompt.strip() + generation_suffix
        # 确保不包含答案，使用 <|im_end|> 作为结束标记（让模型学会在正确位置停止）
        im_end_token = "<|im_end|>"
        full_text = full_prompt + target_answer + im_end_token

        # 3. 编码
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # --- 核心优化：高精度计算 Prompt 长度 ---
        # 我们不直接 encode(full_prompt)，而是通过寻找 target 的起始 token 来确定
        target_ids = self.tokenizer.encode(target_answer, add_special_tokens=False)
        
        # 寻找分界点：在 input_ids 中找到第一个不属于 prompt 的位置
        # 我们可以先 encode 一个完全没带特殊字符的 prompt
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        actual_prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        
        # 屏蔽 Prompt：确保不会越界
        safe_prompt_len = min(actual_prompt_len, self.max_length - 1)
        labels[:safe_prompt_len] = -100
        
        # 屏蔽 Padding
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # --- 屏蔽特殊 Token (保留 EOS 和 <|im_end|>) ---
        # 获取 <|im_end|> 的 token ID，确保它被包含在损失计算中
        im_end_token = "<|im_end|>"
        im_end_id = None
        try:
            # 尝试获取 <|im_end|> 的 token ID
            im_end_ids = self.tokenizer.encode(im_end_token, add_special_tokens=False)
            if im_end_ids:
                im_end_id = im_end_ids[0]  # 通常 <|im_end|> 是一个单独的 token
                # 调试信息（只在第一次打印）
                if not hasattr(self, '_im_end_logged'):
                    print(f"✓ <|im_end|> token ID: {im_end_id}，将被包含在损失计算中")
                    self._im_end_logged = True
        except Exception as e:
            if not hasattr(self, '_im_end_error_logged'):
                print(f"警告: 无法获取 <|im_end|> token ID: {e}")
                self._im_end_error_logged = True
        
        special_ids = set(self.tokenizer.all_special_ids)
        eos_id = self.tokenizer.eos_token_id
        # 保留 EOS 和 <|im_end|> token，让模型学会在正确位置停止
        tokens_to_keep = {eos_id}
        if im_end_id is not None:
            tokens_to_keep.add(im_end_id)
        
        for tid in special_ids:
            if tid not in tokens_to_keep:
                labels[labels == tid] = -100
        
        # 验证 <|im_end|> 是否在 labels 中（用于调试）
        if im_end_id is not None and (labels == im_end_id).any():
            if not hasattr(self, '_im_end_verified'):
                print(f"✓ 确认: <|im_end|> token (ID: {im_end_id}) 已包含在损失计算中")
                self._im_end_verified = True

        # 4. 最终验证：防止 NaN
        if (labels != -100).sum() == 0:
            # 挽救逻辑：如果全被屏蔽了（说明截断太严重），强行暴露最后 32 个 token 
            # 这种情况通常发生在答案极长或截断刚好切在了答案开头
            labels[-32:] = input_ids[-32:]
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CustomTrainer(Trainer):
    """带实时日志的自定义训练器"""
    
    def __init__(self, *args, verbose_logging=False, log_file_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose_logging = verbose_logging
        self.log_file_path = log_file_path
        self.log_entry_count = 0
        
        if self.log_file_path:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            self.log_file.write("[\n")

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.write("\n]")
                self.log_file.close()
            except: pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        if loss is None and "labels" in inputs:
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if self.verbose_logging and (self.state.global_step % self.args.logging_steps == 0):
            self._log_details(inputs, outputs, loss.item())

        return (loss, outputs) if return_outputs else loss

    def clean_output_text(self, text: str) -> str:
        # 移除思考过程
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.replace('<think>', '').replace('</think>', '')
        return text.strip()

    def _log_details(self, inputs, outputs, loss_val):
        """记录训练细节：对比 Target 和模型的预测 (Argmax)"""
        try:
            batch_idx = 0
            ids = inputs['input_ids'][batch_idx]
            lbs = inputs['labels'][batch_idx]
            logits = outputs.get("logits")[batch_idx]
            
            # 统计信息
            total_tokens = len(ids)
            valid_label_count = (lbs != -100).sum().item()
            
            # 解码 Target
            target_ids = [t.item() for t in lbs if t != -100]
            target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True)
            
            # 解码预测 (寻找 label 有效位对应的预测位)
            pred_ids_all = logits.argmax(dim=-1)
            valid_pos = (lbs != -100).nonzero(as_tuple=True)[0]
            pred_ids = [pred_ids_all[p-1].item() for p in valid_pos if p > 0]
            predict_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            # 计算准确匹配的 token 数量
            correct_tokens = 0
            for i, pos in enumerate(valid_pos):
                if pos > 0 and i < len(pred_ids):
                    if lbs[pos].item() == pred_ids_all[pos-1].item():
                        correct_tokens += 1
            
            token_accuracy = correct_tokens / valid_label_count if valid_label_count > 0 else 0
            
            # 打印详细信息
            print(f"\n{'='*100}")
            print(f"[Step {self.state.global_step}] 训练日志")
            print(f"{'='*100}")
            print(f"📊 统计信息:")
            print(f"  Loss: {loss_val:.4f}")
            print(f"  总 Token 数: {total_tokens}")
            print(f"  有效标签数: {valid_label_count} (训练比例: {valid_label_count/total_tokens:.2%})")
            print(f"  Token 准确率: {token_accuracy:.2%} ({correct_tokens}/{valid_label_count})")
            print(f"\n🎯 预测目标 (Target):")
            print(f"  长度: {len(target_text)} 字符")
            if len(target_text) <= 200:
                print(f"  完整内容: {target_text}")
            else:
                print(f"  前100字: {target_text[:100]}")
                print(f"  后100字: {target_text[-100:]}")
            
            print(f"\n🤖 模型预测 (Prediction):")
            print(f"  长度: {len(predict_text)} 字符")
            if len(predict_text) <= 200:
                print(f"  完整内容: {predict_text}")
            else:
                print(f"  前100字: {predict_text[:100]}")
                print(f"  后100字: {predict_text[-100:]}")
            
            # 简单的文本相似度提示
            if target_text == predict_text:
                print(f"\n✅ 完全匹配！")
            elif target_text[:50] == predict_text[:50]:
                print(f"\n⚠️  前50字匹配，后续有差异")
            else:
                print(f"\n❌ 预测与目标差异较大")
            
            print(f"{'='*100}\n")

            # 保存到日志文件
            if hasattr(self, 'log_file'):
                log_data = {
                    "step": self.state.global_step,
                    "loss": loss_val,
                    "stats": {
                        "total_tokens": total_tokens,
                        "valid_labels": valid_label_count,
                        "training_ratio": f"{valid_label_count/total_tokens:.2%}",
                        "token_accuracy": f"{token_accuracy:.2%}",
                        "correct_tokens": correct_tokens
                    },
                    "target": {
                        "text": target_text,
                        "length": len(target_text)
                    },
                    "prediction": {
                        "text": predict_text,
                        "length": len(predict_text)
                    },
                    "match_status": "full" if target_text == predict_text else "partial" if target_text[:50] == predict_text[:50] else "different"
                }
                if self.log_entry_count > 0: self.log_file.write(",\n")
                self.log_file.write(json.dumps(log_data, ensure_ascii=False, indent=2))
                self.log_file.flush()
                self.log_entry_count += 1
        except Exception as e:
            print(f"❌ Log Error: {e}")
            import traceback
            traceback.print_exc()


class AblationTrainer:
    """消融实验主控类"""
    
    def __init__(self, model_path: str, output_dir: str, config: Dict[str, Any], 
                 use_profile: bool = True, use_history: bool = True, use_context: bool = True, log_file_path: Optional[str] = None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        self.log_file_path = log_file_path

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        ).to(self.device)
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def train(self, train_samples: List[Dict[str, Any]], val_samples: Optional[List[Dict[str, Any]]] = None):
        train_config = self.config.get('training', {})
        
        train_dataset = AblationDataset(
            train_samples, self.tokenizer, 
            max_length=train_config.get('max_length', 32768),
            use_profile=self.use_profile, use_history=self.use_history, use_context=self.use_context
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_config.get('num_epochs', 3),
            per_device_train_batch_size=train_config.get('batch_size', 1),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 16),
            learning_rate=train_config.get('learning_rate', 2e-5),
            logging_steps=10,
            save_steps=100,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
            remove_unused_columns=False
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            verbose_logging=True,
            log_file_path=self.log_file_path
        )

        print(f"🚀 开始训练: Profile={self.use_profile}, History={self.use_history}, Context={self.use_context}")
        trainer.train()
        
        # 保存
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 输出截断统计
        if hasattr(train_dataset, 'get_truncation_stats'):
            stats = train_dataset.get_truncation_stats()
            print("\n" + "="*80)
            print("📊 训练数据截断统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  被截断样本数: {stats['truncated_samples']}")
            print(f"  截断率: {stats['truncation_rate']:.2%}")
            print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
            print("="*80)
        
        # 输出 emoji 统计
        if hasattr(train_dataset, 'emoji_stats'):
            emoji_stats = train_dataset.emoji_stats
            print("\n" + "="*80)
            print("🚫 训练过程 Emoji 检测统计:")
            print(f"  检查的样本数: {emoji_stats['checked_samples']}")
            print(f"  运行时发现 emoji 数: {emoji_stats['emoji_found']}")
            if emoji_stats['emoji_found'] > 0:
                print(f"  ⚠️  警告: 有 {emoji_stats['emoji_found']} 个样本在运行时检测到 emoji")
                print(f"     这表明数据过滤可能存在遗漏，请检查")
            else:
                print(f"  ✓ 完美: 运行时未检测到任何 emoji")
            print("="*80)