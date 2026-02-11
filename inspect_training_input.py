#!/usr/bin/env python3
"""
查看训练数据的实际输入脚本
使用方法：python inspect_training_input.py --config config_DMSC.json --ablation_config history_only --num_samples 3
"""

import json
import argparse
import sys
from pathlib import Path

# 导入数据加载模块
from data_loader_more_data import load_train_data, extract_training_samples, get_user_only_history
from train_with_dynamic_padding_Lovink import DynamicPaddingDataset, add_history_to_samples
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='查看训练数据的实际输入')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--num_samples', type=int, default=3, help='查看样本数量')
    parser.add_argument('--output_file', type=str, default=None, help='输出文件路径（默认打印到终端）')
    
    args = parser.parse_args()
    
    # 1. 加载配置
    print(f"正在加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_config = config['model']
    data_config = config['data']
    train_config = config['training']
    ablation_configs = config['ablation_configs']
    
    # 2. 获取消融配置
    ablation_config = ablation_configs[args.ablation_config]
    use_profile = ablation_config['use_profile']
    use_history = ablation_config['use_history']
    use_context = ablation_config['use_context']
    
    print(f"\n消融配置: {args.ablation_config}")
    print(f"  - 使用画像 (Profile): {use_profile}")
    print(f"  - 使用历史 (History): {use_history}")
    print(f"  - 使用上下文 (Context): {use_context}")
    
    # 3. 加载 tokenizer
    print(f"\n正在加载 tokenizer: {model_config['path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['path'],
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4. 加载数据
    print(f"\n正在加载训练数据: {data_config['train_path']}")
    data = load_train_data(data_config['train_path'])
    samples = extract_training_samples(data)
    
    # 添加历史信息
    samples = add_history_to_samples(samples)
    
    print(f"  加载了 {len(samples)} 个训练样本")
    
    # 5. 创建数据集
    print(f"\n创建数据集...")
    dataset = DynamicPaddingDataset(
        samples=samples[:args.num_samples],  # 只取前N个样本
        tokenizer=tokenizer,
        max_length=train_config.get('max_length', 4096),
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context,
        verbose=True,
        use_detailed_template=False  # 使用简短模板
    )
    
    # 6. 输出详细信息
    output_lines = []
    output_lines.append("=" * 120)
    output_lines.append("训练数据实际输入查看")
    output_lines.append("=" * 120)
    output_lines.append("")
    
    for idx in range(len(dataset)):
        raw_sample = samples[idx]
        encoded_sample = dataset[idx]
        
        output_lines.append("")
        output_lines.append("=" * 120)
        output_lines.append(f"样本 #{idx + 1}")
        output_lines.append("=" * 120)
        output_lines.append("")
        
        # 原始样本信息
        output_lines.append("【1. 原始样本信息】")
        output_lines.append(f"User Hash: {raw_sample.get('user_hash', 'N/A')}")
        if raw_sample.get('user_profile'):
            profile = raw_sample['user_profile']
            output_lines.append(f"User Profile: {profile}")
        output_lines.append("")
        
        # 对话上下文
        if use_context:
            output_lines.append("【2. 对话上下文 (Context)】")
            context = raw_sample.get('context', [])
            if context:
                for turn_idx, turn in enumerate(context[-10:], 1):  # 显示最后10轮
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    output_lines.append(f"  轮次{turn_idx} [{role}]: {content}")
                if len(context) > 10:
                    output_lines.append(f"  ... (还有 {len(context) - 10} 轮对话)")
            else:
                output_lines.append("  (无上下文)")
            output_lines.append("")
        
        # 历史信息
        if use_history:
            output_lines.append("【3. 历史信息 (History)】")
            history = raw_sample.get('history', [])
            if history:
                for hist_idx, hist_item in enumerate(history[:10], 1):  # 显示前10条
                    output_lines.append(f"  历史{hist_idx}: {hist_item}")
                if len(history) > 10:
                    output_lines.append(f"  ... (还有 {len(history) - 10} 条历史)")
            else:
                output_lines.append("  (无历史)")
            output_lines.append("")
        
        # 目标输出
        output_lines.append("【4. 目标输出 (Target)】")
        next_question = raw_sample.get('next_question', '')
        output_lines.append(f"{next_question}")
        output_lines.append("")
        
        # 解码实际输入给模型的文本
        output_lines.append("【5. 实际输入给模型的完整文本 (Full Input Text)】")
        output_lines.append("-" * 120)
        input_ids = encoded_sample['input_ids']
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        output_lines.append(full_text)
        output_lines.append("-" * 120)
        output_lines.append("")
        
        # Labels 部分 (只有非 -100 的部分)
        output_lines.append("【6. 模型需要学习的部分 (Labels/Target)】")
        output_lines.append("-" * 120)
        labels = encoded_sample['labels']
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0:
            target_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
            output_lines.append(target_text)
        else:
            output_lines.append("(无有效标签)")
        output_lines.append("-" * 120)
        output_lines.append("")
        
        # 统计信息
        output_lines.append("【7. 统计信息】")
        output_lines.append(f"  Total Tokens: {len(input_ids)}")
        output_lines.append(f"  Prompt Tokens (masked): {(labels == -100).sum().item()}")
        output_lines.append(f"  Target Tokens (to learn): {(labels != -100).sum().item()}")
        output_lines.append(f"  Attention Mask Sum: {encoded_sample['attention_mask'].sum().item()}")
        output_lines.append("")
        
        # Token IDs (前50个和后50个)
        output_lines.append("【8. Token IDs 预览】")
        output_lines.append(f"  前50个 tokens: {input_ids[:50].tolist()}")
        output_lines.append(f"  后50个 tokens: {input_ids[-50:].tolist()}")
        output_lines.append("")
        
        # Labels IDs (前50个和后50个)
        output_lines.append("【9. Labels IDs 预览】")
        output_lines.append(f"  前50个 labels: {labels[:50].tolist()}")
        output_lines.append(f"  后50个 labels: {labels[-50:].tolist()}")
        output_lines.append("")
    
    output_lines.append("=" * 120)
    output_lines.append("查看完成")
    output_lines.append("=" * 120)
    
    # 输出结果
    output_text = "\n".join(output_lines)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n✅ 结果已保存到: {args.output_file}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()
