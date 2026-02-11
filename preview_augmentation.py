#!/usr/bin/env python3
"""
快速预览时序数据扩充效果
使用方法: python preview_augmentation.py --config config_DMSC.json
"""

import json
import argparse
import sys

from data_loader_more_data import load_train_data, extract_training_samples
from data_augmentation_temporal import expand_samples_with_temporal_history, print_augmentation_stats

def main():
    parser = argparse.ArgumentParser(description='预览时序数据扩充效果')
    parser.add_argument('--config', type=str, default='config_DMSC.json', help='配置文件路径')
    parser.add_argument('--min_history_length', type=int, default=1, help='最小历史长度')
    parser.add_argument('--max_samples_per_user', type=int, default=None, help='每用户最大样本数')
    parser.add_argument('--show_examples', type=int, default=5, help='显示多少个样本示例')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载数据
    train_path = config['data']['train_path']
    print(f"加载训练数据: {train_path}")
    train_data = load_train_data(train_path)
    
    # 提取样本
    print("提取训练样本...")
    original_samples = extract_training_samples(train_data, debug=False)
    print(f"✅ 原始样本数: {len(original_samples)}\n")
    
    # 扩充样本
    print("=" * 80)
    print("开始时序数据扩充...")
    print("=" * 80)
    
    expanded_samples = expand_samples_with_temporal_history(
        original_samples,
        min_history_length=args.min_history_length,
        max_samples_per_user=args.max_samples_per_user,
        verbose=True
    )
    
    # 打印统计信息
    print_augmentation_stats(expanded_samples)
    
    # 显示样本示例
    print("=" * 80)
    print(f"样本示例（前 {args.show_examples} 个）")
    print("=" * 80)
    
    for i in range(min(args.show_examples, len(expanded_samples))):
        sample = expanded_samples[i]
        history = sample.get('history', [])
        target = sample.get('next_question', '')
        user_hash = sample.get('user_hash', 'unknown')
        
        print(f"\n样本 #{i+1}")
        print(f"  用户: {user_hash[:12]}...")
        print(f"  历史长度: {len(history)}")
        
        if history:
            print(f"  历史预览:")
            for j, h in enumerate(history[:3], 1):  # 只显示前3个
                print(f"    {j}. {h[:80]}{'...' if len(h) > 80 else ''}")
            if len(history) > 3:
                print(f"    ... (还有 {len(history) - 3} 条历史)")
        else:
            print(f"  历史: 无")
        
        print(f"  预测目标: {target[:80]}{'...' if len(target) > 80 else ''}")
        
        # 如果有扩充元数据，显示它
        if '_augmentation_meta' in sample:
            meta = sample['_augmentation_meta']
            print(f"  扩充信息: 原始索引={meta['original_index']}, 用户总样本数={meta['user_total_samples']}")
    
    print("\n" + "=" * 80)
    print("预览完成")
    print("=" * 80)
    print(f"\n✅ 扩充倍数: {len(expanded_samples) / len(original_samples):.2f}x")
    print(f"   {len(original_samples)} -> {len(expanded_samples)} 样本")
    print("\n💡 使用以下命令开始训练:")
    print(f"   bash run_dmsc_with_augmentation.sh")
    print(f"\n   或者:")
    print(f"   torchrun --nproc_per_node=8 train_distributed_MovieReview.py \\")
    print(f"       --config {args.config} \\")
    print(f"       --ablation_config history_only \\")
    print(f"       --enable_temporal_augmentation \\")
    print(f"       --min_history_length {args.min_history_length}")
    if args.max_samples_per_user:
        print(f"       --max_samples_per_user {args.max_samples_per_user}")

if __name__ == "__main__":
    main()
