#!/bin/bash

# Lovink Dialogue 训练脚本 - 简洁 Prompt + 每用户采样
# 
# 功能：
# - 使用简洁 Prompt 格式（只预测 continuation）
# - 每用户采样 2 个样本，减少训练时间
# - 不进行数据扩充

torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
