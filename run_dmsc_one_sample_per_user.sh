#!/bin/bash

# 豆瓣影评训练脚本 - 每用户一个样本模式
# 
# 功能：使用前 n-1 条影评预测第 n 条
# 大幅减少训练数据量和训练时间

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
