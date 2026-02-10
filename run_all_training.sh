#!/bin/bash
# 训练脚本汇总 - 8个数据集的训练命令
# 使用方法：根据需要选择相应的命令运行

# ============================================================
# 1. IdealSelf/LovinkDialogue
# ============================================================
echo "训练命令 1: LovinkDialogue"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-LovinkDialogue \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 2. IdealSelf/LovinkQuestionnaire
# ============================================================
echo -e "\n训练命令 2: LovinkQuestionnaire"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkQuestionnaire_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-LovinkQuestionnaire \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 3. IdealSelf/RealPersonaChat
# ============================================================
echo -e "\n训练命令 3: RealPersonaChat"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/RealPersonaChat_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-RealPersonaChat \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 4. RealSelf/DMSC
# ============================================================
echo -e "\n训练命令 4: DMSC"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_DMSC.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-DMSC \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 5. RealSelf/MovieLens
# ============================================================
echo -e "\n训练命令 5: MovieLens"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/MovieLens_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 6. RealSelf/Chameleons
# ============================================================
echo -e "\n训练命令 6: Chameleons"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_Chameleons.py \
    --config config_Chameleons.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/Chameleons_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-Chameleons \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 7. RealSelf/PERSONA-Bench
# ============================================================
echo -e "\n训练命令 7: PERSONA-Bench"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_PERSONA_Bench.py \
    --config config_PERSONA_Bench.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/PERSONA_Bench_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-PERSONA-Bench \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

# ============================================================
# 8. RealSelf/REALTALK
# ============================================================
echo -e "\n训练命令 8: REALTALK"
cat << 'EOF'
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_REALTALK.py \
    --config config_REALTALK.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/REALTALK_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-REALTALK \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
EOF

echo -e "\n============================================================"
echo "所有训练命令已列出完成！"
echo "请复制相应的命令到终端执行。"
echo "============================================================"
