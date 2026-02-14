#!/bin/bash

set -euo pipefail

# ===== 配置 =====
# 工作目录
WORK_DIR="/mnt/parallel/CompactSubset_experiement"
# 生成一个唯一的 JOB_ID 用于日志文件命名
JOB_ID="$(date +%Y%m%d_%H%M%S)_$$"

# ===== 创建日志目录 =====
mkdir -p logs

echo "===== JOB INFO ====="
echo "Date: $(date)"
echo "User: $USER"
echo "JobID: ${JOB_ID}"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "将依次运行 8 个训练任务"
echo

# ===== 设置环境变量，让 transformers 只使用 PyTorch 后端 =====
export USE_TF=0
export USE_TORCH=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_USE_TF=0
echo "已设置环境变量: USE_TF=0, USE_TORCH=1, TRANSFORMERS_NO_TF=1, TRANSFORMERS_USE_TF=0"
echo

# ===== 验证环境 =====
echo "===== 验证环境 ====="
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "警告: PyTorch 未安装或无法导入"

# 验证 huggingface-hub 版本
echo "检查 huggingface-hub..."
HF_HUB_CHECK=$(python -c "import huggingface_hub; print(f'huggingface-hub version: {huggingface_hub.__version__}'); print(f'Location: {huggingface_hub.__file__}')" 2>&1)
if [ $? -eq 0 ]; then
    echo "$HF_HUB_CHECK"
    echo "✓ huggingface-hub 版本正确"
else
    echo "⚠ huggingface-hub 检查失败"
    echo "$HF_HUB_CHECK"
fi
echo

# ===== 检查 GPU =====
echo "===== GPU 信息 ====="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo

# ===== 切换到工作目录 =====
echo "===== 切换到工作目录 ====="
echo "工作目录: $WORK_DIR"
cd "$WORK_DIR"

echo "Python 路径: $(which python || which python3 || echo 'not found')"
echo "torchrun 路径: $(which torchrun || echo 'not found')"
echo

# ===== 定义推理函数 =====
run_inference() {
    local task_name=$1
    local master_port=$2
    local checkpoint_dir=$3
    local dataset=$4
    local ablation_config=$5
    local inference_output_dir=$6
    
    echo "=========================================="
    echo "开始推理任务: $task_name"
    echo "时间: $(date)"
    echo "Master Port: $master_port"
    echo "Checkpoint: $checkpoint_dir"
    echo "Dataset: $dataset"
    echo "=========================================="
    
    # 创建任务特定的日志文件
    local task_log_dir="logs/task_${task_name}_${JOB_ID}"
    mkdir -p "$task_log_dir"
    
    # 检查 checkpoint 目录是否存在
    if [ ! -d "$checkpoint_dir" ]; then
        echo "错误: Checkpoint 目录不存在: $checkpoint_dir"
        exit 1
    fi
    
    # 运行推理
    if PYTHONNOUSERSITE=1 \
        USE_TF=0 \
        USE_TORCH=1 \
        TRANSFORMERS_NO_TF=1 \
        torchrun \
        --nproc_per_node=8 \
        --master_port="$master_port" \
        inference_distributed.py \
        --checkpoint_dir "$checkpoint_dir" \
        --dataset "$dataset" \
        --ablation_config "$ablation_config" \
        --num_samples 5 \
        --output_dir "$inference_output_dir" \
        --no_detailed_template \
        > "$task_log_dir/inference_stdout.log" 2> "$task_log_dir/inference_stderr.log"; then
        echo "✓ 推理任务 $task_name 完成"
        echo "完成时间: $(date)"
        
        # 检查推理输出目录是否存在且不为空
        if [ -d "$inference_output_dir" ] && [ "$(ls -A $inference_output_dir 2>/dev/null)" ]; then
            echo "✓ 推理结果已生成: $inference_output_dir"
            return 0
        else
            echo "错误: 推理输出目录为空或不存在: $inference_output_dir"
            exit 1
        fi
    else
        local exit_code=$?
        echo "✗ 推理任务 $task_name 失败，退出码: $exit_code"
        echo "失败时间: $(date)"
        echo "查看日志: $task_log_dir/inference_stderr.log"
        exit $exit_code
    fi
    echo
}

# ===== 定义删除模型函数 =====
delete_model() {
    local checkpoint_dir=$1
    local task_name=$2
    
    echo "=========================================="
    echo "删除模型: $task_name"
    echo "Checkpoint 目录: $checkpoint_dir"
    echo "时间: $(date)"
    echo "=========================================="
    
    if [ -d "$checkpoint_dir" ]; then
        # 删除 checkpoint 目录
        rm -rf "$checkpoint_dir"
        echo "✓ 模型已删除: $checkpoint_dir"
    else
        echo "警告: Checkpoint 目录不存在: $checkpoint_dir"
    fi
    echo
}

# ===== 定义训练函数 =====
run_training() {
    local task_name=$1
    local master_port=$2
    shift 2
    local train_args=("$@")
    
    echo "=========================================="
    echo "开始训练任务: $task_name"
    echo "时间: $(date)"
    echo "Master Port: $master_port"
    echo "=========================================="
    
    # 创建任务特定的日志文件
    local task_log_dir="logs/task_${task_name}_${JOB_ID}"
    mkdir -p "$task_log_dir"
    
    # 运行训练（使用 set -e，失败会自动退出）
    PYTHONNOUSERSITE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TRANSFORMERS_NO_TF=1 \
    torchrun \
        --nproc_per_node=8 \
        --master_port="$master_port" \
        "${train_args[@]}" \
        > "$task_log_dir/train_stdout.log" 2> "$task_log_dir/train_stderr.log"
    
    # 如果到这里说明训练成功
    echo "✓ 训练任务 $task_name 完成"
    echo "完成时间: $(date)"
    echo
}

# ===== 依次运行 8 个训练任务 =====

# 任务 1: Chameleons
run_training "Chameleons" 29502 \
    train_distributed_Chameleons.py \
    --config config_Chameleons_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/Chameleons_Gemma_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Gemma3_27B-Chameleons \
    --wandb_run_name context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "Chameleons" 29502 \
    outputs/Chameleons_Gemma_context_sampled_seed42 \
    Chameleons \
    context_only \
    outputs/leaderboards/Chameleons_Gemma_context_sampled_seed42

delete_model outputs/Chameleons_Gemma_context_sampled_seed42 "Chameleons"

# 任务 2: DMSC
run_training "DMSC" 29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_Gemma_one_per_user_0213 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Gemma3_27B-DMSC \
    --wandb_run_name one_per_user_0213 \
    --prompt_style simple \
    --one_sample_per_user

# 推理和删除模型
run_inference "DMSC" 29505 \
    outputs/DMSC_Gemma_one_per_user_0213 \
    DMSC \
    profile_and_history \
    outputs/leaderboards/DMSC_Gemma_one_per_user_0213

delete_model outputs/DMSC_Gemma_one_per_user_0213 "DMSC"

# 任务 3: LovinkDialogue
run_training "LovinkDialogue" 29510 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_Gemma_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Gemma3_27B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "LovinkDialogue" 29510 \
    outputs/LovinkDialogue_Gemma_profile_context_sampled_seed42 \
    LovinkDialogue \
    profile_and_context \
    outputs/leaderboards/LovinkDialogue_Gemma_profile_context_sampled_seed42

delete_model outputs/LovinkDialogue_Gemma_profile_context_sampled_seed42 "LovinkDialogue"

# 任务 4: LovinkQuestionnaire
run_training "LovinkQuestionnaire" 29515 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config history_only \
    --output_dir outputs/LovinkQuestionnaire_Gemma_history_random_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --history_strategy random \
    --history_ratio 0.9 \
    --wandb_project Gemma3_27B-LovinkQuestionnaire \
    --wandb_run_name history_random_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "LovinkQuestionnaire" 29515 \
    outputs/LovinkQuestionnaire_Gemma_history_random_sampled_seed42 \
    LovinkQuestionnaire \
    history_only \
    outputs/leaderboards/LovinkQuestionnaire_Gemma_history_random_sampled_seed42

delete_model outputs/LovinkQuestionnaire_Gemma_history_random_sampled_seed42 "LovinkQuestionnaire"

# 任务 5: MovieLens
run_training "MovieLens" 29520 \
    train_distributed_MovieLens.py \
    --config config_MovieLens_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config history_only \
    --output_dir outputs/MovieLens_Gemma_history_random_targets_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --history_strategy random_targets \
    --wandb_project Gemma3_27B-MovieLens \
    --wandb_run_name history_random_targets_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "MovieLens" 29520 \
    outputs/MovieLens_Gemma_history_random_targets_seed42 \
    MovieLens \
    history_only \
    outputs/leaderboards/MovieLens_Gemma_history_random_targets_seed42

delete_model outputs/MovieLens_Gemma_history_random_targets_seed42 "MovieLens"

# 任务 6: PERSONA_Bench
run_training "PERSONA_Bench" 29525 \
    train_distributed_PERSONA_Bench.py \
    --config config_PERSONA_Bench_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config history_and_context \
    --output_dir outputs/PERSONA_Bench_Gemma_history_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Gemma3_27B-PERSONA_Bench \
    --wandb_run_name history_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "PERSONA_Bench" 29525 \
    outputs/PERSONA_Bench_Gemma_history_context_sampled_seed42 \
    PERSONA_Bench \
    history_and_context \
    outputs/leaderboards/PERSONA_Bench_Gemma_history_context_sampled_seed42

delete_model outputs/PERSONA_Bench_Gemma_history_context_sampled_seed42 "PERSONA_Bench"

# 任务 7: RealPersonaChat
run_training "RealPersonaChat" 29530 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/RealPersonaChat_Gemma_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Gemma3_27B-RealPersonaChat \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "RealPersonaChat" 29530 \
    outputs/RealPersonaChat_Gemma_profile_context_sampled_seed42 \
    RealPersonaChat \
    profile_and_context \
    outputs/leaderboards/RealPersonaChat_Gemma_profile_context_sampled_seed42

delete_model outputs/RealPersonaChat_Gemma_profile_context_sampled_seed42 "RealPersonaChat"

# 任务 8: REALTALK
run_training "REALTALK" 29535 \
    train_distributed_REALTALK.py \
    --config config_REALTALK_Gemma.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/REALTALK_Gemma_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Gemma3_27B-REALTALK \
    --wandb_run_name context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "REALTALK" 29535 \
    outputs/REALTALK_Gemma_context_sampled_seed42 \
    REALTALK \
    context_only \
    outputs/leaderboards/REALTALK_Gemma_context_sampled_seed42

delete_model outputs/REALTALK_Gemma_context_sampled_seed42 "REALTALK"

echo "=========================================="
echo "===== 所有任务完成（训练+推理+清理） ====="
echo "完成时间: $(date)"
echo "=========================================="
