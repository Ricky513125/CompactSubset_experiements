#!/bin/bash
#SBATCH --job-name=all_30B_train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:8              # 申请 8 张 GPU（根据你的集群调整，可能是 gpu:h200:8）
#SBATCH --time=200:00:00           # 最长运行 200 小时（8个任务，每个约24小时）
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ===== 配置 =====
# 打包的环境文件路径（放在共享存储上）
ENV_TAR="/mnt/parallel/slurm_try/lingyu_env.tar.gz"
# 工作目录
WORK_DIR="/mnt/parallel/CompactSubset_experiement"
# 环境解压后的名称
ENV_NAME="lingyu"

# ===== 创建日志目录 =====
mkdir -p logs

echo "===== JOB INFO ====="
echo "Date: $(date)"
echo "User: $USER"
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "将依次运行 8 个训练任务"
echo

# ===== 检查环境文件是否存在 =====
if [ ! -f "$ENV_TAR" ]; then
    echo "错误: 环境打包文件不存在: $ENV_TAR"
    echo "请先在登录节点运行: bash pack_env.sh"
    exit 1
fi

# ===== 解包环境到节点本地临时目录 =====
echo "===== 解包 conda 环境 ====="
# 如果 SLURM_TMPDIR 未设置，使用 /tmp
TMP_DIR="${SLURM_TMPDIR:-/tmp}"
echo "解压到: $TMP_DIR/$ENV_NAME"
mkdir -p "$TMP_DIR/$ENV_NAME"
tar -xzf "$ENV_TAR" -C "$TMP_DIR/$ENV_NAME"
echo "解压完成"

# ===== 修复环境前缀 =====
echo "===== 修复环境前缀 ====="
"$TMP_DIR/$ENV_NAME/bin/conda-unpack"
echo "前缀修复完成"

# ===== 设置环境变量，确保优先使用环境内的包 =====
export PYTHONPATH="$TMP_DIR/$ENV_NAME/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PATH="$TMP_DIR/$ENV_NAME/bin:$PATH"
echo "PYTHONPATH 已设置: $PYTHONPATH"
echo

# ===== 在计算节点上升级 huggingface-hub（确保版本正确）=====
echo "===== 升级 huggingface-hub 到正确版本 ====="
# 先卸载旧版本（如果存在）
"$TMP_DIR/$ENV_NAME/bin/pip" uninstall -y huggingface-hub 2>/dev/null || true
# 安装正确版本
"$TMP_DIR/$ENV_NAME/bin/pip" install "huggingface-hub>=1.3.0,<2.0" --upgrade --force-reinstall 2>&1 | tail -10
if [ $? -eq 0 ]; then
    echo "✓ huggingface-hub 升级成功"
    # 验证版本
    HF_VERSION=$("$TMP_DIR/$ENV_NAME/bin/python" -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>&1)
    HF_LOCATION=$("$TMP_DIR/$ENV_NAME/bin/python" -c "import huggingface_hub; print(huggingface_hub.__file__)" 2>&1)
    echo "当前版本: $HF_VERSION"
    echo "安装位置: $HF_LOCATION"
    
    # 检查版本是否满足要求
    if [[ "$HF_VERSION" =~ ^1\.[3-9]\. ]] || [[ "$HF_VERSION" =~ ^2\. ]]; then
        echo "✓ 版本检查通过"
    else
        echo "⚠ 警告: 版本可能不满足要求: $HF_VERSION"
    fi
else
    echo "✗ huggingface-hub 升级失败"
    echo "尝试继续执行，但可能会遇到版本问题"
fi
echo

# ===== 设置环境变量，让 transformers 只使用 PyTorch 后端 =====
# 不卸载 TensorFlow（保留环境完整性），只通过环境变量告诉 transformers 忽略它
export USE_TF=0
export USE_TORCH=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_USE_TF=0
echo "已设置环境变量: USE_TF=0, USE_TORCH=1, TRANSFORMERS_NO_TF=1, TRANSFORMERS_USE_TF=0"
echo "注意: TensorFlow 包保留在环境中，但 transformers 将只使用 PyTorch 后端"
echo

# ===== 验证环境 =====
echo "===== 验证环境 ====="
"$TMP_DIR/$ENV_NAME/bin/python" -c "import sys; print(f'Python: {sys.executable}')"
"$TMP_DIR/$ENV_NAME/bin/python" -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "警告: PyTorch 未安装或无法导入"

# 验证 huggingface-hub 版本
echo "检查 huggingface-hub..."
HF_HUB_CHECK=$("$TMP_DIR/$ENV_NAME/bin/python" -c "import huggingface_hub; print(f'huggingface-hub version: {huggingface_hub.__version__}'); print(f'Location: {huggingface_hub.__file__}')" 2>&1)
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

# PATH 和 PYTHONPATH 已在上面设置
echo "PATH 已更新，Python 路径: $(which python || which python3 || echo 'not found')"
echo "torchrun 路径: $(which torchrun || echo 'not found')"
echo "PYTHONPATH: $PYTHONPATH"
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
    local task_log_dir="logs/task_${task_name}_${SLURM_JOB_ID}"
    mkdir -p "$task_log_dir"
    
    # 检查 checkpoint 目录是否存在
    if [ ! -d "$checkpoint_dir" ]; then
        echo "错误: Checkpoint 目录不存在: $checkpoint_dir"
        exit 1
    fi
    
    # 运行推理
    # 显式设置环境变量，确保子进程也使用正确的 PYTHONPATH
    # 同时设置 PYTHONNOUSERSITE=1 避免用户级包干扰
    # 禁用 TensorFlow 后端，只使用 PyTorch
    if PYTHONPATH="$TMP_DIR/$ENV_NAME/lib/python3.10/site-packages:${PYTHONPATH:-}" \
        PYTHONNOUSERSITE=1 \
        USE_TF=0 \
        USE_TORCH=1 \
        TRANSFORMERS_NO_TF=1 \
        "$TMP_DIR/$ENV_NAME/bin/torchrun" \
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
    local task_log_dir="logs/task_${task_name}_${SLURM_JOB_ID}"
    mkdir -p "$task_log_dir"
    
    # 运行训练（使用 set -e，失败会自动退出）
    # 由于 huggingface-hub 已经在解包后升级，直接运行训练脚本即可
    # 显式设置环境变量，确保子进程也使用正确的 PYTHONPATH
    # 同时设置 PYTHONNOUSERSITE=1 避免用户级包干扰
    # 禁用 TensorFlow 后端，只使用 PyTorch
    PYTHONPATH="$TMP_DIR/$ENV_NAME/lib/python3.10/site-packages:${PYTHONPATH:-}" \
    PYTHONNOUSERSITE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TRANSFORMERS_NO_TF=1 \
    "$TMP_DIR/$ENV_NAME/bin/torchrun" \
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
    --config config_Chameleons_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/Chameleons_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-Chameleons \
    --wandb_run_name context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "Chameleons" 29502 \
    outputs/Chameleons_context_sampled_seed42 \
    Chameleons \
    context_only \
    outputs/leaderboards/Chameleons_context_sampled_seed42

delete_model outputs/Chameleons_context_sampled_seed42 "Chameleons"

# 任务 2: DMSC
run_training "DMSC" 29505 \
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

# 推理和删除模型
run_inference "DMSC" 29505 \
    outputs/DMSC_one_per_user_0213 \
    DMSC \
    profile_and_history \
    outputs/leaderboards/DMSC_one_per_user_0213

delete_model outputs/DMSC_one_per_user_0213 "DMSC"

# 任务 3: LovinkDialogue
run_training "LovinkDialogue" 29510 \
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

# 推理和删除模型
run_inference "LovinkDialogue" 29510 \
    outputs/LovinkDialogue_profile_context_sampled_seed42 \
    LovinkDialogue \
    profile_and_context \
    outputs/leaderboards/LovinkDialogue_profile_context_sampled_seed42

delete_model outputs/LovinkDialogue_profile_context_sampled_seed42 "LovinkDialogue"

# 任务 4: LovinkQuestionnaire
run_training "LovinkQuestionnaire" 29515 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config history_only \
    --output_dir outputs/LovinkQuestionnaire_history_random_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --history_strategy random \
    --history_ratio 0.9 \
    --wandb_project Qwen3_30B-LovinkQuestionnaire \
    --wandb_run_name history_random_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "LovinkQuestionnaire" 29515 \
    outputs/LovinkQuestionnaire_history_random_sampled_seed42 \
    LovinkQuestionnaire \
    history_only \
    outputs/leaderboards/LovinkQuestionnaire_history_random_sampled_seed42

delete_model outputs/LovinkQuestionnaire_history_random_sampled_seed42 "LovinkQuestionnaire"

# 任务 5: MovieLens
run_training "MovieLens" 29520 \
    train_distributed_MovieLens.py \
    --config config_MovieLens_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config history_only \
    --output_dir outputs/MovieLens_history_random_targets_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --history_strategy random_targets \
    --wandb_project Qwen3_30B-MovieLens \
    --wandb_run_name history_random_targets_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "MovieLens" 29520 \
    outputs/MovieLens_history_random_targets_seed42 \
    MovieLens \
    history_only \
    outputs/leaderboards/MovieLens_history_random_targets_seed42

delete_model outputs/MovieLens_history_random_targets_seed42 "MovieLens"

# 任务 6: PERSONA_Bench
run_training "PERSONA_Bench" 29525 \
    train_distributed_PERSONA_Bench.py \
    --config config_PERSONA_Bench_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config history_and_context \
    --output_dir outputs/PERSONA_Bench_history_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-PERSONA_Bench \
    --wandb_run_name history_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "PERSONA_Bench" 29525 \
    outputs/PERSONA_Bench_history_context_sampled_seed42 \
    PERSONA_Bench \
    history_and_context \
    outputs/leaderboards/PERSONA_Bench_history_context_sampled_seed42

delete_model outputs/PERSONA_Bench_history_context_sampled_seed42 "PERSONA_Bench"

# 任务 7: RealPersonaChat
run_training "RealPersonaChat" 29530 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/RealPersonaChat_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-RealPersonaChat \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "RealPersonaChat" 29530 \
    outputs/RealPersonaChat_profile_context_sampled_seed42 \
    RealPersonaChat \
    profile_and_context \
    outputs/leaderboards/RealPersonaChat_profile_context_sampled_seed42

delete_model outputs/RealPersonaChat_profile_context_sampled_seed42 "RealPersonaChat"

# 任务 8: REALTALK
run_training "REALTALK" 29535 \
    train_distributed_REALTALK.py \
    --config config_REALTALK_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/REALTALK_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-REALTALK \
    --wandb_run_name context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42

# 推理和删除模型
run_inference "REALTALK" 29535 \
    outputs/REALTALK_context_sampled_seed42 \
    REALTALK \
    context_only \
    outputs/leaderboards/REALTALK_context_sampled_seed42

delete_model outputs/REALTALK_context_sampled_seed42 "REALTALK"

echo "=========================================="
echo "===== 所有任务完成（训练+推理+清理） ====="
echo "完成时间: $(date)"
echo "=========================================="
