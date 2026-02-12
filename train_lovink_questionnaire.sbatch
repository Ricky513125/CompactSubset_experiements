#!/bin/bash
#SBATCH --job-name=lovink_questionnaire
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:8              # 申请 8 张 GPU（根据你的集群调整）
#SBATCH --time=48:00:00           # 最长运行 48 小时（根据实际需要调整）
#SBATCH --output=/mnt/parallel/CompactSubset_experiement/logs/%x_%j.out
#SBATCH --error=/mnt/parallel/CompactSubset_experiement/logs/%x_%j.err

set -euo pipefail

# ===== 配置 =====
# 打包的环境文件路径（放在共享存储上）
ENV_TAR="/mnt/parallel/slurm_try/lingyu_env.tar.gz"
# 工作目录
WORK_DIR="/mnt/parallel/CompactSubset_experiement"
# 环境解压后的名称
ENV_NAME="lingyu"

# ===== 训练参数配置 =====
CONFIG_FILE="config_LovinkQuestionnaire.json"
ABLATION_CONFIG="history_only"
HISTORY_STRATEGY="random"
HISTORY_RATIO=0.5
OUTPUT_DIR="outputs/Lovink_${ABLATION_CONFIG}_${SLURM_JOB_ID}"
WANDB_PROJECT="Qwen3-LovinkQuestionnaire"
WANDB_RUN_NAME="${ABLATION_CONFIG}_${SLURM_JOB_ID}"

# ===== 创建日志目录 =====
mkdir -p "$WORK_DIR/logs"

echo "===== JOB INFO ====="
echo "Date: $(date)"
echo "User: $USER"
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "配置文件: $CONFIG_FILE"
echo "消融配置: $ABLATION_CONFIG"
echo "历史策略: $HISTORY_STRATEGY (ratio=$HISTORY_RATIO)"
echo "输出目录: $OUTPUT_DIR"
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

# ===== 验证环境 =====
echo "===== 验证环境 ====="
"$TMP_DIR/$ENV_NAME/bin/python" -c "import sys; print(f'Python: {sys.executable}')"
"$TMP_DIR/$ENV_NAME/bin/python" -c "import torch; print(f'PyTorch version: {torch.__version__}')"
"$TMP_DIR/$ENV_NAME/bin/python" -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
echo

# ===== 检查 GPU =====
echo "===== GPU 信息 ====="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo

# ===== 自动检测实际分配到的 GPU 数量 =====
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # 从 CUDA_VISIBLE_DEVICES 环境变量计算 GPU 数量
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "从 CUDA_VISIBLE_DEVICES 检测到 $GPU_COUNT 张 GPU: $CUDA_VISIBLE_DEVICES"
else
    # 如果没有 CUDA_VISIBLE_DEVICES，尝试通过 nvidia-smi 检测
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        echo "通过 nvidia-smi 检测到 $GPU_COUNT 张 GPU"
    else
        # 默认使用 8 张（与 #SBATCH --gres=gpu:8 保持一致）
        GPU_COUNT=8
        echo "无法自动检测 GPU 数量，使用默认值: $GPU_COUNT"
    fi
fi

# ===== 切换到工作目录并执行训练 =====
echo "===== 开始训练 ====="
echo "工作目录: $WORK_DIR"
cd "$WORK_DIR"

# 将打包环境的 bin 目录添加到 PATH
export PATH="$TMP_DIR/$ENV_NAME/bin:$PATH"
echo "PATH 已更新，Python 路径: $(which python3.10 || which python || echo 'not found')"

# ===== 设置 W&B（如果需要） =====
# export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_NAME="$WANDB_RUN_NAME"

# ===== 使用 torchrun 启动分布式训练 =====
echo "使用 $GPU_COUNT 个进程启动训练（对应 $GPU_COUNT 张 GPU）"
echo "训练命令: torchrun --nproc_per_node=$GPU_COUNT --master_port=29500 ..."
echo

"$TMP_DIR/$ENV_NAME/bin/torchrun" \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config $CONFIG_FILE \
    --ablation_config $ABLATION_CONFIG \
    --history_strategy $HISTORY_STRATEGY \
    --history_ratio $HISTORY_RATIO \
    --output_dir $OUTPUT_DIR \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --val_ratio 0.1 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --prompt_style simple

EXIT_CODE=$?

echo
echo "===== 训练完成 ====="
echo "Date: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "输出目录: $OUTPUT_DIR"

# ===== 清理临时环境（可选） =====
# echo "清理临时环境..."
# rm -rf "$TMP_DIR/$ENV_NAME"

exit $EXIT_CODE
