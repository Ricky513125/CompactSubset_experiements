#!/bin/bash
#SBATCH --job-name=gemma_train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4              # 申请 4 张 GPU（根据你的集群调整，可能是 gpu:h200:4）
#SBATCH --time=24:00:00           # 最长运行 24 小时（根据实际需要调整）
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ===== 配置 =====
# 打包的环境文件路径（放在共享存储上）
ENV_TAR="/mnt/parallel/slurm_try/lingyu_env.tar.gz"
# 工作目录
WORK_DIR="/mnt/parallel/Qwen_30B_try"
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
"$TMP_DIR/$ENV_NAME/bin/python" -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')" || echo "警告: accelerate 未安装或无法导入"
echo

# ===== 检查 GPU =====
echo "===== GPU 信息 ====="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo
echo "注意: 在 Slurm 中，CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "如果使用 --gpu_ids 参数，请确保与 CUDA_VISIBLE_DEVICES 的映射关系正确"
echo "（例如：如果 CUDA_VISIBLE_DEVICES=0,1,2,3，则 --gpu_ids 应该使用 0,1,2,3 而不是 4,5,6,7）"
echo

# ===== 切换到工作目录并执行训练 =====
echo "===== 开始训练 ====="
echo "工作目录: $WORK_DIR"
cd "$WORK_DIR"

# 将打包环境的 bin 目录添加到 PATH，确保能找到 python3.10 等
export PATH="$TMP_DIR/$ENV_NAME/bin:$PATH"
echo "PATH 已更新，Python 路径: $(which python3.10 || which python || echo 'not found')"

# ===== 自动检测实际分配到的 GPU 数量 =====
# 从 CUDA_VISIBLE_DEVICES 或通过 nvidia-smi 检测
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
        # 默认使用 4 张（与 #SBATCH --gres=gpu:4 保持一致）
        GPU_COUNT=4
        echo "无法自动检测 GPU 数量，使用默认值: $GPU_COUNT"
    fi
fi

# 使用打包环境中的 python 和 accelerate
# 注意：在 Slurm 环境中，不要使用 --gpu_ids 参数
# Slurm 已经通过 CUDA_VISIBLE_DEVICES 限制了可见的 GPU
# Accelerate 会自动使用所有可见的 GPU，num_processes 应该等于实际 GPU 数量
echo "使用 $GPU_COUNT 个进程启动训练（对应 $GPU_COUNT 张 GPU）"
"$TMP_DIR/$ENV_NAME/bin/accelerate" launch --num_processes=$GPU_COUNT train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_context \
    --output_dir outputs/0128_Qwen3_30B_LovinkDialogue_profile_and_context_1100

echo
echo "===== 训练完成 ====="
echo "Date: $(date)"
