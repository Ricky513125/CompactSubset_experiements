#!/bin/bash

# 打包conda环境脚本
# 用法: bash pack_lingyu_env.sh

set -e

echo "================================"
echo "打包 lingyu conda 环境"
echo "================================"

# ===== 配置 =====
CONDA_BASE="/mnt/parallel/lingyuli_miniconda3"
ENV_NAME="lingyu"
OUTPUT_DIR="/mnt/parallel/slurm_try"
OUTPUT_FILE="$OUTPUT_DIR/lingyu_env.tar.gz"

echo "Conda 基础路径: $CONDA_BASE"
echo "环境名称: $ENV_NAME"
echo "输出文件: $OUTPUT_FILE"
echo ""

# ===== 检查conda是否存在 =====
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "错误: conda.sh 不存在: $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

# ===== 激活conda =====
echo "激活conda..."
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ===== 激活指定环境 =====
echo "激活环境: $ENV_NAME"
conda activate "$ENV_NAME"

# ===== 验证环境 =====
echo ""
echo "验证环境..."
echo "Python: $(which python)"
echo "Python 版本: $(python --version)"

# 检查关键包
echo ""
echo "检查关键包..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "⚠️  PyTorch 未安装"

# 检查 transformers 和 huggingface-hub 的兼容性
echo ""
echo "检查 transformers 和 huggingface-hub 兼容性..."
TRANSFORMERS_VER=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "未安装")
HF_HUB_VER=$(python -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "未安装")

echo "Transformers: $TRANSFORMERS_VER"
echo "HuggingFace Hub: $HF_HUB_VER"

# 尝试导入 transformers，如果失败说明版本不兼容
if ! python -c "import transformers" 2>/dev/null; then
    echo ""
    echo "❌ 版本冲突检测！"
    echo "transformers 和 huggingface-hub 版本不兼容"
    echo ""
    echo "正在修复..."
    
    # 方案1: 升级到兼容版本
    pip install "huggingface-hub>=1.3.0,<2.0" --upgrade
    
    # 再次检查
    if python -c "import transformers" 2>/dev/null; then
        echo "✓ 版本冲突已修复"
        HF_HUB_VER=$(python -c "import huggingface_hub; print(huggingface_hub.__version__)")
        echo "新的 HuggingFace Hub 版本: $HF_HUB_VER"
    else
        echo "❌ 修复失败，请手动修复后重新运行"
        echo ""
        echo "建议执行："
        echo "  conda activate $ENV_NAME"
        echo "  pip install \"huggingface-hub>=1.3.0,<2.0\" --force-reinstall"
        echo "  pip install transformers --upgrade"
        exit 1
    fi
else
    echo "✓ Transformers 可以正常导入"
fi

python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" || echo "⚠️  DeepSpeed 未安装"

# ===== 检查conda-pack =====
echo ""
echo "检查 conda-pack..."
if ! conda list | grep -q conda-pack; then
    echo "conda-pack 未安装，正在安装..."
    conda install -y -c conda-forge conda-pack
else
    echo "✓ conda-pack 已安装"
fi

# ===== 创建输出目录 =====
echo ""
echo "创建输出目录..."
mkdir -p "$OUTPUT_DIR"

# ===== 备份旧文件（如果存在） =====
if [ -f "$OUTPUT_FILE" ]; then
    BACKUP_FILE="${OUTPUT_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
    echo "备份旧文件: $OUTPUT_FILE -> $BACKUP_FILE"
    mv "$OUTPUT_FILE" "$BACKUP_FILE"
fi

# ===== 打包环境 =====
echo ""
echo "开始打包环境..."
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

conda-pack -n "$ENV_NAME" -o "$OUTPUT_FILE" --ignore-missing-files

# ===== 检查打包结果 =====
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "================================"
    echo "✓ 环境打包成功！"
    echo "================================"
    echo "文件: $OUTPUT_FILE"
    echo "大小: $FILE_SIZE"
    echo ""
    echo "下一步："
    echo "1. 在sbatch脚本中使用这个环境文件"
    echo "2. 提交作业: sbatch train_lovink_questionnaire.sbatch"
    echo ""
else
    echo ""
    echo "✗ 打包失败！"
    exit 1
fi
