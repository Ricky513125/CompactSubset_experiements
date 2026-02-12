#!/bin/bash

# 修复环境依赖问题
# 用法: bash fix_env_dependencies.sh

echo "================================"
echo "修复 lingyu 环境依赖"
echo "================================"

# ===== 配置 =====
CONDA_BASE="/mnt/parallel/lingyuli_miniconda3"
ENV_NAME="lingyu"

# ===== 激活conda =====
echo "激活conda..."
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ===== 检查当前版本 =====
echo ""
echo "检查当前版本..."
echo "---"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>&1 || echo "Transformers: 导入失败"
python -c "import huggingface_hub; print(f'HuggingFace Hub: {huggingface_hub.__version__}')" 2>&1 || echo "HuggingFace Hub: 未安装"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1 || echo "PyTorch: 未安装"
echo "---"

# ===== 诊断问题 =====
echo ""
echo "诊断问题..."
if ! python -c "import transformers" 2>/dev/null; then
    echo "❌ Transformers 导入失败（版本冲突）"
    
    # 检查 huggingface-hub 版本
    HF_HUB_VER=$(python -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "未安装")
    echo "当前 huggingface-hub 版本: $HF_HUB_VER"
    echo "要求: >=1.3.0,<2.0"
    
    # 比较版本
    if [[ "$HF_HUB_VER" == "0."* ]]; then
        echo "诊断结果: huggingface-hub 版本太旧（$HF_HUB_VER < 1.3.0）"
        FIX_ACTION="upgrade"
    elif [[ "$HF_HUB_VER" =~ ^[2-9]\. ]]; then
        echo "诊断结果: huggingface-hub 版本太新（$HF_HUB_VER >= 2.0）"
        FIX_ACTION="downgrade"
    else
        echo "诊断结果: 版本冲突（可能是其他依赖问题）"
        FIX_ACTION="reinstall"
    fi
else
    echo "✓ Transformers 可以正常导入，无需修复"
    exit 0
fi

# ===== 修复 =====
echo ""
echo "开始修复..."
echo "修复方案: $FIX_ACTION"
echo ""

if [[ "$FIX_ACTION" == "upgrade" ]]; then
    echo "升级 huggingface-hub 到兼容版本..."
    pip install "huggingface-hub>=1.3.0,<2.0" --upgrade
    
elif [[ "$FIX_ACTION" == "downgrade" ]]; then
    echo "降级 huggingface-hub 到兼容版本..."
    pip install "huggingface-hub>=1.3.0,<2.0" --force-reinstall
    
elif [[ "$FIX_ACTION" == "reinstall" ]]; then
    echo "重新安装 transformers 和 huggingface-hub..."
    pip uninstall -y transformers huggingface-hub
    pip install transformers huggingface-hub
fi

# ===== 验证修复结果 =====
echo ""
echo "验证修复结果..."
echo "---"

if python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" 2>&1; then
    python -c "import huggingface_hub; print(f'✓ HuggingFace Hub: {huggingface_hub.__version__}')"
    python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
    echo "---"
    echo ""
    echo "================================"
    echo "✓ 修复成功！"
    echo "================================"
    echo ""
    echo "下一步："
    echo "1. 重新打包环境: bash pack_lingyu_env.sh"
    echo "2. 提交作业: sbatch train_lovink_questionnaire.sbatch"
else
    echo "---"
    echo ""
    echo "================================"
    echo "❌ 修复失败"
    echo "================================"
    echo ""
    echo "请尝试手动修复："
    echo "  conda activate $ENV_NAME"
    echo "  pip uninstall -y transformers huggingface-hub"
    echo "  pip install transformers"
    echo "  pip install \"huggingface-hub>=1.3.0,<2.0\""
    echo ""
    echo "或者重新创建环境："
    echo "  conda create -n ${ENV_NAME}_new python=3.10"
    echo "  conda activate ${ENV_NAME}_new"
    echo "  pip install torch transformers deepspeed accelerate"
    exit 1
fi
