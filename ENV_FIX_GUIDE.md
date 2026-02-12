# 环境依赖修复指南

## 问题诊断

你遇到的错误是：
```
ImportError: huggingface-hub>=1.3.0,<2.0 is required for a normal functioning of this module, but found huggingface-hub==0.35.3.
```

**原因**：`huggingface-hub` 版本太旧（0.35.3 < 1.3.0）

## 🔧 快速修复

### 方法1：使用自动修复脚本（推荐）

```bash
bash fix_env_dependencies.sh
```

这个脚本会：
1. ✓ 诊断具体问题
2. ✓ 自动修复版本冲突
3. ✓ 验证修复结果

### 方法2：手动修复

```bash
# 1. 激活环境
source /mnt/parallel/lingyuli_miniconda3/etc/profile.d/conda.sh
conda activate lingyu

# 2. 升级 huggingface-hub
pip install "huggingface-hub>=1.3.0,<2.0" --upgrade

# 3. 验证
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import huggingface_hub; print(f'✓ HuggingFace Hub: {huggingface_hub.__version__}')"
```

## 📦 修复后重新打包

修复完成后，重新打包环境：

```bash
bash pack_lingyu_env.sh
```

## ✅ 完整流程

```bash
# Step 1: 修复依赖
bash fix_env_dependencies.sh

# Step 2: 重新打包环境
bash pack_lingyu_env.sh

# Step 3: 提交作业
sbatch train_lovink_questionnaire.sbatch
```

## 🔍 验证环境

在打包前验证环境是否正常：

```bash
source /mnt/parallel/lingyuli_miniconda3/etc/profile.d/conda.sh
conda activate lingyu

python -c "
import torch
import transformers
import huggingface_hub
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'HuggingFace Hub: {huggingface_hub.__version__}')
print('✓ All packages imported successfully!')
"
```

## 📊 常见版本组合

推荐的版本组合：

| Package | Version |
|---------|---------|
| transformers | >= 4.30.0 |
| huggingface-hub | >= 1.3.0, < 2.0 |
| torch | >= 2.0.0 |
| deepspeed | >= 0.10.0 |

## 🐛 如果修复失败

### 选项1：重新安装 transformers

```bash
conda activate lingyu
pip uninstall -y transformers huggingface-hub
pip install transformers
```

### 选项2：使用conda安装

```bash
conda activate lingyu
conda install -c conda-forge transformers huggingface_hub
```

### 选项3：重新创建环境

如果上述方法都不行，考虑重新创建环境：

```bash
# 1. 创建新环境
conda create -n lingyu_new python=3.10 -y
conda activate lingyu_new

# 2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.30.0
pip install "huggingface-hub>=1.3.0,<2.0"
pip install deepspeed accelerate
pip install wandb tensorboard

# 3. 验证
python -c "import transformers; print('✓ OK')"

# 4. 打包新环境
# 修改 pack_lingyu_env.sh 中的 ENV_NAME="lingyu_new"
bash pack_lingyu_env.sh
```

## 💡 预防措施

在未来打包环境前，始终运行验证：

```bash
bash fix_env_dependencies.sh
```

这会自动检测并修复版本冲突。

## 📞 获取帮助

如果问题仍然存在，检查：

1. **Python版本**
   ```bash
   python --version  # 应该是 3.10.x
   ```

2. **pip版本**
   ```bash
   pip --version
   ```

3. **完整的包列表**
   ```bash
   conda list | grep -E "transformers|huggingface|torch"
   ```

将这些信息发给技术支持。
