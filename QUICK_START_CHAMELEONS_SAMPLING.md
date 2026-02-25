# Chameleons 数据集采样 - 快速开始指南

## 🎯 问题解答

### Q: 为什么我的 7207 个数据项变成了 69977 个训练样本？

**A**: Chameleons 数据集的结构导致的：

```
7,207 个用户 (user_hash)
    ↓ 每个用户有多个对话
72,471 个 data_item (训练样本)
```

- 您之前的采样是在 **user_hash 级别**（每个用户最多3个数据项）
- 但每个数据项内部仍包含 ~9.8 个 data_item（训练样本）
- 所以 7,207 个用户 × 9.8 = ~70,000 个训练样本

## ✅ 解决方案：使用 data_item 级别采样

### 1. 创建采样数据集（推荐配置）

```bash
# 快速实验（~17K 样本，训练时间 23%）
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di3.json \
    --max_data_items 3 \
    --seed 42

# 中等规模（~39K 样本，训练时间 54%）
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di5.json \
    --max_data_items 5 \
    --seed 42
```

### 2. 训练

```bash
# 方式 1：使用快速测试脚本
bash train_Chameleons_di3_test.sh

# 方式 2：直接运行 torchrun
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_Chameleons.py \
    --config config_Chameleons_30B_di3.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/Chameleons_context_30B_di3 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-Chameleons \
    --wandb_run_name context_di3_seed42 \
    --prompt_style simple
```

## 📊 采样结果对比

| 采样方式 | 文件名 | 文件大小 | 训练样本数 | 减少比例 | 训练速度提升 |
|---------|--------|---------|-----------|---------|------------|
| **原始数据** | train.json | 110M | 72,471 | 0% | 1x |
| ❌ user_hash(3) | train_3.json | 108M | 70,480 | 2.7% | 1.03x |
| ✅ data_item(3) | train_di3.json | 24M | 16,963 | 76.6% | 4.3x |
| ✅ data_item(5) | train_di5.json | 40M | ~39,000 | 46.2% | 1.9x |

## 📝 验证采样结果

```bash
# 检查生成的文件
ls -lh sampled_data/Chameleons/

# 验证训练样本数
python3 << 'PYEOF'
import json
from data_loader import extract_training_samples

# 加载采样数据
with open('sampled_data/Chameleons/train_di3.json') as f:
    data = json.load(f)

# 提取训练样本
samples = extract_training_samples(data, debug=True)

print(f"\n最终训练样本数: {len(samples)}")
PYEOF
```

预期输出：
```
开始提取训练样本，总数据项数: 7797
==================================================
提取完成！有效样本总数: 16963

最终训练样本数: 16963
```

## 🎓 重要概念

### 数据项 vs 训练样本

- **数据项 (user_hash 级别)**: 一个用户的所有对话数据
  - 例如：用户 abc123 的所有电影对话

- **data_item (训练样本)**: 一个具体的 context + continuation 对
  - 例如：前面5轮对话 → 预测用户的下一句话

### 两种采样方式

1. **sample_dataset.py** (user_hash 级别)
   - 每个用户最多保留 N 个**数据项**
   - 对 Chameleons 效果不明显（只减少 2.7%）
   - 适用于: MovieLens, DMSC 等数据集

2. **sample_dataset_data_item_level.py** (data_item 级别) ✅
   - 每个用户最多保留 N 个**训练样本**
   - 对 Chameleons 效果显著（减少 76.6%）
   - 适用于: Chameleons 等多对话数据集

## 🚀 推荐工作流

### 第一次训练（快速验证）

```bash
# 1. 创建小规模采样数据集
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di3.json \
    --max_data_items 3 --seed 42

# 2. 快速训练测试
bash train_Chameleons_di3_test.sh
```

### 正式训练

```bash
# 1. 创建中等规模数据集
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di5.json \
    --max_data_items 5 --seed 42

# 2. 正式训练
torchrun --nproc_per_node=8 --master_port=29502 \
    train_distributed_Chameleons.py \
    --config config_Chameleons_30B_di5.json \
    ...
```

## 📚 相关文档

- `CHAMELEONS_SAMPLING_EXPLAINED.md`: 详细的采样原理说明
- `SAMPLING_GUIDE.md`: 通用采样工具指南
- `sample_dataset_data_item_level.py`: data_item 级别采样脚本
- `sample_dataset.py`: user_hash 级别采样脚本

## 💡 常见问题

**Q: 为什么不直接使用 train_3.json？**  
A: train_3.json 是 user_hash 级别采样，只减少了 2.7% 的训练样本，效果不明显。

**Q: 使用 train_di3.json 会影响模型性能吗？**  
A: 每个用户仍保留 3 个训练样本，模型可以学习到用户的基本行为模式。如果担心，可以使用 train_di5.json 或 train_di10.json。

**Q: 如何选择 max_data_items 的值？**  
A: 
- `3`: 快速实验（17K 样本）
- `5`: 标准训练（39K 样本）✅ 推荐
- `10`: 完整训练（~70K 样本）

**Q: 采样会影响 validation 吗？**  
A: 不会。训练脚本会从采样后的数据中按 `--val_ratio 0.1` 划分验证集。

## 🎉 总结

使用 **data_item 级别采样** (`sample_dataset_data_item_level.py`) 可以：

✅ 将训练样本从 72K 减少到 17K（减少 76.6%）  
✅ 训练速度提升 4.3 倍  
✅ 显存占用显著降低  
✅ 快速验证模型和参数  
✅ 精确控制训练样本数量  

**推荐配置**: `train_di3.json` (快速实验) 或 `train_di5.json` (标准训练)
