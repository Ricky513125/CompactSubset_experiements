# Chameleons 数据集采样说明

## 🔍 问题：为什么 7207 个数据项变成了 69977 个训练样本？

### Chameleons 数据结构

```
数据项 (user_hash 级别)
├── user_hash: "abc123"
├── user: {profile, ...}
└── task:
    └── task_behavior_collections: [
        {
            data: [
                {context: [...], continuation: "..."},  ← 训练样本 1
                {context: [...], continuation: "..."},  ← 训练样本 2
                {context: [...], continuation: "..."},  ← 训练样本 3
                ...                                      ← 更多训练样本
            ]
        }
    ]
```

**关键点**：
- 1 个 **数据项** (user_hash) 包含多个 **data_item**
- 每个 **data_item** = 1 个训练样本（context + continuation）
- 原始数据：7,797 个用户 → 72,471 个 data_item（训练样本）
- 平均每个用户有 **9.3 个训练样本**

## 📊 两种采样方式对比

### 方式 1：user_hash 级别采样 ❌ (不推荐用于 Chameleons)

**使用脚本**: `sample_dataset.py`

```bash
python sample_dataset.py \
    /path/to/train.json \
    /path/to/train_3.json \
    --max_samples 3 \
    --seed 42
```

**采样逻辑**：
- 每个用户（user_hash）最多保留 3 个**数据项**
- 但每个数据项内部仍包含多个 data_item

**结果**（max_samples=3）：
```
原始: 7,797 个用户 → 72,471 个训练样本
↓
采样: 7,207 个用户 → 70,480 个训练样本 (保留 97.3%)
```

**问题**：
- ❌ 只减少了 2.7% 的训练样本（效果不明显）
- ❌ 大多数用户只有 1-2 个数据项，采样几乎没有效果
- ❌ 不能有效控制训练样本数量

---

### 方式 2：data_item 级别采样 ✅ (推荐)

**使用脚本**: `sample_dataset_data_item_level.py`

```bash
python sample_dataset_data_item_level.py \
    /path/to/train.json \
    /path/to/train_di3.json \
    --max_data_items 3 \
    --seed 42
```

**采样逻辑**：
- 每个用户（user_hash）最多保留 3 个 **data_item**（训练样本）
- 直接控制训练样本数量

**结果**（max_data_items=3）：
```
原始: 7,797 个用户 → 72,471 个训练样本
↓
采样: 7,797 个用户 → 16,963 个训练样本 (保留 23.4%)
```

**优点**：
- ✅ 有效减少 76.6% 的训练样本
- ✅ 训练速度提升 3-4 倍
- ✅ 减少过拟合风险
- ✅ 精确控制训练样本数量

## 🚀 推荐配置

### 快速实验（~17K 样本）
```bash
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di3.json \
    --max_data_items 3 \
    --seed 42
```
- **训练样本数**: ~16,963
- **训练时间**: 原始的 ~25%
- **适用场景**: 快速验证、参数调优

### 中等规模（~39K 样本）
```bash
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di5.json \
    --max_data_items 5 \
    --seed 42
```
- **训练样本数**: ~39,000
- **训练时间**: 原始的 ~54%
- **适用场景**: 标准训练

### 完整训练（~72K 样本）
```bash
# 直接使用原始数据，不采样
# 或者使用 --max_data_items 10（几乎覆盖所有样本）
```
- **训练样本数**: ~72,471
- **训练时间**: 100%
- **适用场景**: 最终模型训练

## 📝 使用示例

### 1. 创建采样数据集

```bash
# 推荐：data_item 级别采样（每用户最多3个训练样本）
python sample_dataset_data_item_level.py \
    /mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons/train.json \
    sampled_data/Chameleons/train_di3.json \
    --max_data_items 3 \
    --seed 42
```

### 2. 更新配置文件

创建 `config_Chameleons_30B_di3.json`:

```json
{
  "model": {
    "name": "Qwen3-30B-A3B-Instruct-2507",
    "path": "/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"
  },
  "data": {
    "train_path": "sampled_data/Chameleons/train_di3.json"
  },
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 2,
    "max_length": 1024,
    ...
  }
}
```

### 3. 训练

```bash
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

## 📈 性能对比

| 采样方式 | 用户数 | 训练样本数 | 训练时间 | 显存占用 | 推荐度 |
|---------|-------|-----------|---------|---------|--------|
| **原始数据** | 7,797 | 72,471 | 100% | 高 | ⭐⭐⭐ |
| **user_hash采样(3)** | 7,207 | 70,480 | 97% | 高 | ⭐ |
| **data_item采样(3)** | 7,797 | 16,963 | 23% | 低 | ⭐⭐⭐⭐⭐ |
| **data_item采样(5)** | 7,797 | 39,000 | 54% | 中 | ⭐⭐⭐⭐ |
| **data_item采样(10)** | 7,797 | 70,000+ | 97% | 高 | ⭐⭐⭐ |

## ⚠️ 常见误区

### ❌ 误区 1：以为采样后样本数会显著减少
```bash
# 使用 user_hash 级别采样
python sample_dataset.py ... --max_samples 3

# 结果：7,797 → 7,207 用户，但训练样本从 72,471 → 70,480
# 只减少了 2.7%！
```

### ✅ 正确做法：使用 data_item 级别采样
```bash
# 使用 data_item 级别采样
python sample_dataset_data_item_level.py ... --max_data_items 3

# 结果：训练样本从 72,471 → 16,963
# 减少了 76.6%！
```

## 🎯 总结

对于 **Chameleons 数据集**：

1. **不要使用** `sample_dataset.py`（user_hash 级别采样）
   - 效果不明显（只减少 2.7% 样本）
   - 不能有效控制训练样本数量

2. **推荐使用** `sample_dataset_data_item_level.py`（data_item 级别采样）
   - 精确控制训练样本数量
   - 显著减少训练时间和显存占用
   - `--max_data_items 3`: 快速实验（23% 样本）
   - `--max_data_items 5`: 标准训练（54% 样本）
   - `--max_data_items 10`: 完整训练（97% 样本）

3. **命名规范**：
   - `train_3.json`: user_hash 级别采样（每用户最多3个数据项）
   - `train_di3.json`: data_item 级别采样（每用户最多3个训练样本）✅

这样可以避免混淆，清楚地知道使用的是哪种采样方式。
