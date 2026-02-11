# DMSC 影评数据集训练说明

## 重要说明

DMSC 数据集使用 `train_distributed_MovieReview.py` 脚本，该脚本的数据加载器 **已经自动按时序生成训练样本**！

### 数据自动扩充机制

`data_loader_movie_review.py` 在加载数据时会：
1. 按时间顺序遍历用户的每条影评
2. 为每条影评创建一个训练样本
3. 自动将之前的所有影评作为历史上下文

**示例**：
- 用户有 5 条影评：[r1, r2, r3, r4, r5]
- 自动生成 5 个训练样本：
  - 样本1: history=[] → predict r1
  - 样本2: history=[r1] → predict r2
  - 样本3: history=[r1, r2] → predict r3
  - 样本4: history=[r1, r2, r3] → predict r4
  - 样本5: history=[r1, r2, r3, r4] → predict r5

**结论**：**无需使用 `--enable_temporal_augmentation` 参数**，数据已经扩充！

## 快速开始

```bash
# 方式1: 使用快速脚本
bash run_dmsc_train.sh

# 方式2: 手动命令
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_0211 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --wandb_project Qwen3-DMSC \
    --wandb_run_name history_0211
```

## 可用的消融配置

DMSC 数据集支持的消融配置（`--ablation_config`）：

- `history_only` - 只使用历史影评（推荐）
- `profile_and_history` - 使用用户画像 + 历史
- `profile_only` - 只使用用户画像
- `baseline` - 基线（不使用额外信息）

## 完整命令示例

### 1. History Only（推荐）
```bash
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history \
    --max_epochs 50 \
    --wandb_project Qwen3-DMSC
```

### 2. Profile + History
```bash
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_profile_history \
    --max_epochs 50 \
    --wandb_project Qwen3-DMSC
```

### 3. Profile Only
```bash
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_only \
    --output_dir outputs/DMSC_profile \
    --max_epochs 50 \
    --wandb_project Qwen3-DMSC
```

## 与其他数据集的区别

### DMSC / MovieReview（影评数据）
- ✅ **已自动扩充**
- 使用：`train_distributed_MovieReview.py`
- 数据加载器：`data_loader_movie_review.py`
- **不需要** `--enable_temporal_augmentation`

### Chameleons / RealPersonaChat（对话数据）
- ❌ **需要手动扩充**
- 使用：`train_distributed_Chameleons.py` / `train_distributed_RealPersonaChat.py`
- 数据加载器：`data_loader_more_data.py`
- **需要** `--enable_temporal_augmentation --min_history_length 1`

## 查看训练效果

```bash
# 查看训练日志
tail -f outputs/DMSC_history_0211/training_logs/train.log

# 查看样本预览
cat outputs/DMSC_history_0211/training_samples_preview.txt

# 查看 WandB
# 访问训练日志中打印的 WandB 链接
```

## 数据统计

运行后会自动显示数据统计，例如：

```
加载训练数据...
处理用户: Y
任务描述: 模仿用户风格写影评
影评总数: 20
  为用户 Y 创建了 20 个训练样本

总共提取了 5000 个训练样本
训练集: 4500 个样本
验证集: 500 个样本
```

## 常见问题

### Q: 为什么不需要 `--enable_temporal_augmentation`？
A: DMSC 的数据加载器已经自动为每条影评创建训练样本，并包含历史信息。

### Q: 如果我想控制历史长度怎么办？
A: 可以修改 `data_loader_movie_review.py` 中的 `extract_movie_review_samples` 函数，限制 `history_reviews` 的长度。

### Q: 可以跳过第一条影评（没有历史的）吗？
A: 可以在数据加载时添加过滤条件，或者在训练参数中设置。

## 推荐配置

基于 DMSC 数据特点：

1. **首选**: `history_only` - 专注于历史影评模式学习
2. **次选**: `profile_and_history` - 结合用户画像
3. **基线**: `baseline` - 用于对比实验

---

**立即开始训练**:
```bash
bash run_dmsc_train.sh
```
