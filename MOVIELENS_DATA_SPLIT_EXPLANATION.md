# MovieLens 数据使用说明

## 当前数据使用方式

### 1. 训练数据

✅ **使用 `train.json`**
- 路径: `/mnt/parallel/GIDigitalTwinBench/RealSelf/MovieLens/train.json`
- 10,000 个用户
- 约 1.2GB 数据

### 2. 验证数据  

✅ **从 `train.json` 中划分**
- 使用 `--val_ratio 0.1` (默认10%)
- 划分策略：**用户内划分**
  - 每个用户的样本被随机打乱
  - 90% 用于训练，10% 用于验证
  - **训练集和验证集包含相同的用户**

### 3. 测试数据

❌ **`test.json` 完全没有使用**
- 配置文件中有 `test_path`，但代码从未加载
- test.json 似乎格式有问题或为空

## 当前训练流程

```
train.json (10,000 users)
    ↓
extract_training_samples (提取所有用户的所有评分)
    ↓
add_history_to_samples (添加历史策略)
    ↓
split_train_val (用户内划分，ratio=0.1)
    ├─→ 训练集 (90%)
    └─→ 验证集 (10%)
```

## 这种方式的含义

### ✅ 优点
1. **用户内泛化测试**：评估模型对同一用户新评分的预测能力
2. **数据充分利用**：所有用户都参与训练
3. **简单有效**：无需担心用户分布不平衡

### ⚠️ 局限
1. **无法测试用户间泛化**：不知道模型对新用户的表现
2. **可能过拟合用户特征**：模型可能记住特定用户的偏好
3. **test.json 浪费**：原本准备的测试集未使用

## val_ratio 的作用

`--val_ratio 0.1` 的含义：

```python
# 对每个用户
user_samples = [评分1, 评分2, ..., 评分100]
random.shuffle(user_samples)

# 划分
train = user_samples[:90]  # 前90个
val = user_samples[90:]    # 后10个
```

### 示例

用户 A 有 100 个评分：
- **训练集**: 随机选择90个评分，每个评分的历史来自之前的评分
- **验证集**: 剩余10个评分

用户 B 有 50 个评分：
- **训练集**: 随机选择45个评分
- **验证集**: 剩余5个评分

## 建议的改进方案

### 方案1: 继续使用当前方式（推荐）

**适用场景**: 
- 主要关注用户内的评分预测
- 例如：给定用户A的部分历史，预测其新评分

**命令**（当前）:
```bash
torchrun --nproc_per_node=8 train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --val_ratio 0.1 \
    ...
```

### 方案2: 用户间划分（需要修改代码）

**适用场景**:
- 测试对新用户的泛化能力
- cold-start 问题

**需要实现**:
```python
# 按用户划分
train_users = users[:8000]  # 80% 用户
val_users = users[8000:]     # 20% 用户
```

### 方案3: 时序划分（需要修改代码）

**适用场景**:
- 预测用户未来的评分
- 更符合实际应用场景

**需要实现**:
```python
# 对每个用户，按时间划分
train = 前70%的评分
val = 后30%的评分
```

### 方案4: 使用 test.json（需要修复数据）

**前提**: test.json 格式正确

**需要实现**:
- 修复 test.json 格式
- 添加测试数据加载逻辑
- 区分验证集和测试集

## 当前最佳实践

### 对于 MovieLens 训练

**推荐配置**:
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --history_strategy random \
    --history_ratio 0.5 \
    --val_ratio 0.1 \
    --output_dir outputs/MovieLens_history_0211_0 \
    --max_epochs 50 \
    --wandb_project Qwen3-MovieLens
```

**含义**:
- 使用所有10,000个用户的数据
- 每个用户90%的评分用于训练，10%用于验证
- 历史策略：随机选择50%的之前评分
- 早停基于验证集loss

## 验证集的作用

### 在训练中的作用

1. **早停 (Early Stopping)**
   ```python
   --early_stopping_patience 3
   ```
   - 如果验证集loss连续3个epoch不下降，停止训练
   - 防止过拟合

2. **模型选择**
   - 保存验证集loss最低的checkpoint
   - 作为最终模型

3. **超参数调优**
   - 对比不同history_strategy的效果
   - 对比不同history_ratio的效果

### 不是真正的测试集

⚠️ **注意**: 
- 验证集用户在训练集中出现过
- 只是这些用户的另一部分评分
- 不能评估对新用户的泛化能力

## 总结

### 当前方式

| 项目 | 说明 |
|------|------|
| 训练数据 | train.json (100%) |
| 验证数据 | 从train.json划分 (10%) |
| 测试数据 | ❌ 未使用 |
| 划分方式 | 用户内随机划分 |
| val_ratio | 控制验证集比例（默认0.1） |
| 用户重叠 | ✅ 训练和验证包含相同用户 |

### 评估目标

- ✅ **用户内泛化**: 预测已知用户的新评分
- ❌ **用户间泛化**: 预测新用户的评分

### 建议

1. **继续使用当前方式** - 对于大多数推荐系统场景已经足够
2. **监控训练/验证loss差距** - 判断是否过拟合
3. **使用W&B对比不同配置** - 找到最佳history_strategy
4. **如需用户间泛化** - 需要修改代码实现用户级划分

---

**当前命令已经正确**:
```bash
torchrun --nproc_per_node=8 train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --val_ratio 0.1 \
    ...
```

`val_ratio` 正在被使用，从 train.json 中划分出10%作为验证集！
