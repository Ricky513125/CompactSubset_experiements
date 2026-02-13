# DMSC 训练 - 每用户一个样本模式

## 🎯 核心思想

**问题**：原始模式会生成大量训练样本，导致训练时间过长。

**解决方案**：每个用户只生成**1个训练样本**，使用该用户的前 n-1 条影评作为历史，预测第 n 条。

---

## 📊 数据对比

### 原始模式（默认）

```
用户 A 有 100 条影评 (r1, r2, ..., r100)

生成 100 个训练样本：
  样本 1:  [] → r1
  样本 2:  [r1] → r2
  样本 3:  [r1, r2] → r3
  ...
  样本 100: [r1, r2, ..., r99] → r100

假设 100 个用户 → 总样本数 = 10,000
```

### 每用户一个样本模式（新）

```
用户 A 有 100 条影评 (r1, r2, ..., r100)

生成 1 个训练样本：
  样本 1: [r1, r2, ..., r99] → r100

假设 100 个用户 → 总样本数 = 100 ✅ 减少 100 倍！
```

---

## 🚀 性能提升

### 训练时间对比

假设 DMSC 数据：
- 100 个用户
- 平均每用户 100 条影评
- 总影评数：10,000

| 模式 | 样本数 | 训练步数 | 预估时间 |
|------|--------|----------|----------|
| **原始模式** | 10,000 | ~156 steps/epoch | ~4 小时/epoch |
| **每用户一个样本** | 100 | ~2 steps/epoch | ~5 分钟/epoch ✅ |

**提升**：训练时间减少 **~50倍**！

### 显存使用

每用户一个样本模式的样本更长（包含完整历史），但：
- 样本数减少 100 倍
- Batch size 仍然是 1
- **显存压力反而更小**（因为总样本数少）

---

## 💡 优势

1. **训练时间大幅缩短**
   - 从几小时 → 几分钟
   - 快速迭代实验

2. **每个样本质量更高**
   - 包含用户的**完整历史**
   - 更好地捕捉用户风格演变

3. **更适合长序列**
   - 充分利用 16K max_length
   - 每个样本都是"完整"的用户历史

4. **减少过拟合风险**
   - 样本数少，不会反复学习同一用户的早期影评

---

## 📝 使用方法

### 命令行参数

```bash
# 启用每用户一个样本模式
--one_sample_per_user
```

### 完整训练命令

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_one_per_user \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name one_per_user \
    --prompt_style simple \
    --one_sample_per_user  # 🔥 启用每用户一个样本模式
```

### 或使用脚本

```bash
./run_dmsc_one_per_user.sh
```

---

## 🔍 样本格式

### 生成的样本结构

```python
{
    'user_profile': {...},
    'user_hash': 'user_13162',
    'task_description': '...',
    
    # 前 n-1 条影评作为历史
    'history': [
        {'movie': '钢铁侠1', 'review': 'boring', 'timestamp': '2008-09-09'},
        {'movie': '复仇者联盟', 'review': 'Again！', 'timestamp': '2012-06-01'},
        ...
        # 共 99 条（假设用户有 100 条影评）
    ],
    
    # 第 n 条影评（要预测的）
    'movie_name': '美国队长3',
    'next_question': '前面铺垫时间太长了',
    'timestamp': '2016-05-06',
    
    # 元数据
    'total_reviews': 100,      # 用户总影评数
    'history_count': 99,       # 历史数量
}
```

### Prompt 示例

```
[USER_HASH=user_13162]

[USER_PROFILE]
[USER_NAME=user_13162]

[TASK]
基于用户在 MovieLens 上的历史评分和标签数据，模拟该用户的电影偏好和行为模式

[HISTORY]
1. 电影《钢铁侠1》: boring
2. 电影《复仇者联盟》: Again！Again！Again！
3. 电影《泰囧》: 好笑又有启发性，难得的国产电影
...
99. 电影《疯狂动物城》: 还不赶紧结婚？！

模仿用户风格为电影《美国队长3》写一条影评：
```

---

## ⚙️ 配置建议

### 推荐配置（16K 长度）

**`config_DMSC_30B.json`**:
```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 2,
    "max_length": 16384,  // 16K，容纳完整历史
    "max_context_turns": 15
  }
}
```

**DeepSpeed**:
```bash
--deepspeed ds_config_zero3_optimized.json
```
（已启用 `cpu_checkpointing: true` 和 `number_checkpoints: 8`）

### 如果 16K 仍然 OOM

#### 选项 1: 减少 max_length

```json
{
  "training": {
    "max_length": 8192  // 8K
  }
}
```

Prompt 构建时会自动截断历史（保留最近的）。

#### 选项 2: 使用 Ulysses 序列并行

```bash
--deepspeed ds_config_zero3_ulysses.json
```

---

## 📈 训练监控

### 查看样本预览

```bash
cat outputs/DMSC_one_per_user/training_samples_preview.txt
```

**关键指标**：
- `历史影评: X条` - 应该接近用户的总影评数 - 1
- `编码长度: X tokens` - 查看是否超过 max_length

### 训练进度

```bash
tail -f outputs/DMSC_one_per_user/training_logs/detailed_training_log.txt
```

**预期**：
- 每个 epoch 只有很少的步数（例如 2-10 步）
- 每步时间可能稍长（因为序列长）
- 但总训练时间大幅缩短

---

## 🔬 消融配置

### 推荐配置

```bash
# 1. Profile + History（推荐）
--ablation_config profile_and_history
```

包含用户信息 + 完整历史，最能体现"累积历史"的优势。

### 其他配置

```bash
# 2. History Only
--ablation_config history_only

# 3. Profile Only（不推荐，丢失历史信息）
--ablation_config profile_only
```

---

## 🆚 与采样模式的区别

| 特性 | 每用户一个样本 | 每用户采样N个 |
|------|---------------|--------------|
| **样本生成** | 用前n-1条 → 第n条 | 随机选N条，每条用之前的作为历史 |
| **样本数** | 用户数 × 1 | 用户数 × N |
| **历史长度** | 最长（完整历史） | 中等（取决于采样位置） |
| **训练时间** | 最短 | 中等 |
| **适用场景** | 快速实验 | 平衡训练质量和速度 |

### 对比示例

100 个用户，每用户 100 条影评：

```bash
# 模式 1: 每用户一个样本
--one_sample_per_user
# 样本数: 100
# 训练时间: ~5 分钟/epoch

# 模式 2: 每用户采样 5 个
--max_samples_per_user 5
# 样本数: 500
# 训练时间: ~25 分钟/epoch

# 模式 3: 完整数据
# （不加任何采样参数）
# 样本数: 10,000
# 训练时间: ~4 小时/epoch
```

---

## 💡 最佳实践

### 推荐工作流程

```bash
# Step 1: 快速验证（每用户一个样本）
./run_dmsc_one_per_user.sh
# → 5 分钟，验证代码和配置

# Step 2: 中等规模（每用户采样 5 个）
torchrun ... --max_samples_per_user 5
# → 25 分钟，获得更好的效果

# Step 3: 完整训练（如果需要）
torchrun ... 
# （不加采样参数）
# → 4 小时，最佳效果
```

### 超参数调优

使用每用户一个样本模式快速测试：
- Learning rate
- Gradient accumulation steps
- Max length
- 消融配置

验证后再用完整数据训练。

---

## ⚠️ 注意事项

### 1. 数据分布

每用户一个样本模式：
- ✅ 每个用户贡献**相同数量**的样本（1个）
- ✅ 避免活跃用户主导训练
- ⚠️ 但每个样本的长度差异很大

### 2. 验证集划分

```python
# train_distributed_MovieReview.py 会自动处理
val_ratio = 0.1  # 10% 的用户用于验证
```

验证集也是每用户一个样本。

### 3. 用户影评数要求

```python
if len(reviews) < 2:
    # 跳过该用户（至少需要 2 条影评）
    continue
```

至少需要 2 条影评（1条历史 + 1条预测）。

---

## 📊 实际效果预估

### DMSC 数据集（假设）

- 总用户数：200
- 平均每用户影评数：50

### 训练配置

```json
{
  "batch_size": 1,
  "gradient_accumulation_steps": 2,
  "max_length": 16384,
  "max_epochs": 50
}
```

### 时间估算

```
样本数: 200 (每用户1个)
每个epoch步数: 200 / (1 × 8 × 2) = ~13 steps
每步时间: ~20秒（长序列）
每个epoch时间: 13 × 20 = ~4.3 分钟

50 epochs: 4.3 × 50 = ~215 分钟 = ~3.6 小时 ✅
```

对比原始模式（10,000 样本）：
```
50 epochs × 4 小时 = 200 小时 ❌
```

**提升**：200小时 → 3.6小时，缩短 **55倍**！

---

## 🎉 总结

**每用户一个样本模式**是为了：
1. ✅ **大幅缩短训练时间**（50+ 倍）
2. ✅ **充分利用用户完整历史**
3. ✅ **快速迭代实验**
4. ✅ **更好的样本质量**（每个样本都是"完整"的）

**立即开始**：
```bash
./run_dmsc_one_per_user.sh
```

祝训练顺利！🚀
