# LovinkDialogue 训练 - 不扩充 + 采样模式

## 🎯 核心修改

### 问题
原始 `train_distributed_LovinkDialogue.py` 使用 `data_loader_more_data.py`，会进行数据扩充，生成大量训练样本。

### 解决方案
1. **切换到 `data_loader.py`**：不进行数据扩充
2. **添加采样功能**：每用户最多N个样本

---

## 📊 数据对比

### 原始模式（data_loader_more_data.py）

```
用户 A 的对话：
  context: [turn1, turn2, turn3, turn4, turn5]
  continuation: "用户回复"

会生成多个样本（数据扩充）:
  样本 1: [turn1] → 预测 turn2
  样本 2: [turn1, turn2] → 预测 turn3
  ...
  样本 N: [turn1...turn5] → 预测 continuation

假设 100 个用户，每个 10 条对话 → 可能生成 5000+ 样本
```

### 新模式（data_loader.py + 采样）

```
用户 A 的对话：
  data_item 1: context → continuation 1
  data_item 2: context → continuation 2
  data_item 3: context → continuation 3
  ...

生成样本（不扩充）:
  样本 1: context → continuation 1
  样本 2: context → continuation 2
  样本 3: context → continuation 3

采样（每用户最多2个）:
  样本 1, 样本 2

假设 100 个用户，每用户采样 2 个 → 生成 200 样本 ✅
```

---

## 🚀 性能提升

### 训练时间对比

假设 LovinkDialogue 数据：
- 100 个用户
- 每用户平均 10 个 data_item
- 原始模式可能扩充到 50 个样本/用户

| 模式 | 样本数 | 训练步数/epoch | 预估时间/epoch |
|------|--------|---------------|--------------|
| **原始模式（扩充）** | 5,000 | ~78 | ~2 小时 |
| **不扩充 + 采样2个** | 200 | ~3 | ~5 分钟 ✅ |

**提升**：训练时间缩短约 **24 倍**！

---

## 📝 使用方法

### 命令行参数

```bash
# 必选参数
--max_samples_per_user 2  # 每用户最多2个样本
--sample_seed 42          # 随机种子

# 可选参数（调整采样数量）
--max_samples_per_user 5  # 每用户最多5个样本
```

### 完整训练命令

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

### 或使用脚本

```bash
./run_lovinkdialogue_sampled.sh
```

---

## 🔍 数据流程

### Step 1: 提取样本（不扩充）

```python
# 使用 data_loader.py
from data_loader import extract_training_samples

all_samples = extract_training_samples(train_data)
# 每个 data_item → 1 个样本（不扩充）
```

**数据结构**：
```python
{
    'context': [
        {'role': 'user', 'content': '...'},
        {'role': 'assistant', 'content': '...'},
        ...
    ],
    'next_question': '要预测的用户回复',
    'user_profile': {...},
    'user_hash': 'user_A'
}
```

### Step 2: 采样（可选）

```python
# 使用 sample_per_user
from sample_per_user import sample_per_user

all_samples = sample_per_user(
    all_samples,
    max_samples_per_user=2,
    random_seed=42
)
# 每用户随机选择最多2个样本
```

### Step 3: 添加历史

```python
# 如果启用 use_history
all_samples = add_history_to_samples(all_samples, all_samples)
```

---

## 🆚 与 data_loader_more_data.py 的区别

| 特性 | data_loader.py | data_loader_more_data.py |
|------|---------------|-------------------------|
| **数据扩充** | ❌ 不扩充 | ✅ 扩充 |
| **样本生成** | 1 data_item → 1 样本 | 1 data_item → N 样本 |
| **样本数量** | 少 | 多 |
| **训练时间** | 短 | 长 |
| **适用场景** | 快速实验、避免过拟合 | 完整训练、更多数据 |

### 代码对比

**data_loader.py**（不扩充）:
```python
# extract_training_samples 逻辑
for data_item in collection.get('data', []):
    context = data_item.get('context', [])
    continuation = data_item.get('continuation', '')
    
    # 只创建一个样本：context → continuation
    samples.append({
        'context': full_dialogue,
        'next_question': continuation,
        ...
    })
```

**data_loader_more_data.py**（扩充）:
```python
# extract_training_samples 逻辑
for data_item in collection.get('data', []):
    context = data_item.get('context', [])
    
    # 从 context 中生成多个样本（数据扩充）
    for i in range(len(context)):
        samples.append({
            'context': context[:i],
            'next_question': context[i],
            ...
        })
    
    # 再加上 continuation
    samples.append({
        'context': context,
        'next_question': continuation,
        ...
    })
```

---

## ⚙️ 配置建议

### 快速实验

```bash
--max_samples_per_user 2
--max_epochs 10
```
**时间**: ~5 分钟

### 中等规模

```bash
--max_samples_per_user 5
--max_epochs 30
```
**时间**: ~20 分钟

### 不采样（使用所有数据）

```bash
# 不加 --max_samples_per_user 参数
--max_epochs 50
```
**时间**: 取决于数据量（可能 1-2 小时）

---

## 📈 监控训练

### 查看样本预览

```bash
cat outputs/LovinkDialogue_profile_context_sampled_seed42/training_samples_preview.txt
```

### 实时日志

```bash
tail -f outputs/LovinkDialogue_profile_context_sampled_seed42/training_logs/detailed_training_log.txt
```

### GPU 监控

```bash
watch -n 1 nvidia-smi
```

---

## 🔬 消融配置

### 推荐配置

```bash
# 1. Profile + Context（您的命令）
--ablation_config profile_and_context
```
包含用户信息 + 对话上下文。

### 其他配置

```bash
# 2. Profile + History + Context（完整）
--ablation_config profile_and_history_and_context

# 3. History + Context（不含 Profile）
--ablation_config history_and_context

# 4. Context Only（只用对话上下文）
--ablation_config context_only
```

---

## 💡 与其他数据集的一致性

现在所有数据集都使用相同的策略：

| 数据集 | Data Loader | 扩充？ | 采样？ |
|--------|------------|-------|-------|
| **DMSC** | `data_loader_movie_review.py` | ❌ | ✅ `--max_samples_per_user` 或 `--one_sample_per_user` |
| **Chameleons** | `data_loader.py` | ❌ | ✅ `--max_samples_per_user` |
| **LovinkDialogue** | `data_loader.py` | ❌ | ✅ `--max_samples_per_user` |
| **MovieLens** | `data_loader_movielens_history.py` | ❌ | ✅ `--max_samples_per_user` |
| **PERSONA_Bench** | `data_loader_persona_bench_history.py` | ❌ | ✅ `--max_samples_per_user` |

**统一策略**：
1. ✅ 不进行数据扩充
2. ✅ 支持每用户采样
3. ✅ 快速训练 + 避免过拟合

---

## ⚠️ 注意事项

### 1. 配置文件

确保使用 30B 配置：
```bash
--config config_LovinkDialogue_30B.json
```

如果没有，需要创建（参考 `config_DMSC_30B.json`）。

### 2. DeepSpeed 配置

使用优化后的 ZeRO-3：
```bash
--deepspeed ds_config_zero3_optimized.json
```

### 3. Prompt 风格

使用 `simple` 风格（默认）：
```bash
--prompt_style simple
```

---

## 🎯 预期结果

### 数据量

假设 LovinkDialogue 有 100 个用户：

```
原始数据:
  100 用户 × 10 data_items = 1,000 samples（不扩充）

采样后:
  100 用户 × 2 samples = 200 samples
```

### 训练时间

```
每 epoch 步数: 200 / (1 × 8 × 8) ≈ 3 steps
每步时间: ~20 秒
每 epoch 时间: ~1 分钟

50 epochs: ~50 分钟 ✅
```

---

## 📚 相关文档

- `DATA_LOADER_COMPARISON.md` - data_loader.py vs data_loader_more_data.py
- `CHAMELEONS_SAMPLING_GUIDE.md` - 采样功能详细说明
- `DMSC_ONE_SAMPLE_PER_USER.md` - 每用户一个样本模式

---

## 🚀 立即开始

```bash
# 开始训练
./run_lovinkdialogue_sampled.sh

# 或手动执行
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

祝训练顺利！🚀
