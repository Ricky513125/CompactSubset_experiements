# Data Loader 对比说明

## 两个 Data Loader 的区别

项目中有两个数据加载模块，功能不同：

### 1. `data_loader.py` - 简化版（只预测 continuation）

**特点**：
- ✅ **只预测 continuation**，不进行数据扩充
- ✅ 每个 `data_item` 生成 **1 个样本**
- ✅ 训练样本数 = 原始数据项数
- ✅ 训练速度更快，内存占用更小

**适用场景**：
- Chameleons 数据集（电影对话）
- 数据量较大，不需要额外扩充的场景
- 30B 模型训练（显存受限）

**提取逻辑**：
```python
# 对于每个 data_item:
# context: [turn1, turn2, ..., turnN]
# continuation: "用户的回复"

# 只生成一个样本:
{
    'context': [turn1, turn2, ..., turnN],  # 完整 context
    'next_question': continuation            # 预测 continuation
}
```

**示例**（Chameleons 数据）：
```json
// 原始数据
{
  "context": [
    {"source": "Rick", "content": "I'm in great pain. Please help me."},
    {"source": "Morty", "content": "All right, Rick, I'm going in."}
  ],
  "continuation": "Morty, be careful."
}

// 生成 1 个样本:
// 样本1: context=[Rick, Morty] -> 预测 "Morty, be careful."
```

---

### 2. `data_loader_more_data.py` - 完整版（数据扩充）

**特点**：
- ✅ **数据扩充**：从完整对话中切分出多个训练样本
- ✅ 每个 `data_item` 可能生成 **N 个样本**（N = context中目标用户的发言次数 + 1）
- ✅ 训练样本数 >> 原始数据项数（扩充倍数 2-5x）
- ✅ 更充分利用数据，提升模型性能

**适用场景**：
- LovinkDialogue、RealPersonaChat 等对话数据集
- 数据量较小，需要扩充的场景
- 4B-13B 模型训练（显存充足）

**提取逻辑**：
```python
# 对于每个 data_item:
# context: [turn1, turn2, ..., turnN]
# continuation: "用户的回复"

# 构建完整对话: context + continuation
full_dialogue = context + [continuation]

# 在每个"目标用户"发言的位置切分，生成多个样本:
# 样本1: context=[] -> 预测 turn1（如果turn1是目标用户）
# 样本2: context=[turn1, turn2] -> 预测 turn3（如果turn3是目标用户）
# ...
# 样本N: context=[完整context] -> 预测 continuation
```

**示例**（LovinkDialogue 数据）：
```json
// 原始数据
{
  "context": [
    {"source": "other_user", "content": "How are you?"},  // User
    {"source": "user", "content": "I'm fine."},           // Assistant (目标用户)
    {"source": "other_user", "content": "Good to hear!"}  // User
  ],
  "continuation": "Thank you!"  // Assistant (目标用户)
}

// 生成 2 个样本:
// 样本1: context=["How are you?"] -> 预测 "I'm fine."
// 样本2: context=["How are you?", "I'm fine.", "Good to hear!"] -> 预测 "Thank you!"
```

---

## 对比表格

| 特性 | `data_loader.py` | `data_loader_more_data.py` |
|------|-----------------|---------------------------|
| **样本数量** | 1个/data_item | N个/data_item (N≥1) |
| **数据扩充** | ❌ 否 | ✅ 是 |
| **训练速度** | ✅ 快 | ⚠️ 慢（样本多） |
| **显存占用** | ✅ 小 | ⚠️ 大（样本多） |
| **数据利用率** | ⚠️ 低 | ✅ 高 |
| **适用模型** | 30B | 4B-13B |
| **适用数据集** | Chameleons, PERSONA_Bench | LovinkDialogue, RealPersonaChat |

---

## 如何选择？

### 使用 `data_loader.py`（简化版）的情况：

1. ✅ **30B 大模型**
   ```bash
   # Chameleons 30B
   --config config_Chameleons_30B.json  # 使用 30B 配置
   --deepspeed ds_config_zero3_optimized.json  # Zero-3
   ```

2. ✅ **数据量大**（>10000 samples）
   ```bash
   # Chameleons 有很多对话数据
   ```

3. ✅ **显存受限**
   ```bash
   # 需要减少训练样本数量
   ```

### 使用 `data_loader_more_data.py`（完整版）的情况：

1. ✅ **4B-13B 模型**
   ```bash
   # LovinkDialogue 4B
   --config config_LovinkDialogue.json
   --deepspeed ds_config_zero2.json  # Zero-2 足够
   ```

2. ✅ **数据量小**（<5000 samples）
   ```bash
   # LovinkDialogue, RealPersonaChat 数据较少
   ```

3. ✅ **显存充足**
   ```bash
   # H100 80GB，可以处理更多样本
   ```

---

## 当前配置

### Chameleons 训练

**你的命令**：
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_Chameleons.py \
    --config config_Chameleons.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/Chameleons_context_0213_3 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-Chameleons \
    --wandb_run_name context_0213_3 \
    --prompt_style simple
```

**当前状态**：
- ✅ 已修改 `train_distributed_Chameleons.py` 使用 `data_loader.py`
- ✅ 只预测 continuation，不做数据扩充
- ✅ 适合 30B 模型 + Zero-3

**预期效果**：
- 训练样本数 = 原始数据项数（不扩充）
- 训练速度更快
- 显存占用更小

---

## 技术细节

### `data_loader.py` 的关键代码

```python
# 只创建一个样本：context -> continuation
if len(full_dialogue) > 0 and full_dialogue[-1]['role'] == 'user':
    # context 最后一轮是 user
    samples.append({
        'context': full_dialogue,
        'next_question': continuation
    })
elif len(full_dialogue) == 0:
    # 没有 context，直接预测（首次发言）
    samples.append({
        'context': [],
        'next_question': continuation
    })
```

### `data_loader_more_data.py` 的关键代码

```python
# 切分逻辑：在每个目标用户发言的位置生成样本
full_dialogue = context + [{"role": "assistant", "content": continuation}]

for i in range(len(full_dialogue)):
    if full_dialogue[i]['role'] == 'assistant':  # 目标用户发言
        input_context = full_dialogue[:i]
        target_text = full_dialogue[i]['content']
        
        if target_text and len(input_context) > 0:
            if input_context[-1]['role'] == 'user':
                samples.append({
                    'context': input_context,
                    'next_question': target_text
                })
```

---

## 注意事项

⚠️ **重要**：
1. 两个 data_loader 的 `build_simple_training_prompt` 函数基本相同
2. `DynamicPaddingDataset` 会自动选择使用哪个 prompt 函数
3. 只有 `extract_training_samples` 的逻辑不同

✅ **建议**：
- 30B 模型：使用 `data_loader.py`（已配置）
- 4B-13B 模型：使用 `data_loader_more_data.py`（默认）
- PERSONA_Bench, MovieLens：使用专用的 history 模块

---

## 验证

训练开始后，检查日志中的样本数：

```bash
# data_loader.py（简化版）
提取了 10000 个训练样本  # ≈ 原始数据项数

# data_loader_more_data.py（完整版）
提取了 35000 个训练样本  # >> 原始数据项数（扩充了3.5倍）
```
