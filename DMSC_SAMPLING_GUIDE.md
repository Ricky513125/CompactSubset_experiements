# DMSC (Movie Review) 30B 训练 - 采样版本

## 快速开始

```bash
# 每个用户采样 2 个影评进行训练
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

## 关键修改

### 1. ✅ 不使用数据扩充
`data_loader_movie_review.py` **已经是不扩充版本**：
- 每条影评 → 1 个训练样本
- 不做任何数据扩充
- 保持时间顺序

### 2. ✅ 添加采样功能
- `--max_samples_per_user 2`：每个用户最多 2 个样本
- `--sample_seed 42`：随机种子，保证可复现

### 3. ✅ 优化配置
`config_DMSC_30B.json` 的优化：
- `gradient_accumulation_steps`: 32 → **8**（减少 4x 通信）
- `max_length`: 16384 → **2048**（减少显存占用）

## 数据流程

### 原始数据
```
用户 A: 100 条影评（按时间排序）
用户 B: 80 条影评
用户 C: 120 条影评
...
总计: N 个用户，M 条影评
```

### 提取样本（不扩充）
```
每条影评 = 1 个样本

样本结构：
{
    'user_hash': 'user_A',
    'history': [之前的影评],  # 时序历史
    'movie_name': '电影名',
    'next_question': '要预测的影评内容'
}

总样本数 = M（与影评数相同）
```

### 采样（如果启用）
```
用户 A: 从 100 条中随机选 2 条
用户 B: 从 80 条中随机选 2 条
用户 C: 从 120 条中随机选 2 条
...

采样后总样本数 = N × 2（每用户 2 条）
```

## 与 Chameleons 的对比

| 特性 | Chameleons | DMSC (Movie Review) |
|------|-----------|---------------------|
| **数据扩充** | ❌ 不扩充 | ❌ 不扩充 |
| **data_loader** | `data_loader.py` | `data_loader_movie_review.py` |
| **样本生成** | 每个 data_item → 1 样本 | 每条影评 → 1 样本 |
| **历史处理** | 通用 `add_history_to_samples` | 专用时序历史 |
| **采样支持** | ✅ 支持 | ✅ 支持 |

## 配置对比

### DMSC 30B（优化后）
```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 2048
  }
}
```

### Chameleons 30B（对比）
```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 1024
  }
}
```

**DMSC 的 max_length 更长**（2048 vs 1024），因为影评文本通常更长。

## 训练时间估算

假设 DMSC 数据：
- 用户数：1,000
- 平均影评数/用户：50
- 总影评数：50,000

### 不采样
```
总样本数: 50,000
训练步数: ~50,000 / (1 × 8) / 8 = ~780 steps
每步时间: ~20秒（估计，取决于序列长度）
总时间: ~4.3 小时
```

### 采样 2 个/用户
```
总样本数: 1,000 × 2 = 2,000
训练步数: ~2,000 / 8 / 8 = ~31 steps
每步时间: ~20秒
总时间: ~10 分钟 ✅
```

## 消融配置说明

### `profile_and_context`（你的命令）
```json
{
  "use_profile": true,   // 用户信息
  "use_history": false,  // 不使用历史影评
  "use_context": true    // 使用对话上下文（如果有）
}
```

**Prompt 格式**：
```
[USER_HASH=xxx]

[USER_PROFILE]
[USER_NAME=张三]
[USER_AGE=25]

[TASK]
预测用户对电影的评分

[MOVIE]
电影名称（类型）

预测用户的影评：
```

### 其他推荐配置

**`profile_and_history`**（推荐）：
```bash
--ablation_config profile_and_history
```
包含历史影评，更好地捕捉用户偏好变化。

**Prompt 格式**：
```
[USER_HASH=xxx]

[USER_PROFILE]
...

[HISTORY]
1. [电影A] 评分：5.0
2. [电影B] 评分：3.0
...

预测用户对该电影的影评：
```

## 监控训练

```bash
# 查看日志
tail -f outputs/DMSC_profile_context_sampled_seed42/training_logs/detailed_training_log.txt

# 查看 WandB
# https://wandb.ai/your-username/Qwen3_30B-DMSC/runs/profile_context_sampled_seed42
```

## 建议的训练流程

```bash
# Step 1: 快速验证（2 样本/用户，~10 分钟）
torchrun ... \
    --max_samples_per_user 2 \
    --sample_seed 42 \
    --ablation_config profile_and_context

# Step 2: 包含历史（2 样本/用户，~10 分钟）
torchrun ... \
    --max_samples_per_user 2 \
    --sample_seed 42 \
    --ablation_config profile_and_history

# Step 3: 更多数据（5 样本/用户，~25 分钟）
torchrun ... \
    --max_samples_per_user 5 \
    --sample_seed 42 \
    --ablation_config profile_and_history

# Step 4: 完整训练（不采样，~4 小时）
torchrun ... \
    --ablation_config profile_and_history
```

## 注意事项

⚠️ **重要**：
1. DMSC 数据已经按时间排序，采样会保持时间信息
2. 每个用户的历史影评也按时间排序
3. 采样是**随机的**，不按时间选择

✅ **优点**：
1. 训练时间极短（10 分钟 vs 4 小时）
2. 快速迭代实验
3. 每个用户都有代表性样本
4. 不做数据扩充，样本质量高

## 完整训练命令（复制即用）

```bash
# 停止当前训练（如果有）
pkill -f train_distributed_MovieReview

# 清除 Python 缓存
cd /mnt/parallel/CompactSubset_experiement
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 开始训练
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

训练完成后，检查输出目录和 WandB 看结果！
