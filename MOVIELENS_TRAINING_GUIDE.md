# MovieLens 训练指南 - 支持历史策略

## 数据格式

MovieLens 数据包含用户的电影评分历史：
- 每个用户有多个电影评分（按时间顺序）
- 每条评分包含：电影名、类型、评分（1.0-5.0）

### 示例数据
```json
{
  "continuation": "5.0",
  "continuation_prefix": "Lilo & Stitch (2002) (Adventure, Animation, Children, Sci-Fi): "
}
```

## 历史策略

MovieLens 支持多种历史划分策略：

### 1. all_previous（默认）
所有之前的评分作为历史

```bash
--history_strategy all_previous
```

**适用场景**：历史较短（<50个评分）

### 2. random（推荐）
随机选择一定比例的之前评分

```bash
--history_strategy random --history_ratio 0.5
```

**优势**：
- ✅ 增加训练多样性
- ✅ 避免模型只记住最近的评分
- ✅ 适合历史很长的情况

### 3. fixed_ratio
固定比例的最近评分

```bash
--history_strategy fixed_ratio --history_ratio 0.3
```

### 4. fixed_count
固定数量的最近评分

```bash
--history_strategy fixed_count --fixed_history_count 10
```

### 5. none
不使用历史（基线对比）

```bash
--history_strategy none
```

## 快速开始

### 推荐配置（random 策略）

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
    --output_dir outputs/MovieLens_history_random_0211 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name history_random_0211 \
    --prompt_style simple
```

## 完整命令示例

### 1. Random 历史（50%，推荐）
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
    --output_dir outputs/MovieLens_history_random_50_0211 \
    --max_epochs 50 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name history_random_50
```

### 2. All Previous（全部历史）
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --history_strategy all_previous \
    --output_dir outputs/MovieLens_history_all_0211 \
    --max_epochs 50 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name history_all
```

### 3. Fixed Count（最近10个）
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --history_strategy fixed_count \
    --fixed_history_count 10 \
    --output_dir outputs/MovieLens_history_count10_0211 \
    --max_epochs 50 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name history_count10
```

### 4. No History（基线）
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --history_strategy none \
    --output_dir outputs/MovieLens_no_history_0211 \
    --max_epochs 50 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name no_history
```

## 输入格式示例

### 使用 random 策略（ratio=0.5）

```
[TASK]
基于用户在 MovieLens 上的历史评分和标签数据，模拟该用户的电影偏好和行为模式

[HISTORICAL_RATINGS]
1. Movie A (2020) (Action): 5.0
2. Movie C (2021) (Comedy): 4.5
3. Movie E (2022) (Drama): 3.0
...

[RECENT_DIALOGUE]
User: Movie Z (2023) (Thriller):

预测用户对该电影的评分：
```

## 对比实验

建议运行以下对比实验：

```bash
# 实验1: 基线（无历史）
bash run_movielens_none.sh

# 实验2: 全部历史
bash run_movielens_all.sh

# 实验3: 随机50%
bash run_movielens_random_50.sh

# 实验4: 随机30%
bash run_movielens_random_30.sh

# 实验5: 固定10个
bash run_movielens_count_10.sh
```

## 验证训练输入

检查训练日志确认历史信息被正确包含：

```bash
cat outputs/MovieLens_history_random_0211/training_logs/detailed_training_log.txt | head -100
```

应该看到：
- ✅ `[HISTORICAL_RATINGS]` 部分
- ✅ 历史评分列表
- ✅ 输入长度合理（不会过长或过短）

## 历史策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| all_previous | 完整信息 | 可能过长 | 历史短（<50） |
| random | 多样性高 | 随机性 | 历史长，想增加泛化 |
| fixed_ratio | 平衡 | 可能忽略早期偏好 | 中等长度历史 |
| fixed_count | 可控长度 | 信息有限 | 需要固定输入长度 |
| none | 基线对比 | 无历史信息 | 对比实验 |

## 推荐配置

基于 MovieLens 数据特点（用户通常有100-1000个评分）：

1. **首选**: `random` + `ratio=0.5` - 平衡多样性和信息量
2. **次选**: `fixed_count` + `count=20` - 固定最近20个评分
3. **基线**: `none` - 用于对比实验

## 注意事项

1. **历史格式**: MovieLens 历史已经包含电影名和评分，无需额外处理
2. **显存占用**: random 策略的历史长度不固定，建议监控显存
3. **训练时间**: 历史越长，训练越慢，random 策略可以平衡
4. **评分预测**: 模型输出应该是 1.0-5.0 之间的评分

---

**立即开始训练**:
```bash
torchrun --nproc_per_node=8 train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --history_strategy random \
    --history_ratio 0.5 \
    --output_dir outputs/MovieLens_history_0211_0 \
    --max_epochs 50 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name history_0211_0
```
