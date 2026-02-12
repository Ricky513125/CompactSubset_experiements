# Lovink问卷训练 - 快速开始

## 🎯 核心问题

**Lovink问卷数据没有时间顺序，如何确定哪些问题作为"先验知识"？**

答案：需要**实验性地尝试**不同的历史策略。

## 🚀 快速开始（3步）

### 步骤1: 准备数据

将问卷数据保存为JSON：

```json
[{
  "user": {"profile": {"name": "user_id"}},
  "task": {
    "description": "模拟用户回答风格",
    "task_behavior_collections": [{
      "type": "dialogue",
      "data": [
        {
          "context": [{"source": "questionnaire", "content": "问题内容"}],
          "continuation": "用户回答"
        }
      ]
    }]
  }
}]
```

### 步骤2: 测试历史策略

```bash
# 快速测试所有策略（只训练1个epoch）
bash test_history_strategies.sh your_data.json
```

这会测试5种历史策略，帮你快速找到最佳策略。

### 步骤3: 正式训练

使用测试中效果最好的策略：

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy fixed_ratio \
    --history_ratio 0.5 \
    --output_dir outputs/Lovink_final \
    --max_epochs 50
```

## 📊 5种历史策略对比

| 策略 | 说明 | 何时使用 | 命令 |
|------|------|----------|------|
| **all_previous** | 使用所有之前的问答 | 问卷有隐含顺序 | `--history_strategy all_previous` |
| **fixed_ratio** ⭐ | 前N%问题作为先验 | **最推荐**，灵活可调 | `--history_strategy fixed_ratio --history_ratio 0.5` |
| **fixed_count** | 固定N个问答 | 限制先验数量 | `--history_strategy fixed_count --fixed_history_count 5` |
| **random** | 随机选择 | 测试鲁棒性 | `--history_strategy random` |
| **none** | 不使用历史 | Baseline | `--history_strategy none` |

## 💡 推荐策略

### 方案A：fixed_ratio（最灵活）

```bash
# 测试不同比例
for ratio in 0.3 0.5 0.7; do
    torchrun --nproc_per_node=8 train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config profile_and_history \
        --history_strategy fixed_ratio \
        --history_ratio $ratio \
        --output_dir outputs/Lovink_ratio_$ratio \
        --max_epochs 50
done
```

**选择标准**：
- ratio=0.3: 先验少，适合问题独立性强的问卷
- ratio=0.5: **推荐起点**
- ratio=0.7: 先验多，适合问题关联性强的问卷

### 方案B：位置划分（模拟真实场景）

```bash
torchrun --nproc_per_node=8 train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy all_previous \
    --use_position_split \
    --train_question_ratio 0.7 \
    --output_dir outputs/Lovink_position_split \
    --max_epochs 50
```

**特点**：前70%问题训练，后30%问题测试

## 🧪 完整实验流程

### 实验1：探索最佳历史策略

```bash
# 一键测试所有策略
bash test_history_strategies.sh your_data.json

# 或手动运行每个策略
strategies=("all_previous" "fixed_ratio" "fixed_count" "random" "none")
for strategy in "${strategies[@]}"; do
    torchrun --nproc_per_node=8 train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config profile_and_history \
        --history_strategy $strategy \
        --output_dir outputs/Lovink_strategy_$strategy \
        --max_epochs 10
done
```

### 实验2：微调最佳策略参数

假设实验1发现 `fixed_ratio` 最好：

```bash
# 测试不同比例
for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    torchrun --nproc_per_node=8 train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config profile_and_history \
        --history_strategy fixed_ratio \
        --history_ratio $ratio \
        --output_dir outputs/Lovink_tuning_ratio_$ratio \
        --max_epochs 30
done
```

### 实验3：完整消融实验

使用最佳策略（假设 ratio=0.5）：

```bash
ablations=("profile_and_history_and_context" "profile_and_history" "profile_only" "history_only" "context_only")
for config in "${ablations[@]}"; do
    torchrun --nproc_per_node=8 train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config $config \
        --history_strategy fixed_ratio \
        --history_ratio 0.5 \
        --output_dir outputs/Lovink_final_$config \
        --max_epochs 50
done
```

## 📈 如何判断策略好坏？

查看训练输出目录中的：

1. **training_samples_log.txt** - 查看历史是否合理
   ```bash
   cat outputs/Lovink_xxx/training_samples_log.txt
   ```

2. **验证集loss** - 越低越好
   ```bash
   tensorboard --logdir outputs/Lovink_xxx
   ```

3. **W&B监控** - 对比不同策略的曲线
   ```bash
   # 训练时添加W&B参数
   --wandb_project Lovink-Questionnaire \
   --wandb_run_name strategy_xxx
   ```

## ⚙️ 关键参数说明

### 必需参数
```bash
--config config_LovinkQuestionnaire.json  # 配置文件
--ablation_config profile_and_history     # 消融配置
```

### 历史策略参数
```bash
--history_strategy fixed_ratio            # 历史策略
--history_ratio 0.5                       # 历史比例（仅fixed_ratio）
--fixed_history_count 5                   # 固定数量（仅fixed_count）
```

### 数据划分
```bash
--use_position_split                      # 使用位置划分
--train_question_ratio 0.7                # 训练集问题比例
--val_ratio 0.1                           # 验证集比例（随机划分时）
```

### 训练参数
```bash
--max_epochs 50                           # 训练轮次
--output_dir outputs/xxx                  # 输出目录
--deepspeed ds_config_zero2.json          # DeepSpeed配置
```

## 🔍 与DMSC/影评的区别

| 特性 | DMSC/影评 | **Lovink问卷** |
|------|-----------|----------------|
| 数据顺序 | 有时间顺序 | ❌ **无固定顺序** |
| 历史划分 | 按时间累积 | ✅ **需要实验选择** |
| 最佳策略 | all_previous | ✅ **fixed_ratio 0.5** |
| 数据划分 | 时间顺序 | ✅ **随机或位置** |

## 💡 实践建议

### 新手推荐流程

```bash
# Step 1: 快速测试（10分钟）
bash test_history_strategies.sh your_data.json

# Step 2: 选择策略并训练（根据Step 1结果）
torchrun --nproc_per_node=8 train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy fixed_ratio \
    --history_ratio 0.5 \
    --output_dir outputs/Lovink_main \
    --max_epochs 50 \
    --wandb_project Lovink
```

### 进阶用户流程

```bash
# 完整的策略探索 + 消融实验
# 1. 探索历史策略（预计1-2小时/策略）
# 2. 微调参数（预计2-3小时）
# 3. 完整消融（预计8-10小时）
```

## 🐛 常见问题

### Q: 我该选哪个历史策略？

**A**: 先运行 `bash test_history_strategies.sh`，看哪个策略的样本最合理。通常推荐 **fixed_ratio** with **ratio=0.5**。

### Q: history_ratio设多少合适？

**A**: 
- 0.3-0.4: 问题独立性强
- 0.5-0.6: **推荐**，平衡性好
- 0.7-0.8: 问题关联性强

### Q: 位置划分 vs 随机划分？

**A**:
- **位置划分**: 更贴近真实（前面→后面）
- **随机划分**: 训练/测试更平衡
- 建议：**两种都试试**

### Q: 训练很慢怎么办？

**A**:
```bash
# 使用DeepSpeed加速
--deepspeed ds_config_zero2.json

# 减少batch size
# 在config文件中: "batch_size": 1

# 减少序列长度
# 在config文件中: "max_length": 8192
```

## 📚 完整文档

- **README_LovinkQuestionnaire.md** - 详细说明
- **test_history_strategies.sh** - 快速测试脚本
- **config_LovinkQuestionnaire.json** - 配置文件

## ✅ 检查清单

训练前确认：

- [ ] 数据格式正确（运行 `python data_loader_lovink_questionnaire.py your_data.json`）
- [ ] 配置文件中路径正确
- [ ] 已测试历史策略（`bash test_history_strategies.sh`）
- [ ] GPU可用（`nvidia-smi`）
- [ ] 选择了合适的历史策略和参数

训练后检查：

- [ ] 查看样本日志（`training_samples_log.txt`）
- [ ] 对比不同策略的loss
- [ ] 评估生成质量
- [ ] 确定最佳配置

## 🎉 开始训练！

推荐命令（复制即用）：

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy fixed_ratio \
    --history_ratio 0.5 \
    --output_dir outputs/Lovink_$(date +%m%d_%H%M) \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --wandb_project Lovink-Questionnaire \
    --wandb_run_name exp_$(date +%m%d_%H%M)
```

祝训练顺利！🚀
