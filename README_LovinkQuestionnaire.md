# Lovink问卷训练指南

## 数据特点

Lovink问卷数据是**问答对**格式：
- 每个样本包含一个问题和用户的回答
- 问题之间**没有固定的时间顺序**
- 需要选择合适的策略来划分"先验知识"

## 关键问题：如何处理历史？

与影评数据（有明确时间顺序）不同，问卷数据需要**实验性地决定**哪些问题作为先验知识。

### 历史策略对比

| 策略 | 说明 | 适用场景 | 命令参数 |
|------|------|----------|----------|
| **all_previous** | 使用所有之前的问答 | 数据有隐含顺序 | `--history_strategy all_previous` |
| **fixed_ratio** | 前N%问题作为先验 | 实验不同先验比例 | `--history_strategy fixed_ratio --history_ratio 0.5` |
| **fixed_count** | 固定N个问答作为先验 | 限制先验数量 | `--history_strategy fixed_count --fixed_history_count 5` |
| **random** | 随机选择一部分 | 测试鲁棒性 | `--history_strategy random` |
| **none** | 不使用历史 | Baseline | `--history_strategy none` |

## 训练/测试划分策略

### 策略1：随机划分（默认）

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --output_dir outputs/Lovink_random_split \
    --val_ratio 0.1
```

**特点**：每个用户的问答随机打乱后划分

### 策略2：基于位置划分（推荐实验）

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --output_dir outputs/Lovink_position_split \
    --use_position_split \
    --train_question_ratio 0.7
```

**特点**：
- 前70%的问题用于训练
- 后30%的问题用于测试
- 模拟"学习前面的回答风格，预测后面的回答"

## 完整实验示例

### 实验1：不同历史策略对比

```bash
# 实验1.1: 使用所有历史
torchrun --nproc_per_node=8 --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy all_previous \
    --output_dir outputs/Lovink_exp1_1_all_history \
    --wandb_run_name exp1_1_all_history

# 实验1.2: 前50%作为先验
torchrun --nproc_per_node=8 --master_port=29501 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy fixed_ratio \
    --history_ratio 0.5 \
    --output_dir outputs/Lovink_exp1_2_half_history \
    --wandb_run_name exp1_2_half_history

# 实验1.3: 固定5个问答作为先验
torchrun --nproc_per_node=8 --master_port=29502 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy fixed_count \
    --fixed_history_count 5 \
    --output_dir outputs/Lovink_exp1_3_fixed5_history \
    --wandb_run_name exp1_3_fixed5_history

# 实验1.4: 随机历史
torchrun --nproc_per_node=8 --master_port=29503 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy random \
    --output_dir outputs/Lovink_exp1_4_random_history \
    --wandb_run_name exp1_4_random_history

# 实验1.5: 无历史（Baseline）
torchrun --nproc_per_node=8 --master_port=29504 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_only \
    --history_strategy none \
    --output_dir outputs/Lovink_exp1_5_no_history \
    --wandb_run_name exp1_5_no_history
```

### 实验2：不同先验比例

```bash
for ratio in 0.3 0.5 0.7 0.9; do
    port=$((29500 + $(echo "$ratio * 10" | bc | cut -d. -f1)))
    torchrun --nproc_per_node=8 --master_port=$port \
        train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config profile_and_history \
        --history_strategy fixed_ratio \
        --history_ratio $ratio \
        --output_dir outputs/Lovink_exp2_ratio_${ratio} \
        --wandb_run_name exp2_ratio_${ratio}
done
```

### 实验3：位置划分 vs 随机划分

```bash
# 实验3.1: 随机划分
torchrun --nproc_per_node=8 --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --val_ratio 0.2 \
    --output_dir outputs/Lovink_exp3_1_random_split \
    --wandb_run_name exp3_1_random_split

# 实验3.2: 位置划分（前80%训练，后20%测试）
torchrun --nproc_per_node=8 --master_port=29501 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --use_position_split \
    --train_question_ratio 0.8 \
    --output_dir outputs/Lovink_exp3_2_position_split \
    --wandb_run_name exp3_2_position_split
```

### 实验4：完整消融实验

```bash
# 使用最佳历史策略（假设通过实验1确定）
BEST_HISTORY_STRATEGY="fixed_ratio"
BEST_HISTORY_RATIO=0.5

ablation_configs=("profile_and_history_and_context" "profile_and_history" "profile_and_context" "history_and_context" "profile_only" "history_only" "context_only")

for i in "${!ablation_configs[@]}"; do
    config="${ablation_configs[$i]}"
    port=$((29500 + i))
    
    torchrun --nproc_per_node=8 --master_port=$port \
        train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config $config \
        --history_strategy $BEST_HISTORY_STRATEGY \
        --history_ratio $BEST_HISTORY_RATIO \
        --output_dir outputs/Lovink_exp4_${config} \
        --wandb_run_name exp4_${config} \
        --max_epochs 50
done
```

## 参数说明

### 数据和模型
- `--config`: 配置文件（默认：config_LovinkQuestionnaire.json）
- `--ablation_config`: 消融实验配置（必需）

### 历史策略（核心参数）
- `--history_strategy`: 历史划分策略（默认：all_previous）
  - `all_previous`: 所有之前的问答
  - `fixed_ratio`: 前N%问题作为先验
  - `fixed_count`: 固定N个问答
  - `random`: 随机选择
  - `none`: 不使用历史

- `--history_ratio`: 历史比例（默认：0.5）
  - 仅当 `history_strategy=fixed_ratio` 时有效
  
- `--fixed_history_count`: 固定历史数量
  - 仅当 `history_strategy=fixed_count` 时有效

### 数据划分
- `--use_position_split`: 使用基于位置的划分
- `--train_question_ratio`: 训练集问题比例（默认：0.7）
  - 仅当 `use_position_split=True` 时有效
- `--val_ratio`: 验证集比例（默认：0.1）
  - 仅当 `use_position_split=False` 时有效

### 训练参数
- `--max_epochs`: 最大训练轮次（默认：50）
- `--early_stopping_patience`: 早停耐心值（默认：3）
- `--output_dir`: 输出目录
- `--deepspeed`: DeepSpeed配置文件

### W&B监控
- `--wandb_project`: W&B项目名称（默认：Qwen3-LovinkQuestionnaire）
- `--wandb_run_name`: W&B运行名称

## 推荐实验流程

### 第1步：探索最佳历史策略

运行**实验1**（不同历史策略），观察：
- 训练loss下降速度
- 验证集performance
- 生成的回答质量

### 第2步：微调历史比例

如果 `fixed_ratio` 效果好，运行**实验2**找最佳比例。

### 第3步：测试划分策略

运行**实验3**对比随机划分和位置划分。

### 第4步：完整消融实验

使用最佳策略运行**实验4**（所有消融配置）。

## 数据格式

问卷数据格式：

```json
[{
  "user": {
    "profile": {"name": "user_78901533"}
  },
  "task": {
    "description": "模拟用户回答风格",
    "task_behavior_collections": [{
      "type": "dialogue",
      "data": [
        {
          "context": [{
            "source": "questionnaire",
            "content": "安全的生活环境对我来说很重要..."
          }],
          "continuation": "同意"
        }
      ]
    }]
  }
}]
```

## 输出分析

训练后每个实验会生成：
- `training_config.json`: 包含使用的历史策略
- `training_samples_log.txt`: 样本示例
- `test_samples.json`: 测试集（如果有）

对比不同实验的：
1. 验证集loss
2. 生成质量
3. 训练稳定性

## 常见问题

### Q1: 应该用哪种历史策略？

**建议**：先运行实验1的所有策略，根据你的数据特点选择：
- 如果问卷有隐含顺序 → `all_previous` 或 `fixed_ratio`
- 如果问卷完全随机 → `random` 或 `fixed_count`
- 如果想测试纯粹的profile能力 → `none`

### Q2: history_ratio应该设多少？

**建议**：运行实验2测试0.3、0.5、0.7、0.9，观察：
- 比例太小：先验不足，模型难以学习风格
- 比例太大：训练样本不足
- 通常0.5-0.7效果较好

### Q3: 位置划分 vs 随机划分？

**位置划分**优势：
- 更贴近真实场景（已知前面的回答，预测后面的）
- 可以测试模型的泛化能力

**随机划分**优势：
- 训练/测试更平衡
- 适合问题之间独立的情况

**建议**：两种都试试（实验3）

### Q4: 消融实验选哪些配置？

**最小集合**（快速测试）：
- `profile_and_history`: 完整模型
- `profile_only`: 测试profile作用
- `history_only`: 测试历史作用
- `context_only`: Baseline

**完整集合**（论文级别）：
- 所有7种配置（实验4）

## 示例命令（快速开始）

```bash
# 推荐：使用固定比例历史策略 + 位置划分
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config profile_and_history \
    --history_strategy fixed_ratio \
    --history_ratio 0.5 \
    --use_position_split \
    --train_question_ratio 0.7 \
    --output_dir outputs/Lovink_quick_start \
    --max_epochs 50 \
    --wandb_project Lovink-Questionnaire \
    --wandb_run_name quick_start
```

## 与DMSC/影评训练的区别

| 特性 | DMSC/影评 | Lovink问卷 |
|------|-----------|------------|
| 数据顺序 | 有明确时间顺序 | **无固定顺序** |
| 历史处理 | 按时间累积 | **需要选择策略** |
| 数据划分 | 时间顺序划分 | **随机或位置划分** |
| Context | 多轮对话 | **单个问题** |
| 先验知识 | 明确（之前的对话） | **实验性确定** |

## 结论

Lovink问卷数据训练的**关键**是找到合适的历史策略。建议：

1. **先探索**：运行实验1，对比不同策略
2. **再优化**：运行实验2，微调参数
3. **后评估**：运行实验4，完整消融

祝训练顺利！
