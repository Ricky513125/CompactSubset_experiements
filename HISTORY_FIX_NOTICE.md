# LovinkQuestionnaire 历史信息修复说明

## 问题

之前运行 LovinkQuestionnaire 训练时，虽然在训练日志中显示有历史信息，但**实际输入给模型的文本中完全没有包含历史**！

### 问题表现
- 样本显示有 5-158 条历史记录
- 但实际输入只有 38 tokens
- 输入文本只包含：`系统提示 + 任务描述 + 答案`
- **历史信息完全缺失**

## 原因

`data_loader_more_data.py` 中的 `build_simple_training_prompt` 函数虽然接受 `history` 参数，但从未使用它。

## 修复

已在 `build_simple_training_prompt` 函数中添加历史信息处理逻辑（第 730-753 行）：

```python
# 2.5. HISTORY 部分 - 历史信息（在 TASK 和 RECENT_DIALOGUE 之间）
if use_history and history and len(history) > 0:
    history_parts = ["[HISTORY]"]
    max_history_items = 15  # 最多保留15条历史
    
    for i, item in enumerate(history_to_use, 1):
        content = ...  # 提取历史内容
        history_parts.append(f"{i}. {content}")
    
    system_parts.append("\n".join(history_parts))
```

## 现在的输入格式

修复后，训练输入应该包含：

```
[USER_PROFILE]
...

[TASK]
基于用户在 Lovink 问卷中的回答数据，模拟该用户的回答风格和行为模式

[HISTORY]
1. 问题：xxx 回答：xxx
2. 问题：yyy 回答：yyy
...

[RECENT_DIALOGUE]
User: 当前问题

预测用户的下一条消息:
```

## 重新训练

请重新运行训练命令：

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --ablation_config history_only \
    --history_strategy random \
    --history_ratio 0.5 \
    --output_dir outputs/LovinkQuestionnaire_history_fixed_0211 \
    --max_epochs 50 \
    --wandb_project Qwen3-LovinkQuestionnaire \
    --wandb_run_name history_fixed_0211
```

## 验证修复

训练开始后，检查 `outputs/LovinkQuestionnaire_history_fixed_0211/training_logs/detailed_training_log.txt`：

应该看到：
- ✅ 输入长度显著增加（例如从 38 tokens 增加到 200+ tokens）
- ✅ 输入文本中包含 `[HISTORY]` 部分
- ✅ 历史信息在输入文本中可见

## 受影响的数据集

使用 `data_loader_more_data.py` 的所有数据集都受影响：
- ✅ **已修复**: LovinkQuestionnaire
- ✅ **已修复**: LovinkDialogue  
- ✅ **已修复**: RealPersonaChat
- ✅ **已修复**: Chameleons
- ⚠️ **不受影响**: DMSC/MovieReview（使用不同的数据加载器）

## 建议

1. **停止当前训练**（如果还在运行）
2. **重新训练**所有使用 `history_only` 或 `history_and_context` 配置的实验
3. **检查训练日志**确认历史信息被正确包含
4. **对比效果**：修复前 vs 修复后的模型效果

---

**修复状态**: ✅ 已完成  
**影响范围**: 所有使用 `data_loader_more_data.py` 的对话类数据集  
**修复日期**: 2026-02-11
