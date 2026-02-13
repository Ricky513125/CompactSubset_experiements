# Chameleons 30B 训练 - 每角色采样 2 个对话

## 快速开始

```bash
# 停止当前训练（如果有）
pkill -f train_distributed_Chameleons

# 每个角色采样 2 个对话进行训练
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_Chameleons.py \
    --config config_Chameleons_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config context_only \
    --output_dir outputs/Chameleons_context_0213_sampled \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-Chameleons \
    --wandb_run_name context_0213_sampled \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

## 效果对比

### 不采样（当前）
```
原始样本数: 71,964
训练步数: 195,700 steps
每步时间: ~16秒
总训练时间: 36 天 ❌
```

### 采样后（每角色 2 个样本）
```
原始样本数: 71,964
用户（角色）数: 7,797
采样后样本数: 7,797 × 2 = 15,594
训练步数: ~1,950 steps（减少 100x）
每步时间: ~16秒
总训练时间: ~8.7 小时 ✅
```

## 参数说明

### `--max_samples_per_user N`
每个用户（角色）最多保留 N 个样本

| N | 总样本数 | 训练时间 |
|---|---------|---------|
| 1 | ~7,797 | ~4.3 小时 |
| 2 | ~15,594 | ~8.7 小时 |
| 5 | ~38,985 | ~22 小时 |
| 10 | ~77,970 | ~1.8 天 |
| None | 71,964 | 36 天 |

**推荐**: 
- 快速实验：`--max_samples_per_user 2`
- 完整训练：`--max_samples_per_user 10`

### `--sample_seed N`
随机种子，保证每次采样结果一致（默认：42）

## 每个 Epoch 重新采样

如果想要每个 epoch 采样不同的样本，可以在训练脚本中实现：

```python
for epoch in range(max_epochs):
    # 每个 epoch 使用不同的随机种子
    epoch_samples = sample_per_user(
        all_samples,
        max_samples_per_user=2,
        random_seed=42 + epoch  # 每个 epoch 不同的种子
    )
    # 训练...
```

但当前实现是**整个训练过程使用固定的采样样本**，更简单高效。

## 验证集

- 验证集从**采样后的数据**中划分
- 验证集占比：10%（由 `--val_ratio 0.1` 控制）
- 验证集样本数：~1,559

## 数据分布

采样是**每个角色独立随机采样**：
- 角色 A：从 279 个对话中随机选 2 个
- 角色 B：从 120 个对话中随机选 2 个
- 角色 C：从 350 个对话中随机选 2 个
- ...

保证了：
1. ✅ 每个角色都有样本
2. ✅ 数据多样性（不同角色的不同对话场景）
3. ✅ 大幅减少训练时间

## 监控训练

```bash
# 查看训练日志
tail -f outputs/Chameleons_context_0213_sampled/training_logs/detailed_training_log.txt

# 查看 WandB
# https://wandb.ai/your-username/Qwen3_30B-Chameleons/runs/context_0213_sampled
```

## 注意事项

⚠️ **重要**：
1. 采样会减少数据量，可能影响模型性能
2. 建议先用小的 `max_samples_per_user` 快速验证
3. 如果效果好，再增加采样数量
4. 完整训练建议 `max_samples_per_user >= 10`

✅ **优点**：
1. 训练时间大幅缩短（100x）
2. 每个角色都有样本
3. 可以快速迭代实验
4. 仍然保留数据多样性

## 完整训练流程建议

```bash
# Step 1: 快速验证（2 个样本/角色，~9 小时）
torchrun ... --max_samples_per_user 2 --wandb_run_name quick_test

# Step 2: 中等规模（5 个样本/角色，~22 小时）
torchrun ... --max_samples_per_user 5 --wandb_run_name medium_train

# Step 3: 完整训练（10 个样本/角色，~2 天）
torchrun ... --max_samples_per_user 10 --wandb_run_name full_train

# Step 4: 如果效果好，考虑使用全部数据（36 天）
torchrun ... --wandb_run_name complete_train
```
