# DMSC 训练 - 快速命令参考

## 🚀 立即开始

### 1. 测试新功能
```bash
cd /mnt/parallel/CompactSubset_experiement
./test_one_sample_per_user.sh
```
查看原始模式 vs 每用户一个样本模式的对比。

### 2. 开始训练（每用户一个样本）
```bash
./run_dmsc_one_per_user.sh
```

---

## 📋 三种训练模式对比

| 模式 | 命令 | 样本数 | 训练时间 |
|------|------|--------|----------|
| **每用户一个样本** | `--one_sample_per_user` | 最少（用户数） | 最短 ⚡ |
| **每用户采样N个** | `--max_samples_per_user 5` | 中等（用户数×N） | 中等 |
| **完整数据** | （默认） | 最多（总影评数） | 最长 🐌 |

---

## 🎯 推荐使用场景

### 场景 1: 快速实验/验证
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_quick_test \
    --max_epochs 10 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name quick_test \
    --prompt_style simple \
    --one_sample_per_user  # 🔥 每用户一个样本
```
**预估时间**: ~5-10 分钟

### 场景 2: 超参数调优
```bash
# 测试不同 learning rate
for lr in 1e-5 5e-6 1e-6; do
    torchrun ... \
        --one_sample_per_user \
        --learning_rate $lr \
        --output_dir outputs/DMSC_lr_${lr}
done
```
每个配置只需要几分钟！

### 场景 3: 消融实验
```bash
# Profile + History
torchrun ... --ablation_config profile_and_history --one_sample_per_user

# Profile Only
torchrun ... --ablation_config profile_only --one_sample_per_user

# History Only
torchrun ... --ablation_config history_only --one_sample_per_user
```
快速对比不同配置的效果。

### 场景 4: 中等规模训练
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_medium \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name medium_training \
    --prompt_style simple \
    --max_samples_per_user 5  # 每用户采样5个
```
**预估时间**: ~30-60 分钟

### 场景 5: 完整训练（最佳效果）
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_full \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name full_training \
    --prompt_style simple
    # 不加任何采样参数 = 使用所有数据
```
**预估时间**: 数小时（取决于数据量）

---

## 🔧 配置调整

### 如果 16K OOM

#### 方案 A: 启用 CPU Checkpointing（已默认）
```json
// ds_config_zero3_optimized.json
{
  "activation_checkpointing": {
    "cpu_checkpointing": true,  // ✅ 已启用
    "number_checkpoints": 8     // ✅ 已设置
  }
}
```

#### 方案 B: 减少 max_length
```json
// config_DMSC_30B.json
{
  "training": {
    "max_length": 8192  // 16384 → 8192
  }
}
```

#### 方案 C: 使用 Ulysses 序列并行
```bash
--deepspeed ds_config_zero3_ulysses.json
```

---

## 📊 监控训练

### 查看样本预览
```bash
cat outputs/DMSC_one_per_user_0213/training_samples_preview.txt
```

### 实时监控日志
```bash
tail -f outputs/DMSC_one_per_user_0213/training_logs/detailed_training_log.txt
```

### 监控 GPU
```bash
watch -n 1 nvidia-smi
```

### WandB 可视化
访问: https://wandb.ai/your-username/Qwen3_30B-DMSC

---

## 💡 最佳实践工作流

```bash
# Step 1: 快速验证（5分钟）
./test_one_sample_per_user.sh

# Step 2: 训练 1 个 epoch 验证代码（5分钟）
./run_dmsc_one_per_user.sh
# 修改 --max_epochs 1

# Step 3: 完整训练（3-4小时）
./run_dmsc_one_per_user.sh
```

---

## ❓ FAQ

### Q1: 每用户一个样本会不会效果差？
A: 不会！因为：
- 每个样本包含用户的**完整历史**
- 样本质量更高（充分利用了时序信息）
- 避免过拟合（不会反复学习同一用户的早期影评）

### Q2: 如果想要更多数据怎么办？
A: 使用采样模式：
```bash
--max_samples_per_user 5  # 每用户采样5个
```
在训练时间和数据量之间取得平衡。

### Q3: 验证集怎么处理？
A: 自动处理：
- 按 `val_ratio` 划分用户
- 训练集和验证集的用户不重叠
- 验证集也是每用户一个样本

### Q4: 可以和 `--max_samples_per_user` 一起使用吗？
A: 不推荐。两者冲突：
- `--one_sample_per_user`: 每用户固定1个
- `--max_samples_per_user N`: 每用户最多N个

选择其中之一即可。

---

## 📚 相关文档

- `DMSC_ONE_SAMPLE_PER_USER.md` - 详细说明
- `DMSC_LONG_SEQUENCE_SUMMARY.md` - 长序列训练方案
- `SEQUENCE_PARALLELISM_OPTIONS.md` - 序列并行选项
- `DMSC_SAMPLING_GUIDE.md` - 采样训练指南

---

## 🎉 开始训练

```bash
# 测试
./test_one_sample_per_user.sh

# 训练
./run_dmsc_one_per_user.sh
```

祝训练顺利！🚀
