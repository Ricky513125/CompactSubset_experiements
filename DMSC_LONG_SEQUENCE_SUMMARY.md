# DMSC 超长序列训练方案总结

## 当前配置

您的 `config_DMSC_30B.json` 已经设置了：
```json
{
  "training": {
    "max_length": 16384,  // 16K 超长序列
    "gradient_accumulation_steps": 2
  }
}
```

这是一个**雄心勃勃的配置**！但需要优化才能跑通。

---

## 🚀 推荐的测试流程

### Step 1: 测试当前配置（16K + CPU Checkpointing）

```bash
cd /mnt/parallel/CompactSubset_experiement
./test_16k_length.sh
```

**优化点**：
- ✅ 启用 `cpu_checkpointing: true`（已修改）
- ✅ 增加 `number_checkpoints: 8`（已修改）
- ✅ 使用 `expandable_segments` 减少碎片

**预期结果**：
- ✅ **成功**：可以直接使用 16K 长度训练
- ⚠️ **OOM**：需要尝试 Step 2

---

### Step 2: 如果 OOM，使用 DeepSpeed Ulysses 序列并行

```bash
./test_ulysses.sh
```

**工作原理**：
```
原始: 1个样本 × 16384 tokens → 全部在每个 GPU 上
Ulysses: 1个样本 × 16384 tokens → 分成 8 份
  GPU 0: tokens [0:2048]
  GPU 1: tokens [2048:4096]
  ...
  GPU 7: tokens [14336:16384]
```

**优势**：
- 每个 GPU 只需要处理 1/8 的序列
- 激活内存大幅减少
- 支持更长的序列（32K+）

**要求**：
- DeepSpeed >= 0.10.0
- FlashAttention 2（Qwen3 已支持）

---

## 📊 性能对比

| 方案 | max_length | 每 GPU 激活内存 | 训练速度 | OOM 风险 |
|------|------------|----------------|----------|----------|
| **当前（无优化）** | 16384 | ~40GB | 快 | 🔴 高 |
| **CPU Checkpointing** | 16384 | ~20GB | 中等 | 🟡 中 |
| **Ulysses (8-way)** | 16384 | ~5GB | 中等 | 🟢 低 |
| **Ulysses (8-way)** | 32768 | ~10GB | 中等 | 🟢 低 |

---

## 🔧 配置文件说明

### 1. `ds_config_zero3_optimized.json`（已优化）

用于 **Step 1**：
```json
{
  "activation_checkpointing": {
    "cpu_checkpointing": true,  // ✅ 启用 CPU checkpointing
    "number_checkpoints": 8     // ✅ 增加 checkpoint 数量
  }
}
```

**trade-off**：
- ✅ 减少 GPU 内存占用（约 50%）
- ⚠️ 训练速度变慢（约 20-30%）

### 2. `ds_config_zero3_ulysses.json`（新增）

用于 **Step 2**：
```json
{
  "sequence_parallel": {
    "enabled": true,
    "size": 8,
    "type": "all_to_all"
  },
  "activation_checkpointing": {
    "cpu_checkpointing": false  // 不需要 CPU checkpointing
  }
}
```

**优势**：
- ✅ 激活内存减少 87.5%（1/8）
- ✅ 训练速度更快（GPU-to-GPU 通信）
- ✅ 支持更长序列

---

## ⚡ 快速决策树

```
开始
 │
 ├─ 想要 16K 长度？
 │   │
 │   ├─ 是 → 运行 ./test_16k_length.sh
 │   │       │
 │   │       ├─ 成功 ✅ → 使用 ds_config_zero3_optimized.json
 │   │       └─ OOM ❌ → 运行 ./test_ulysses.sh
 │   │                    │
 │   │                    ├─ 成功 ✅ → 使用 ds_config_zero3_ulysses.json
 │   │                    └─ 失败 ❌ → 减少 max_length 到 8192
 │   │
 │   └─ 否 → 使用 4K-8K 长度 + 标准 Zero-3
 │
完成
```

---

## 📝 完整训练命令

### 方案 A: 16K + CPU Checkpointing

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_16k_full \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name 16k_full_training \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

### 方案 B: 16K + Ulysses 序列并行

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_ulysses.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_16k_ulysses_full \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name 16k_ulysses_full \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

---

## 🎯 建议

### 对于 DMSC 影评数据

影评文本通常不会特别长，**16K 可能过大**。建议：

1. **先用 4K-8K 测试**：
   ```json
   "max_length": 4096  // 或 8192
   ```

2. **监控实际长度**：
   ```bash
   # 检查实际样本长度分布
   grep "Prompt tokens" outputs/DMSC_*/training_logs/detailed_training_log.txt | sort | uniq -c
   ```

3. **根据实际需求调整**：
   - 如果 90% 样本 < 4K → 使用 4096
   - 如果 90% 样本 < 8K → 使用 8192
   - 如果确实需要 16K → 使用 Ulysses

---

## 🔍 监控和调试

### 1. 监控显存使用

```bash
# 终端 1：训练
./test_16k_length.sh

# 终端 2：监控
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
```

### 2. 检查实际序列长度

在训练日志中查看：
```bash
tail -f outputs/DMSC_16k_test/training_logs/detailed_training_log.txt | grep "tokens"
```

### 3. 性能分析

```bash
# 查看每步时间
grep "seconds/step" outputs/DMSC_*/training_logs/detailed_training_log.txt
```

---

## 💡 其他优化建议

### 如果仍然 OOM

1. **减少 batch_size**（已经是 1）
2. **增加 gradient_accumulation_steps**：
   ```json
   "gradient_accumulation_steps": 4  // 当前是 2
   ```

3. **减少 max_context_turns**：
   ```json
   "max_context_turns": 10  // 当前是 15
   ```

4. **启用混合精度优化**：
   ```json
   "fp16": {"enabled": true}  // 如果 bf16 不够
   ```

---

## 📚 相关文档

- `SEQUENCE_PARALLELISM_OPTIONS.md` - 序列并行方案详解
- `DMSC_SAMPLING_GUIDE.md` - 采样训练指南
- `ds_config_zero3_optimized.json` - 标准 Zero-3 配置
- `ds_config_zero3_ulysses.json` - Ulysses 序列并行配置

---

## ✅ 下一步

```bash
# 1. 测试 16K 长度
./test_16k_length.sh

# 2. 查看结果
tail -f outputs/DMSC_16k_test/training_logs/detailed_training_log.txt

# 3. 如果成功，开始完整训练
# 如果失败，尝试 Ulysses 或减少 max_length
```

祝训练顺利！🚀
