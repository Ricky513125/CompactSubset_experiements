# 序列并行方案 - 处理超长序列

## 问题背景

DMSC 影评数据可能包含很长的文本，当前 `max_length=2048` 可能不够。但增加 `max_length` 会导致：
- 激活内存急剧增加（与序列长度的平方成正比，对于 self-attention）
- OOM 错误

## 解决方案对比

| 方案 | 优点 | 缺点 | 实现难度 | 推荐度 |
|------|------|------|----------|--------|
| **1. 增加 max_length + 优化** | 简单，不需要改代码 | 受限于显存 | ⭐ 简单 | ⭐⭐⭐⭐ |
| **2. DeepSpeed Ulysses** | 原生支持，训练快 | 需要 DeepSpeed 0.10+ | ⭐⭐ 中等 | ⭐⭐⭐ |
| **3. Megatron-DeepSpeed** | 成熟，功能完整 | 需要重写训练代码 | ⭐⭐⭐⭐⭐ 复杂 | ⭐⭐ |
| **4. Ring Attention** | 支持极长序列 | 实验性功能 | ⭐⭐⭐⭐ 困难 | ⭐ |

---

## 方案 1: 增加 max_length + 内存优化（推荐）

### 配置修改

**`config_DMSC_30B.json`**:
```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 4,  // 减少到4
    "max_length": 4096,  // 增加到4096
    "max_context_turns": 15
  }
}
```

**`ds_config_zero3_optimized.json`**:
```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,  // 启用 CPU checkpointing
    "contiguous_memory_optimization": true,
    "number_checkpoints": 8  // 增加 checkpoint 数量
  }
}
```

### 优点
- ✅ 不需要修改代码
- ✅ 可以立即测试
- ✅ 支持 4096-8192 长度

### 缺点
- ⚠️ 训练速度会变慢（CPU checkpointing）
- ⚠️ 极限长度约 8192（30B 模型 + 8 × H100）

### 使用方法
```bash
# 测试 max_length=4096
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_4k_test \
    --max_epochs 1 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name 4k_length_test \
    --prompt_style simple \
    --max_samples_per_user 5  # 少量样本测试
```

---

## 方案 2: DeepSpeed Ulysses 序列并行（推荐 ⭐）

### 什么是 Ulysses？

DeepSpeed 的 Ulysses 是一种**序列并行**技术：
- 将序列维度切分到多个 GPU
- 每个 GPU 处理序列的一部分
- 通过 All-to-All 通信同步

### 架构图

```
原始序列: [0, 1, 2, ..., 4095]  (4096 tokens)

切分后:
GPU 0: [0:512]    → 处理这部分的 attention
GPU 1: [512:1024] → 处理这部分的 attention
...
GPU 7: [3584:4096]

All-to-All 通信 → 合并结果
```

### 配置方法

**新建 `ds_config_zero3_ulysses.json`**:
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  
  "bf16": {
    "enabled": true
  },
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 1e8,
    "stage3_prefetch_bucket_size": 1e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  
  "sequence_parallel": {
    "enabled": true,
    "type": "ulysses",
    "size": 8
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4
  },
  
  "communication_data_type": "fp16"
}
```

### 代码修改

需要在 `train_distributed_MovieReview.py` 中启用序列并行：

```python
# 在模型初始化时
from transformers import AutoModelForCausalLM
import deepspeed

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 必须使用 FA2
    trust_remote_code=True
)

# DeepSpeed 会自动处理序列并行（如果配置文件中启用）
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config_path,
    ...
)
```

### 优点
- ✅ 支持 8K-16K 序列长度
- ✅ 训练速度快（GPU-to-GPU 通信）
- ✅ 与 ZeRO-3 兼容

### 缺点
- ⚠️ 需要 DeepSpeed >= 0.10.0
- ⚠️ 需要启用 FlashAttention 2
- ⚠️ 通信开销（All-to-All）

### 兼容性检查

```bash
# 检查 DeepSpeed 版本
python3 -c "import deepspeed; print(deepspeed.__version__)"

# 如果版本 < 0.10.0，需要升级
pip install --upgrade deepspeed
```

---

## 方案 3: Megatron-DeepSpeed（最强但最复杂）

### 什么是 Megatron-DeepSpeed？

NVIDIA Megatron-LM + Microsoft DeepSpeed 的集成：
- 张量并行（Tensor Parallelism）
- 流水线并行（Pipeline Parallelism）
- 序列并行（Sequence Parallelism）
- ZeRO 优化

### 架构

```
模型切分方式:
- 张量并行: 将模型的每一层切分到多个 GPU
- 流水线并行: 将模型的层切分到多个阶段
- 序列并行: 将序列切分到多个 GPU

例如 (TP=4, PP=2):
[Layer 1-15] → GPU 0-3 (张量并行)
[Layer 16-30] → GPU 4-7 (张量并行)
```

### 需要的改动

1. **安装 Megatron-DeepSpeed**:
```bash
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
pip install -e .
```

2. **转换模型格式**:
```bash
# Hugging Face → Megatron 格式
python tools/convert_checkpoint/convert_hf_to_megatron.py \
    --model-type GPT \
    --load-dir /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507 \
    --save-dir /mnt/parallel/models/Qwen3-30B-Megatron \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2
```

3. **重写训练脚本**（使用 Megatron 的训练循环）

### 优点
- ✅ 支持 32K+ 序列长度
- ✅ 极致性能优化
- ✅ 成熟稳定

### 缺点
- ❌ 需要重写整个训练流程
- ❌ 学习曲线陡峭
- ❌ 模型格式转换复杂

---

## 方案 4: Ring Attention（实验性）

### 什么是 Ring Attention？

通过环形通信将注意力计算分布到多个 GPU：
- 支持百万级 token 序列
- GPU 间通过环形拓扑通信

### 实现库
- [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)
- [flash-attn](https://github.com/Dao-AILab/flash-attention) (部分支持)

### 集成难度
- ⚠️ 需要自定义 attention 层
- ⚠️ 可能与 Hugging Face Trainer 不兼容
- ⚠️ 实验性功能，可能不稳定

---

## 推荐方案

### 根据序列长度需求

| 需求 | 推荐方案 | 配置 |
|------|---------|------|
| **2K-4K** | 方案 1: max_length + CPU checkpointing | 立即可用 ✅ |
| **4K-8K** | 方案 2: DeepSpeed Ulysses | 中等改动 ⭐⭐⭐ |
| **8K-16K** | 方案 2 或 3 | 需要评估 |
| **16K+** | 方案 3: Megatron-DeepSpeed | 大改动 |

### 我的建议

**Step 1: 先测试方案 1（5 分钟）**
```bash
# 修改 config_DMSC_30B.json 的 max_length 到 4096
# 修改 ds_config_zero3_optimized.json 启用 cpu_checkpointing
# 运行小规模测试
```

**Step 2: 如果方案 1 不够，再考虑方案 2（1 小时）**
- 检查 DeepSpeed 版本
- 创建 Ulysses 配置文件
- 小规模测试

**Step 3: 如果都不行，考虑方案 3（1-2 天）**

---

## 立即可用的配置（方案 1 优化版）

### 1. 修改 `config_DMSC_30B.json`

```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_length": 4096,  // 2048 → 4096
    "max_context_turns": 20
  }
}
```

### 2. 修改 `ds_config_zero3_optimized.json`

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,  // 启用
    "contiguous_memory_optimization": true,
    "number_checkpoints": 8  // 4 → 8
  }
}
```

### 3. 测试命令

```bash
# 先用少量数据测试
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_4k_test \
    --max_epochs 1 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name 4k_test \
    --prompt_style simple \
    --max_samples_per_user 5
```

### 4. 监控显存

```bash
# 另一个终端运行
watch -n 1 nvidia-smi
```

如果显存足够 → 可以增加到 8192
如果 OOM → 需要方案 2（Ulysses）

---

## 总结

- **立即测试**: 方案 1（增加 max_length + CPU checkpointing）
- **如果不够**: 方案 2（DeepSpeed Ulysses）
- **终极方案**: 方案 3（Megatron-DeepSpeed）

您希望我先帮您实现**方案 1（立即可用）**还是**方案 2（Ulysses）**？
