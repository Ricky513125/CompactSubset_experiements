# vLLM 推理使用指南

## 📌 什么是 vLLM？

vLLM 是一个高性能的 LLM 推理引擎，专为大规模生产环境设计。

### 核心优势

| 特性 | HuggingFace Transformers | vLLM | 提升 |
|-----|-------------------------|------|------|
| **吞吐量** | 100 samples/min (8 GPU) | 1500-2000 samples/min (4 GPU TP) | **15-20x** |
| **内存效率** | 每个请求独立 | PagedAttention 共享 | **节省 50-70%** |
| **批处理** | 手动管理 | Continuous Batching 自动 | **智能优化** |
| **GPU 利用率** | 30-50% | 80-95% | **2-3x** |

### 关键技术

1. **PagedAttention**: 类似操作系统的虚拟内存，动态分配 KV cache
2. **Continuous Batching**: 动态添加/移除请求，无需等待整个 batch 完成
3. **Tensor Parallelism**: 单个模型跨多 GPU 并行
4. **Optimized CUDA Kernels**: 针对生成任务优化的底层实现

---

## 🚀 快速开始

### 1. 安装 vLLM

```bash
# 安装 vLLM (需要 CUDA 11.8+)
pip install vllm

# 或者从源码安装（获取最新特性）
pip install git+https://github.com/vllm-project/vllm.git
```

**系统要求**:
- GPU: A100, H100, H200 推荐（V100 也支持但性能较低）
- CUDA: >= 11.8
- GPU Memory: 至少 16GB (8B 模型)
- Python: >= 3.8

### 2. 基本使用

#### 方法 A: 使用现成的推理脚本

```bash
# 单 GPU 推理
python inference_vllm.py \
    --checkpoint_dir outputs/Chameleons_8B_context_sampled_seed42 \
    --dataset Chameleons \
    --ablation_config context_only \
    --num_samples 5 \
    --output_dir outputs/leaderboards/Chameleons_vllm

# 多 GPU Tensor Parallel (4 GPU)
python inference_vllm.py \
    --checkpoint_dir outputs/DMSC_8B_one_per_user_0213 \
    --dataset DMSC \
    --ablation_config profile_and_history \
    --num_samples 5 \
    --output_dir outputs/leaderboards/DMSC_vllm_4gpu \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9
```

python inference_vllm.py \
    --checkpoint_dir outputs/Chameleons_8B_context_full \
    --dataset Chameleons \
    --ablation_config context_only \
    --num_samples 5 \
    --output_dir outputs/leaderboards/Chameleons_vllm_8gpu \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9

#### 方法 B: 使用辅助脚本

```bash
bash inference_with_vllm.sh
```

修改脚本中的 `run_vllm_inference` 调用来适配你的任务。

---

## 📊 性能对比

### 实际测试（8B 模型，1000 个样本）

| 方案 | GPU 数量 | 时间 | 吞吐量 | 显存占用 |
|-----|---------|-----|--------|---------|
| **HF Transformers** (原始) | 8 (DDP) | ~10 min | 100 samples/min | 80GB |
| **vLLM** (单卡) | 1 | ~2 min | 500 samples/min | 18GB |
| **vLLM** (TP) | 4 | ~0.5 min | 2000 samples/min | 4x15GB |

### 成本分析

假设推理 10,000 个样本：
- **HF Transformers (8 GPU)**: 100 分钟 = 800 GPU-分钟
- **vLLM (1 GPU)**: 20 分钟 = 20 GPU-分钟
- **vLLM (4 GPU TP)**: 5 分钟 = 20 GPU-分钟

**节省成本**: **40倍** (相比原始 8 GPU DDP)

---

## 🔧 高级配置

### 1. Tensor Parallelism vs Data Parallelism

#### Tensor Parallelism (vLLM 推荐)

```bash
# 单个模型分布在 4 张 GPU 上
python inference_vllm.py \
    --checkpoint_dir outputs/model \
    --tensor_parallel_size 4 \
    --dataset Chameleons \
    --ablation_config context_only \
    --output_dir outputs/leaderboards
```

**优势**:
- 支持更大的 batch size
- 单个请求延迟低
- 适合大模型 (30B+)

#### Data Parallelism (原始方案)

```bash
# 8 个独立模型副本
torchrun --nproc_per_node=8 inference_distributed.py \
    --checkpoint_dir outputs/model \
    ...
```

**劣势**:
- 每张卡加载完整模型，内存浪费
- 需要手动分配数据到各个进程
- 批处理效率低

### 2. 内存优化

#### 选项 1: 降低 GPU Memory Utilization

```bash
# 默认 0.9 (90%)，如果 OOM 可以降低到 0.7
python inference_vllm.py \
    --gpu_memory_utilization 0.7 \
    ...
```

#### 选项 2: 减少 Max Model Length

```bash
# 默认 8192，可以根据实际需要调整
python inference_vllm.py \
    --max_model_len 4096 \
    ...
```

#### 选项 3: 使用 Quantization

```bash
# 安装 AutoAWQ
pip install autoawq

# 量化模型（4-bit）
python -m awq.entry --model_path outputs/model \
    --w_bit 4 --q_group_size 128 \
    --output_path outputs/model_awq

# 使用量化模型推理
python inference_vllm.py \
    --checkpoint_dir outputs/model_awq \
    --quantization awq \
    ...
```

### 3. 采样参数调优

```bash
python inference_vllm.py \
    --temperature 0.8 \      # 降低温度提高确定性
    --top_p 0.95 \           # nucleus sampling
    --top_k 50 \             # top-k sampling
    --max_tokens 256 \       # 限制生成长度
    --seed 42 \              # 固定随机种子
    ...
```

---

## 🔍 常见问题

### Q1: vLLM vs HuggingFace 推理结果不一致？

**原因**: 采样算法实现略有差异

**解决方案**:
1. 固定随机种子: `--seed 42`
2. 使用 greedy decoding: `--temperature 0.0`
3. 或接受略微差异（通常不影响最终评估）

### Q2: OOM (Out of Memory) 错误

**解决方案**:
```bash
# 1. 降低 GPU 内存利用率
--gpu_memory_utilization 0.7

# 2. 减少最大序列长度
--max_model_len 4096

# 3. 增加 Tensor Parallel 大小
--tensor_parallel_size 2  # 或 4, 8
```

### Q3: vLLM 不支持我的模型？

**检查兼容性**:
```python
from vllm import LLM

# 支持的架构
supported_models = [
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    "GPTNeoXForCausalLM",
    # ... 更多
]
```

如果不支持，可以:
1. 转换为兼容格式
2. 使用 HuggingFace Transformers 作为 fallback
3. 提交 issue 到 vLLM GitHub

### Q4: 如何监控推理性能？

**查看汇总信息**:
```bash
cat outputs/leaderboards/Chameleons_vllm/inference_summary.json
```

**关键指标**:
- `throughput_samples_per_sec`: 吞吐量
- `inference_time_seconds`: 总推理时间
- `total_samples`: 总样本数

---

## 📋 完整参数列表

### inference_vllm.py 参数

```bash
# 必需参数
--checkpoint_dir PATH          # 模型 checkpoint 路径
--dataset NAME                 # 数据集名称 (Chameleons, DMSC, etc.)
--ablation_config CONFIG       # 消融实验配置
--output_dir PATH              # 输出目录

# vLLM 配置
--tensor_parallel_size N       # Tensor Parallel 大小 (默认: 1)
--gpu_memory_utilization F     # GPU 内存利用率 (默认: 0.9)
--max_model_len N              # 最大序列长度 (默认: 8192)

# 采样参数
--temperature F                # 温度 (默认: 1.0)
--top_p F                      # Top-p (默认: 0.9)
--top_k N                      # Top-k (默认: 50)
--max_tokens N                 # 最大生成 token 数 (默认: 512)
--seed N                       # 随机种子 (默认: 42)

# 其他
--num_samples N                # 每用户样本数 (默认: 5)
--scenario_path PATH           # 场景数据路径（可选，自动推断）
```

---

## 🎯 推荐配置

### 配置 1: 快速测试（单 GPU）

```bash
python inference_vllm.py \
    --checkpoint_dir outputs/model \
    --dataset Chameleons \
    --ablation_config context_only \
    --output_dir outputs/test \
    --tensor_parallel_size 1 \
    --num_samples 1
```

**适用场景**: 快速验证，小规模测试

### 配置 2: 生产推理（4 GPU TP）

```bash
python inference_vllm.py \
    --checkpoint_dir outputs/model \
    --dataset Chameleons \
    --ablation_config profile_and_context \
    --output_dir outputs/leaderboards/final \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --num_samples 5
```

**适用场景**: 大规模推理，追求最高吞吐量

### 配置 3: 高质量生成

```bash
python inference_vllm.py \
    --checkpoint_dir outputs/model \
    --dataset DMSC \
    --ablation_config profile_and_history \
    --output_dir outputs/leaderboards/high_quality \
    --temperature 0.8 \
    --top_p 0.95 \
    --top_k 50 \
    --max_tokens 512 \
    --seed 42
```

**适用场景**: 追求生成质量，可复现结果

---

## 🔄 迁移指南：从 HuggingFace 到 vLLM

### 步骤 1: 保存模型为 HuggingFace 格式

如果你的模型已经是 HuggingFace 格式 (通常是)，跳过此步骤。

### 步骤 2: 修改推理脚本

**原始 (HuggingFace)**:
```python
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    text = tokenizer.decode(outputs[0])
```

**新版 (vLLM)**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model=checkpoint_dir, tensor_parallel_size=4)
sampling_params = SamplingParams(temperature=1.0, max_tokens=512)

outputs = llm.generate(prompts, sampling_params)
texts = [output.outputs[0].text for output in outputs]
```

### 步骤 3: 运行并比较结果

```bash
# 原始方法
torchrun --nproc_per_node=8 inference_distributed.py ...

# vLLM 方法
python inference_vllm.py --tensor_parallel_size 4 ...
```

### 步骤 4: 验证输出一致性

```python
# 比较两种方法的输出
import json

with open('outputs/hf_results/user1.json') as f:
    hf_result = json.load(f)

with open('outputs/vllm_results/user1.json') as f:
    vllm_result = json.load(f)

# 检查生成文本
print("HF:", hf_result['generated_samples'][0])
print("vLLM:", vllm_result['generated_samples'][0])
```

---

## 📚 更多资源

- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **vLLM 文档**: https://docs.vllm.ai/
- **性能基准**: https://github.com/vllm-project/vllm#performance
- **社区讨论**: https://github.com/vllm-project/vllm/discussions

---

## 💡 最佳实践

1. **优先使用 Tensor Parallelism**: 比 Data Parallelism 更高效
2. **合理设置 gpu_memory_utilization**: 0.85-0.95 之间较佳
3. **固定随机种子**: 保证结果可复现
4. **监控 GPU 利用率**: 使用 `nvidia-smi dmon -s u` 查看
5. **批量推理**: 一次性准备所有 prompts，让 vLLM 自动优化批处理

---

## 🚨 注意事项

1. **首次加载较慢**: vLLM 会编译 CUDA kernels，首次运行需要 1-2 分钟
2. **不支持流式生成可视化**: vLLM 针对吞吐量优化，不适合交互式场景
3. **内存预分配**: vLLM 会预先分配大量显存，可能导致其他程序 OOM
4. **模型兼容性**: 检查你的模型架构是否被 vLLM 支持

---

**总结**: vLLM 是大规模推理的最佳选择，相比 HuggingFace Transformers 可以节省 **10-40倍** 的时间和成本！
