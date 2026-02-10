# Qwen 30B 模型训练和推理命令

## 模型信息
- **模型路径**: `/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507`
- **配置文件**: `config_RealPersonaChat_Qwen30B.json`
- **DeepSpeed配置**: `ds_config_zero3_30b.json` (ZeRO-3 + CPU Offload)

## 针对 30B 模型的优化调整

相比 4B 模型，30B 模型需要以下调整：

1. **DeepSpeed ZeRO-3**: 使用 ZeRO-3 + CPU Offload，将优化器状态和模型参数 offload 到 CPU
2. **梯度累积步数**: 从 32 增加到 64，保持有效 batch size
3. **学习率**: 从 1e-5 降低到 5e-6，大模型需要更小的学习率
4. **最大序列长度**: 从 16384 降低到 8192，节省显存
5. **Activation Checkpointing**: 启用分区激活，进一步节省显存

## 训练命令

### 基础训练（8卡）

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Qwen30B.json \
    --deepspeed ds_config_zero3_30b.json \
    --ablation_config profile_and_context \
    --output_dir outputs/Qwen30B_RealPersonaChat_profile_context_0210 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen30B-RealPersonaChat \
    --wandb_run_name profile_context_0210 \
    --prompt_style simple
```

### 其他消融实验配置

#### 1. Profile + History + Context (全部特征)
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Qwen30B.json \
    --deepspeed ds_config_zero3_30b.json \
    --ablation_config profile_and_history_and_context \
    --output_dir outputs/Qwen30B_RealPersonaChat_all_features_0210 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen30B-RealPersonaChat \
    --wandb_run_name all_features_0210 \
    --prompt_style simple
```

#### 2. Profile Only
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Qwen30B.json \
    --deepspeed ds_config_zero3_30b.json \
    --ablation_config profile_only \
    --output_dir outputs/Qwen30B_RealPersonaChat_profile_only_0210 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen30B-RealPersonaChat \
    --wandb_run_name profile_only_0210 \
    --prompt_style simple
```

## 推理/评估命令

### 单样本推理测试

```bash
python inference_test_30b.py \
    --model_path /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507 \
    --checkpoint_path outputs/Qwen30B_RealPersonaChat_profile_context_0210/checkpoint-best \
    --test_data_path /mnt/parallel/GIDigitalTwinBench/IdealSelf/RealPersonaChat/test.json \
    --output_file results/Qwen30B_inference_results_0210.json
```

### 批量推理（8卡并行）

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    inference_distributed_30b.py \
    --model_path /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507 \
    --checkpoint_path outputs/Qwen30B_RealPersonaChat_profile_context_0210/checkpoint-best \
    --test_data_path /mnt/parallel/GIDigitalTwinBench/IdealSelf/RealPersonaChat/test.json \
    --output_dir results/Qwen30B_batch_inference_0210 \
    --batch_size 2 \
    --max_length 8192
```

## 监控训练进度

### 1. 查看实时日志
```bash
tail -f outputs/Qwen30B_RealPersonaChat_profile_context_0210/training_logs/train.log
```

### 2. 使用 Weights & Biases
访问: https://wandb.ai/your-team/Qwen30B-RealPersonaChat

### 3. 查看 GPU 使用情况
```bash
watch -n 1 nvidia-smi
```

### 4. 查看训练进程
```bash
ps aux | grep train_distributed
```

## 显存估算

对于 Qwen 30B 模型（8卡 A100/H100 80GB）：

- **模型参数**: ~30B × 2 bytes (bf16) = 60GB
- **ZeRO-3 优化**: 参数分片到 8 卡，每卡 ~7.5GB
- **梯度 + 优化器**: Offload 到 CPU
- **激活值**: ~15-20GB/卡（取决于 batch size 和序列长度）
- **预估总显存/卡**: ~25-30GB

如果显存不足，可以进一步调整：
- 降低 `max_length` 到 4096
- 增加 `gradient_accumulation_steps` 到 128
- 启用 `cpu_checkpointing`

## 注意事项

1. **首次运行**: 模型加载可能需要 5-10 分钟，请耐心等待
2. **CPU 内存**: ZeRO-3 会使用大量 CPU 内存（建议 >256GB）
3. **保存检查点**: ZeRO-3 保存检查点时需要额外时间，这是正常的
4. **多端口**: 如果 29500 端口被占用，更改为 29501、29502 等
5. **断点续训**: 使用 `--resume_from_checkpoint` 参数指定检查点路径

## 故障排查

### 显存溢出 (OOM)
```bash
# 方案1: 减小序列长度
# 在配置文件中修改: "max_length": 4096

# 方案2: 增加梯度累积
# 在配置文件中修改: "gradient_accumulation_steps": 128

# 方案3: 启用 CPU checkpointing
# 在 ds_config_zero3_30b.json 中修改:
# "cpu_checkpointing": true
```

### NCCL 超时
```bash
# 设置更长的超时时间
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
```

### 端口冲突
```bash
# 查找占用端口的进程
lsof -i :29500
# 或使用其他端口
--master_port=29502
```

## 性能优化建议

1. **使用 FlashAttention-2**: 如果支持，速度提升 2-3x
2. **开启 bf16**: 已在配置中启用
3. **调整 overlap_comm**: 已启用通信与计算重叠
4. **预分配显存**: 设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`

## 配置文件说明

### config_RealPersonaChat_Qwen30B.json
- 模型路径配置
- 数据集路径配置  
- 训练超参数（针对 30B 优化）
- 消融实验配置

### ds_config_zero3_30b.json
- ZeRO-3 优化策略
- CPU Offload 配置
- Activation Checkpointing
- 通信优化参数
