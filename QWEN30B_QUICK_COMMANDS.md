# Qwen 30B 快速命令参考

## 1. 快速开始训练（推荐配置）

```bash
# 使用 profile + context 配置（推荐）
bash run_qwen30b_train.sh profile_and_context v1

# 使用所有特征
bash run_qwen30b_train.sh profile_and_history_and_context v1

# 仅使用 profile
bash run_qwen30b_train.sh profile_only v1
```

## 2. 自定义训练命令

```bash
# 完整命令格式
bash run_qwen30b_train.sh [消融配置] [输出后缀] [GPU数量] [端口]

# 示例：4卡训练
bash run_qwen30b_train.sh profile_and_context v1 4 29500

# 示例：使用不同端口
bash run_qwen30b_train.sh profile_and_context v2 8 29502
```

## 3. 快速推理

```bash
# 使用训练好的模型推理
bash run_qwen30b_inference.sh outputs/Qwen30B_RealPersonaChat_profile_context_v1/checkpoint-best inference_v1

# 自定义推理参数
bash run_qwen30b_inference.sh [检查点路径] [输出名称] [GPU数量] [端口] [批次大小]
```

## 4. 手动训练命令（完整版）

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

## 5. 监控和调试命令

### 查看GPU状态
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志（实时）
```bash
tail -f outputs/Qwen30B_RealPersonaChat_profile_context_0210/training_logs/train.log
```

### 查看训练进程
```bash
ps aux | grep train_distributed_RealPersonaChat
```

### 检查端口占用
```bash
lsof -i :29500
```

### 杀死训练进程
```bash
pkill -f train_distributed_RealPersonaChat
```

## 6. 故障恢复

### 从检查点恢复训练
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Qwen30B.json \
    --deepspeed ds_config_zero3_30b.json \
    --ablation_config profile_and_context \
    --output_dir outputs/Qwen30B_RealPersonaChat_profile_context_0210 \
    --resume_from_checkpoint outputs/Qwen30B_RealPersonaChat_profile_context_0210/checkpoint-1000 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen30B-RealPersonaChat \
    --wandb_run_name profile_context_0210_resumed \
    --prompt_style simple
```

## 7. 消融实验所有配置

| 配置名称 | use_profile | use_history | use_context |
|---------|-------------|-------------|-------------|
| profile_and_history_and_context | ✓ | ✓ | ✓ |
| profile_and_history | ✓ | ✓ | ✗ |
| profile_and_context | ✓ | ✗ | ✓ |
| history_and_context | ✗ | ✓ | ✓ |
| profile_only | ✓ | ✗ | ✗ |
| history_only | ✗ | ✓ | ✗ |
| context_only | ✗ | ✗ | ✓ |

## 8. 性能优化环境变量

```bash
# 设置显存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 设置NCCL超时（如果遇到通信问题）
export NCCL_TIMEOUT=3600

# 启用NCCL调试（排查问题时使用）
export NCCL_DEBUG=INFO

# 禁用tokenizer并行（避免fork警告）
export TOKENIZERS_PARALLELISM=false
```

## 9. 验证模型路径

```bash
# 检查模型是否存在
ls -lh /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507

# 检查模型文件
ls -lh /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507/*.safetensors

# 检查配置文件
cat /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507/config.json | jq .model_type
```

## 10. 批量运行所有消融实验

```bash
# 创建批量运行脚本
cat > run_all_ablations.sh << 'EOF'
#!/bin/bash
CONFIGS=("profile_and_context" "profile_and_history_and_context" "profile_only" "history_only" "context_only")

for config in "${CONFIGS[@]}"; do
    echo "开始训练配置: $config"
    bash run_qwen30b_train.sh $config v1 8 29500
    
    # 等待训练完成（可选）
    wait
    
    echo "完成配置: $config"
    echo "-----------------------------------"
done
EOF

chmod +x run_all_ablations.sh
bash run_all_ablations.sh
```

## 11. 清理和维护

### 清理临时文件
```bash
find outputs/ -name "*.tmp" -delete
find outputs/ -name "*.lock" -delete
```

### 查看磁盘使用
```bash
du -sh outputs/*
df -h /mnt/parallel
```

### 压缩旧的检查点
```bash
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/
```

## 相关文件说明

- `config_RealPersonaChat_Qwen30B.json` - Qwen 30B 模型配置文件
- `ds_config_zero3_30b.json` - DeepSpeed ZeRO-3 配置（针对30B优化）
- `run_qwen30b_train.sh` - 训练启动脚本
- `run_qwen30b_inference.sh` - 推理启动脚本
- `QWEN30B_TRAINING_GUIDE.md` - 详细使用指南
