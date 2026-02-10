# 训练命令汇总 - 8个数据集

本文档包含8个数据集的训练命令，每个数据集都有独立的配置文件和训练脚本。

## 文件结构

### 配置文件 (Config Files)
1. `config_LovinkDialogue.json` - IdealSelf/LovinkDialogue
2. `config_LovinkQuestionnaire.json` - IdealSelf/LovinkQuestionnaire
3. `config_RealPersonaChat.json` - IdealSelf/RealPersonaChat
4. `config_DMSC.json` - RealSelf/DMSC
5. `config_MovieLens.json` - RealSelf/MovieLens
6. `config_Chameleons.json` - RealSelf/Chameleons
7. `config_PERSONA_Bench.json` - RealSelf/PERSONA-Bench
8. `config_REALTALK.json` - RealSelf/REALTALK

### 训练脚本 (Training Scripts)
1. `train_distributed_LovinkDialogue.py`
2. `train_distributed_LovinkQuestionnaire.py`
3. `train_distributed_RealPersonaChat.py`
4. `train_distributed_DMSC.py`
5. `train_distributed_MovieLens.py`
6. `train_distributed_Chameleons.py`
7. `train_distributed_PERSONA_Bench.py`
8. `train_distributed_REALTALK.py`

---

## 训练命令

### 1. LovinkDialogue (IdealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-LovinkDialogue \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 2. LovinkQuestionnaire (IdealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config config_LovinkQuestionnaire.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkQuestionnaire_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-LovinkQuestionnaire \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 3. RealPersonaChat (IdealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/RealPersonaChat_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-RealPersonaChat \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 4. DMSC (RealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_DMSC.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/DMSC_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-DMSC \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 5. MovieLens (RealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/MovieLens_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 6. Chameleons (RealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_Chameleons.py \
    --config config_Chameleons.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/Chameleons_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-Chameleons \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 7. PERSONA-Bench (RealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_PERSONA_Bench.py \
    --config config_PERSONA_Bench.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/PERSONA_Bench_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-PERSONA-Bench \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

### 8. REALTALK (RealSelf)

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_REALTALK.py \
    --config config_REALTALK.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_context \
    --output_dir outputs/REALTALK_profile_context_0209_v1.1 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-REALTALK \
    --wandb_run_name profile_context_0209_v1.1 \
    --prompt_style simple
```

---

## 参数说明

- `--nproc_per_node=8`: 使用8张GPU进行训练
- `--master_port=29500`: 分布式训练主节点端口
- `--config`: 指定配置文件
- `--deepspeed`: DeepSpeed配置文件（使用ZeRO-2优化）
- `--ablation_config profile_and_context`: 消融实验配置（使用用户画像和上下文）
- `--output_dir`: 模型输出目录
- `--max_epochs 50`: 最大训练轮次
- `--early_stopping_patience 3`: 早停耐心值
- `--early_stopping_threshold 0.001`: 早停阈值
- `--val_ratio 0.1`: 验证集比例（10%）
- `--wandb_project`: Weights & Biases项目名称
- `--wandb_run_name`: W&B运行名称
- `--prompt_style simple`: 使用简洁的prompt风格

## 消融实验配置选项

可以通过修改 `--ablation_config` 参数来进行不同的消融实验：

- `profile_and_history_and_context`: 使用画像、历史和上下文（全部）
- `profile_and_history`: 使用画像和历史
- `profile_and_context`: 使用画像和上下文（推荐）
- `history_and_context`: 使用历史和上下文
- `profile_only`: 仅使用画像
- `history_only`: 仅使用历史
- `context_only`: 仅使用上下文

## 注意事项

1. 确保所有依赖的文件都在正确的位置：
   - `data_loader_more_data.py`
   - `train_with_dynamic_padding_Lovink.py`
   - `ds_config_zero2.json`

2. 确保数据集路径正确：
   - IdealSelf数据集：`/mnt/parallel/GIDigitalTwinBench/IdealSelf/`
   - RealSelf数据集：`/mnt/parallel/GIDigitalTwinBench/RealSelf/`

3. 输出目录会自动创建在 `outputs/` 下

4. 训练日志会保存在输出目录的 `training_logs/` 子目录中

5. 如需修改其他参数（如学习率、batch size等），请编辑相应的配置JSON文件
