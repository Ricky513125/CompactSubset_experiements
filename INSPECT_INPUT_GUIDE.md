# 查看训练输入数据指南

## 方法1: 使用专用查看脚本（推荐）

### 查看 DMSC 数据集（history_only 配置）

```bash
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --num_samples 3
```

### 保存到文件

```bash
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --num_samples 5 \
    --output_file inspection_output.txt
```

### 查看不同消融配置

```bash
# 查看 profile + context
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config profile_and_context \
    --num_samples 3

# 查看 context only
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config context_only \
    --num_samples 3

# 查看 history + context
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_and_context \
    --num_samples 3
```

## 方法2: 查看训练时自动生成的预览文件

训练开始后，会自动在输出目录生成以下文件：

### 1. 简要预览（推荐快速查看）
```bash
cat outputs/DMSC_history_0210_0/training_samples_preview.txt
```

### 2. 详细训练日志
```bash
cat outputs/DMSC_history_0210_0/training_logs/detailed_training_log.txt
```

### 3. 实时查看（训练过程中）
```bash
tail -f outputs/DMSC_history_0210_0/training_logs/detailed_training_log.txt
```

## 输出内容说明

脚本会输出以下信息：

1. **原始样本信息**: User Hash, Profile 等元数据
2. **对话上下文**: 完整的对话历史（如果使用 context）
3. **历史信息**: 用户的历史记录（如果使用 history）
4. **目标输出**: 模型需要学习生成的内容
5. **完整输入文本**: 实际输入给模型的文本（包含特殊 tokens）
6. **Labels 部分**: 模型需要学习的部分（去掉 masked 部分）
7. **统计信息**: Token 数量、Prompt/Target 长度等
8. **Token IDs**: 实际的 token ID 数组

## 所有数据集查看命令

### DMSC (MovieReview)
```bash
python inspect_training_input.py --config config_DMSC.json --ablation_config history_only --num_samples 3
```

### Chameleons
```bash
python inspect_training_input.py --config config_Chameleons.json --ablation_config context_only --num_samples 3
```

### MovieLens
```bash
python inspect_training_input.py --config config_MovieLens.json --ablation_config history_only --num_samples 3
```

### RealPersonaChat
```bash
python inspect_training_input.py --config config_RealPersonaChat.json --ablation_config profile_and_context --num_samples 3
```

### LovinkDialogue
```bash
python inspect_training_input.py --config config_LovinkDialogue.json --ablation_config profile_and_context --num_samples 3
```

### LovinkQuestionnaire
```bash
python inspect_training_input.py --config config_LovinkQuestionnaire.json --ablation_config profile_and_context --num_samples 3
```

## 30B 模型配置查看

```bash
# Chameleons 30B
python inspect_training_input.py --config config_Chameleons_30B.json --ablation_config context_only --num_samples 3

# DMSC 30B
python inspect_training_input.py --config config_DMSC_30B.json --ablation_config history_only --num_samples 3

# RealPersonaChat 30B
python inspect_training_input.py --config config_RealPersonaChat_Qwen30B.json --ablation_config profile_and_context --num_samples 3
```

## 典型使用场景

### 场景1: 调试 Prompt 格式
```bash
# 查看实际输入格式是否正确
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --num_samples 1 \
    | grep -A 50 "完整文本"
```

### 场景2: 检查 Token 长度
```bash
# 查看 token 长度分布
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --num_samples 10 \
    | grep "Total Tokens"
```

### 场景3: 验证消融实验配置
```bash
# 对比不同配置的输入差异
python inspect_training_input.py --config config_DMSC.json --ablation_config history_only --num_samples 1 > history_only.txt
python inspect_training_input.py --config config_DMSC.json --ablation_config context_only --num_samples 1 > context_only.txt
diff history_only.txt context_only.txt
```

### 场景4: 检查 Labels 是否正确
```bash
# 查看模型需要学习的部分
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --num_samples 3 \
    | grep -A 10 "模型需要学习的部分"
```

## 快速查看正在训练的配置

```bash
# 如果你正在运行这个命令：
# torchrun --nproc_per_node=8 train_distributed_MovieReview.py --config config_DMSC.json --ablation_config history_only ...

# 可以在训练过程中查看：
cat outputs/DMSC_history_0210_0/training_samples_preview.txt

# 或查看更详细的：
less outputs/DMSC_history_0210_0/training_logs/detailed_training_log.txt
```

## 注意事项

1. **首次运行**: 需要加载模型的 tokenizer，可能需要几秒钟
2. **大样本数**: 如果 `--num_samples` 很大，输出会很长，建议保存到文件
3. **特殊字符**: 输出包含 `<|im_start|>`, `<|im_end|>` 等特殊 tokens，这是正常的
4. **中文显示**: 确保终端支持 UTF-8 编码以正确显示中文

## 故障排查

### 问题: ModuleNotFoundError
```bash
# 确保在正确的目录运行
cd /mnt/parallel/CompactSubset_experiement
python inspect_training_input.py ...
```

### 问题: Tokenizer 加载失败
```bash
# 检查模型路径是否正确
cat config_DMSC.json | grep "path"
ls -lh /mnt/parallel/models/Qwen3-8B/
```

### 问题: 配置文件不存在
```bash
# 列出所有可用的配置文件
ls -lh config_*.json
```
