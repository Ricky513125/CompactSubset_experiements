# Qwen 30B 模型训练 + 时序数据扩充 - 完整方案

## 🎯 目标

1. ✅ 配置 Qwen 30B 模型进行 8 卡训练和推理
2. ✅ 实现时序数据扩充，基于用户历史生成多个训练样本
3. ✅ 解决 30B 模型 OOM 问题

---

## 📁 文件清单

### Qwen 30B 配置文件
- `config_RealPersonaChat_Qwen30B.json` - RealPersonaChat 数据集 30B 配置
- `config_Chameleons_30B.json` - Chameleons 数据集 30B 配置
- `config_DMSC_30B.json` - DMSC 数据集 30B 配置
- `ds_config_zero3_30b.json` - DeepSpeed ZeRO-3 配置（带 CPU offload）
- `ds_config_zero3_optimized.json` - DeepSpeed ZeRO-3 优化配置（无 CPU offload，更快但需要更多显存）
- `ds_config_zero2_30b.json` - DeepSpeed ZeRO-2 配置（最快，推荐）

### 时序数据扩充
- `data_augmentation_temporal.py` - 数据扩充核心模块
- `train_distributed_DMSC.py` - 已更新支持扩充的训练脚本
- `preview_augmentation.py` - 预览扩充效果的脚本

### 启动脚本
- `run_qwen30b_train.sh` - Qwen 30B 训练脚本
- `run_qwen30b_inference.sh` - Qwen 30B 推理脚本
- `run_dmsc_with_augmentation.sh` - DMSC 数据扩充训练脚本

### 查看工具
- `inspect_training_input.py` - 查看实际输入给模型的数据
- `check_qwen30b_env.py` - 检查 30B 模型训练环境

### 文档
- `QWEN30B_TRAINING_GUIDE.md` - Qwen 30B 训练详细指南
- `QWEN30B_QUICK_COMMANDS.md` - 快速命令参考
- `TEMPORAL_AUGMENTATION_GUIDE.md` - 时序数据扩充指南
- `INSPECT_INPUT_GUIDE.md` - 数据查看指南
- `SUMMARY.md` - 本文档

---

## 🚀 快速开始

### 1. 预览数据扩充效果

```bash
# 预览 DMSC 数据集扩充效果
python preview_augmentation.py --config config_DMSC.json --min_history_length 1

# 输出示例:
# 原始样本数: 5000
# 扩充后样本数: 20000
# 扩充倍数: 4.00x
```

### 2. 使用时序扩充训练 DMSC（8B 模型）

```bash
# 方式1: 使用快速脚本
bash run_dmsc_with_augmentation.sh

# 方式2: 手动命令
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_DMSC.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_augmented \
    --enable_temporal_augmentation \
    --min_history_length 1 \
    --max_epochs 50 \
    --wandb_project Qwen3-DMSC
```

### 3. 训练 Qwen 30B 模型

```bash
# RealPersonaChat 数据集
bash run_qwen30b_train.sh profile_and_context v1

# 或使用完整命令
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Qwen30B.json \
    --deepspeed ds_config_zero2_30b.json \
    --ablation_config profile_and_context \
    --output_dir outputs/Qwen30B_RealPersonaChat \
    --max_epochs 50 \
    --wandb_project Qwen30B-RealPersonaChat
```

### 4. 查看训练输入数据

```bash
# 查看实际输入给模型的内容
python inspect_training_input.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --num_samples 3

# 或查看训练时生成的预览
cat outputs/DMSC_history_augmented/training_samples_preview.txt
```

---

## 💡 时序数据扩充原理

### 问题
原始方式：每个用户只生成 1 个样本（使用全部历史预测最后一个）
- 用户有 5 个影评 → 1 个训练样本
- 数据利用率低

### 解决方案
时序扩充：基于历史生成多个样本
- 用户有 5 个影评 → 4 个训练样本（设置 min_history_length=1）
  - 样本1: [r1] → r2
  - 样本2: [r1, r2] → r3
  - 样本3: [r1, r2, r3] → r4
  - 样本4: [r1, r2, r3, r4] → r5

### 优势
- ✅ 数据量增加 4-10 倍
- ✅ 模型学习更丰富的时序模式
- ✅ 不同历史长度的样本，提升泛化能力
- ✅ 无需额外标注，自动生成

---

## 🔧 30B 模型 OOM 问题解决

### 问题
- 30B 模型 + ZeRO-3 + 无 CPU offload → OOM
- 原因：激活值占用大，通信缓冲区大

### 解决方案

#### 方案1: 使用 ZeRO-2（推荐，最快）
```bash
--deepspeed ds_config_zero2_30b.json
```
- ✅ 速度快（2-5x 比 ZeRO-3）
- ✅ H100 80GB 显存足够
- ✅ 无需 CPU offload

#### 方案2: 降低序列长度
```json
// config_Chameleons_30B.json
"max_length": 1024  // 从 2048 降低到 1024
```

#### 方案3: 使用 ZeRO-3 + CPU offload（慢但稳定）
```bash
--deepspeed ds_config_zero3_30b.json
```
- ⚠️ 速度慢（每步 15 分钟+）
- ✅ 显存占用最低
- 适合显存不足的情况

#### 方案4: 设置环境变量
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 📊 推荐配置组合

### 组合1: DMSC + 8B 模型 + 时序扩充（推荐）
```bash
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --enable_temporal_augmentation \
    --min_history_length 1 \
    --output_dir outputs/DMSC_8B_augmented
```
- 数据扩充 4-5x
- 训练时间约 10-15 小时
- 推荐用于影评/评分类数据

### 组合2: RealPersonaChat + 30B 模型
```bash
torchrun --nproc_per_node=8 train_distributed_RealPersonaChat.py \
    --config config_RealPersonaChat_Qwen30B.json \
    --deepspeed ds_config_zero2_30b.json \
    --ablation_config profile_and_context \
    --output_dir outputs/RealPersonaChat_30B
```
- 使用 ZeRO-2，速度快
- 训练时间约 20-30 小时
- 推荐用于对话类数据

### 组合3: Chameleons + 30B 模型 + 滑动窗口
```bash
torchrun --nproc_per_node=8 train_distributed_Chameleons.py \
    --config config_Chameleons_30B.json \
    --deepspeed ds_config_zero2_30b.json \
    --ablation_config context_only \
    --enable_temporal_augmentation \
    --use_sliding_window \
    --window_size 10 \
    --output_dir outputs/Chameleons_30B_window
```
- 30B 模型 + 数据扩充
- 滑动窗口控制序列长度
- 训练时间约 25-35 小时

---

## 🎓 最佳实践

### 1. 数据扩充
- ✅ **推荐**: `--min_history_length 1`（确保每个样本都有历史）
- ✅ 历史短（<20）：使用完整历史扩充
- ✅ 历史长（>20）：使用 `--max_samples_per_user 20` 限制
- ✅ 历史很长：使用滑动窗口

### 2. 模型选择
- **8B 模型**: 适合快速实验，4-8 小时/epoch
- **30B 模型**: 适合最终模型，10-20 小时/epoch

### 3. DeepSpeed 配置
- **ZeRO-2**: 最快，推荐用于 H100 80GB
- **ZeRO-3 无 offload**: 次快，需要优化序列长度
- **ZeRO-3 + CPU offload**: 最慢但最稳定

### 4. 早停策略
- 不扩充：`--early_stopping_patience 3`
- 扩充后：`--early_stopping_patience 5`（数据量大，需要更多 patience）

### 5. 监控
- 使用 WandB 实时监控
- 查看 `nvidia-smi` 监控显存
- 查看日志文件确认扩充效果

---

## 🔍 故障排查

### 问题1: OOM
```bash
# 解决方案：
# 1. 降低序列长度
"max_length": 1024

# 2. 使用 ZeRO-2
--deepspeed ds_config_zero2_30b.json

# 3. 减少 batch size（已经是1，无法再减）

# 4. 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 问题2: 扩充后训练太慢
```bash
# 解决方案：
# 1. 限制每用户样本数
--max_samples_per_user 10

# 2. 使用滑动窗口
--use_sliding_window --window_size 5

# 3. 不使用扩充（回退）
# 移除 --enable_temporal_augmentation
```

### 问题3: 扩充没有生效
```bash
# 检查：
# 1. 是否添加了 --enable_temporal_augmentation
# 2. 查看训练日志中的扩充统计
# 3. min_history_length 是否过高
```

### 问题4: ModuleNotFoundError
```bash
# 确保在正确的目录
cd /mnt/parallel/CompactSubset_experiement

# 检查文件是否存在
ls data_augmentation_temporal.py
```

---

## 📝 完整训练命令示例

### DMSC 数据集（推荐新手）
```bash
# 1. 预览扩充效果
python preview_augmentation.py --config config_DMSC.json

# 2. 开始训练（8B + 扩充）
bash run_dmsc_with_augmentation.sh

# 3. 监控训练
tail -f outputs/DMSC_history_augmented_0211/training_logs/train.log
```

### RealPersonaChat（30B 模型）
```bash
# 1. 检查环境
python check_qwen30b_env.py

# 2. 开始训练
bash run_qwen30b_train.sh profile_and_context v1

# 3. 查看 WandB
# 访问打印出的 WandB 链接
```

### 所有数据集批量训练（带扩充）
```bash
# 创建批量脚本
cat > run_all_with_augmentation.sh << 'EOF'
#!/bin/bash
DATASETS=("DMSC" "MovieLens" "Chameleons")

for dataset in "${DATASETS[@]}"; do
    echo "开始训练 $dataset ..."
    torchrun --nproc_per_node=8 train_distributed_${dataset}.py \
        --config config_${dataset}.json \
        --deepspeed ds_config_zero2.json \
        --ablation_config history_only \
        --enable_temporal_augmentation \
        --min_history_length 1 \
        --output_dir outputs/${dataset}_augmented \
        --max_epochs 30
done
EOF

chmod +x run_all_with_augmentation.sh
bash run_all_with_augmentation.sh
```

---

## 🎯 总结

### 已完成
1. ✅ Qwen 30B 模型配置（8卡训练）
2. ✅ DeepSpeed ZeRO-2/ZeRO-3 配置
3. ✅ 时序数据扩充实现
4. ✅ OOM 问题解决方案
5. ✅ 完整的训练和推理脚本
6. ✅ 数据查看工具
7. ✅ 详细文档和示例

### 下一步
1. 运行 `preview_augmentation.py` 查看扩充效果
2. 使用 `run_dmsc_with_augmentation.sh` 开始训练
3. 监控 WandB 查看训练效果
4. 根据效果调整参数（扩充倍数、序列长度等）

### 相关链接
- 模型路径: `/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507`
- 数据路径: `/mnt/parallel/GIDigitalTwinBench/`
- 输出路径: `outputs/`
- WandB 项目: `Qwen3-DMSC`, `Qwen30B-RealPersonaChat` 等

---

**祝训练顺利！🚀**
