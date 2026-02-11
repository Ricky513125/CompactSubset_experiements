# 时序数据扩充使用指南

## 概述

时序数据扩充可以将每个用户的时间序列历史转换为多个训练样本，大幅增加训练数据量。

### 原理示例

假设用户有5个影评，按时间顺序：`[r1, r2, r3, r4, r5]`

**不使用扩充**（原始方式）:
- 只有1个样本：历史[r1, r2, r3, r4] -> 预测 r5

**使用扩充**（推荐设置 min_history_length=1）:
- 样本1: 历史[r1] -> 预测 r2
- 样本2: 历史[r1, r2] -> 预测 r3
- 样本3: 历史[r1, r2, r3] -> 预测 r4
- 样本4: 历史[r1, r2, r3, r4] -> 预测 r5

**数据量**: 从 1 个样本扩充到 4 个样本（4x）

💡 **注意**: 跳过第一个样本（r1）是因为它没有历史，无法体现时序模式。

## 使用方法

### 方法1: 完整历史扩充（推荐）

每个样本使用从开始到当前位置的所有历史。

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_augmented_0211 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-DMSC \
    --wandb_run_name history_augmented_0211 \
    --prompt_style simple \
    --enable_temporal_augmentation \
    --min_history_length 1
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_temporal_augmentation` | 启用时序扩充 | False（不启用） |
| `--min_history_length` | 最小历史长度（**推荐设为1**，确保每个样本都有历史） | 1 |
| `--max_samples_per_user` | 每个用户最多生成的样本数（None表示不限制） | None |

### 方法2: 滑动窗口扩充

只保留固定窗口大小的历史，适合历史很长的情况。

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_window_0211 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-DMSC \
    --wandb_run_name history_window_0211 \
    --prompt_style simple \
    --enable_temporal_augmentation \
    --use_sliding_window \
    --window_size 5 \
    --window_stride 1
```

### 滑动窗口参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_sliding_window` | 使用滑动窗口（而不是完整历史） | False |
| `--window_size` | 窗口大小 | 5 |
| `--window_stride` | 滑动步长 | 1 |

## 不同配置对比

### 配置1: 不扩充（基线）
```bash
# 数据量: 1x（原始）
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_baseline \
    # ... 其他参数
```

### 配置2: 完整扩充（min_history=1，推荐）
```bash
# 数据量: 约4-9x（跳过没有历史的第一个样本）
# ✅ 推荐：确保每个样本都有历史信息
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_full_augment \
    --enable_temporal_augmentation \
    --min_history_length 1
```

### 配置3: 包含零历史样本（min_history=0，不推荐）
```bash
# 数据量: 约5-10x（包含第一个样本）
# ⚠️ 不推荐：第一个样本没有历史，无法体现时序信息
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_with_zero_history \
    --enable_temporal_augmentation \
    --min_history_length 0
```

### 配置4: 限制数量扩充
```bash
# 数据量: 最多每用户10个样本
# 适合用户历史很长的情况
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_limited_augment \
    --enable_temporal_augmentation \
    --min_history_length 0 \
    --max_samples_per_user 10
```

### 配置5: 滑动窗口
```bash
# 数据量: 约3-6x（取决于窗口大小和步长）
# 每个样本只保留最近5个历史
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_window \
    --enable_temporal_augmentation \
    --use_sliding_window \
    --window_size 5 \
    --window_stride 1
```

## 实际应用场景

### 场景1: DMSC 影评数据（推荐完整扩充）
```bash
# 用户历史影评较短（通常5-20个），使用完整扩充
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_augmented \
    --enable_temporal_augmentation \
    --min_history_length 1 \
    --max_epochs 30 \
    --wandb_run_name history_aug
```

### 场景2: MovieLens 评分数据（推荐限制数量）
```bash
# 用户历史可能很长（几百个），限制每用户最多20个样本
torchrun --nproc_per_node=8 train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --ablation_config history_only \
    --output_dir outputs/MovieLens_history_augmented \
    --enable_temporal_augmentation \
    --min_history_length 1 \
    --max_samples_per_user 20 \
    --max_epochs 30
```

### 场景3: Chameleons 对话数据（推荐滑动窗口）
```bash
# 对话历史可能很长，使用滑动窗口保持最近10轮
torchrun --nproc_per_node=8 train_distributed_Chameleons.py \
    --config config_Chameleons.json \
    --ablation_config context_only \
    --output_dir outputs/Chameleons_window_augmented \
    --enable_temporal_augmentation \
    --use_sliding_window \
    --window_size 10 \
    --window_stride 2 \
    --max_epochs 30
```

## 查看扩充效果

训练开始时会打印扩充统计信息：

```
================================================================================
时序数据扩充
================================================================================
原始样本数: 5000
最小历史长度: 1
每用户最大样本数: 不限制
================================================================================

用户 a1b2c3d4... : 10 个原始样本 -> 生成 10 个扩充样本
用户 e5f6g7h8... : 8 个原始样本 -> 生成 8 个扩充样本
...

================================================================================
扩充完成
================================================================================
扩充后样本数: 25000
扩充倍数: 5.00x
================================================================================

================================================================================
数据扩充统计
================================================================================
总样本数: 25000

历史长度分布:
  最小: 1
  最大: 20
  平均: 5.50
  中位数: 5

详细分布:
  长度 1:  3000 ( 12.0%) ██████
  长度 2:  2800 ( 11.2%) █████
  长度 3:  2600 ( 10.4%) █████
  ...
```

## 注意事项

### 1. 训练时间
- 扩充后样本数增加，训练时间会相应增加
- 5x 扩充 ≈ 5x 训练时间（但模型效果通常会更好）

### 2. 显存使用
- 样本数增加不会直接影响显存（batch size 固定）
- 但更长的历史可能导致序列更长，建议监控显存使用

### 3. 验证集划分
- 扩充在数据划分之前进行
- 验证集也会被扩充（保持比例一致）

### 4. 与 history_only 配置最佳匹配
- 数据扩充主要增强历史信息的使用
- 推荐配合 `--ablation_config history_only` 或 `history_and_context` 使用

### 5. 早停策略调整
- 数据量增加后，可能需要调整早停参数
- 建议增加 `--early_stopping_patience` 到 5 或更高

## 完整示例命令

### DMSC 数据集 + 完整扩充
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_augmented_full_0211 \
    --max_epochs 50 \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-DMSC-Augmented \
    --wandb_run_name history_full_aug_0211 \
    --prompt_style simple \
    --enable_temporal_augmentation \
    --min_history_length 1
```

### 对比实验：不扩充 vs 扩充
```bash
# 1. 基线（不扩充）
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_baseline_0211 \
    --wandb_run_name baseline_no_aug

# 2. 完整扩充
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_augmented_0211 \
    --wandb_run_name full_aug \
    --enable_temporal_augmentation \
    --min_history_length 1

# 3. 查看 W&B 对比结果
# 访问 W&B 项目页面查看两个实验的对比
```

## 故障排查

### 问题: ModuleNotFoundError: No module named 'data_augmentation_temporal'
```bash
# 确保文件存在
ls /mnt/parallel/CompactSubset_experiement/data_augmentation_temporal.py

# 在正确的目录运行
cd /mnt/parallel/CompactSubset_experiement
```

### 问题: 扩充后样本数没有变化
- 检查是否添加了 `--enable_temporal_augmentation` 参数
- 检查 `--min_history_length` 是否过高（导致很多样本被过滤）
- 查看训练日志中的扩充统计信息

### 问题: 训练太慢
- 减少扩充倍数：使用 `--max_samples_per_user 10`
- 使用滑动窗口：`--use_sliding_window --window_size 5`
- 或者不使用扩充，保持原始训练方式

## 测试扩充效果

```bash
# 测试扩充脚本
python -c "
from data_augmentation_temporal import expand_samples_with_temporal_history, print_augmentation_stats

test_samples = [
    {'user_hash': 'user1', 'next_question': 'review1', 'context': []},
    {'user_hash': 'user1', 'next_question': 'review2', 'context': []},
    {'user_hash': 'user1', 'next_question': 'review3', 'context': []},
]

expanded = expand_samples_with_temporal_history(test_samples, min_history_length=0, verbose=True)
for i, s in enumerate(expanded):
    print(f'样本{i+1}: history={s[\"history\"]} -> target={s[\"next_question\"]}')
"
```

## 推荐配置

基于数据特点的推荐：

1. **DMSC/MovieReview**: 完整扩充 + min_history=1
2. **MovieLens**: 完整扩充 + max_samples_per_user=20
3. **Chameleons**: 滑动窗口 + window_size=10
4. **RealPersonaChat**: 完整扩充 + min_history=0
5. **LovinkDialogue/Questionnaire**: 完整扩充 + min_history=1
