# Slurm 环境使用指南

## 📦 步骤1: 打包conda环境

首先需要将你的conda环境打包，以便在计算节点上使用。

```bash
# 运行打包脚本
bash pack_lingyu_env.sh
```

这个脚本会：
1. ✓ 激活 `lingyu` 环境
2. ✓ 检查必要的包（PyTorch、Transformers、DeepSpeed等）
3. ✓ 使用 conda-pack 打包环境
4. ✓ 保存到 `/mnt/parallel/slurm_try/lingyu_env.tar.gz`

**输出示例：**
```
================================
✓ 环境打包成功！
================================
文件: /mnt/parallel/slurm_try/lingyu_env.tar.gz
大小: 12G
```

**注意事项：**
- 打包需要几分钟时间
- 确保有足够的磁盘空间（通常需要10-15GB）
- 如果环境已经打包过，旧文件会自动备份

## 🚀 步骤2: 提交训练作业

### 方法A：使用便捷脚本（推荐）

```bash
# 提交单个作业
bash submit_lovink_job.sh <ablation_config> <history_strategy> <history_ratio>

# 示例
bash submit_lovink_job.sh profile_and_history fixed_ratio 0.5
```

这个脚本会：
1. ✓ 自动创建 logs 目录
2. ✓ 修改训练参数
3. ✓ 提示确认后提交作业
4. ✓ 显示作业ID和日志位置

### 方法B：批量提交消融实验

```bash
# 一次提交所有消融实验
bash batch_submit_lovink_experiments.sh
```

这会提交8个实验：
- profile_and_history + fixed_ratio (0.3, 0.5, 0.7)
- profile_and_history + random
- profile_and_history + all_previous
- profile_only
- history_only
- context_only

### 方法C：直接使用sbatch

```bash
# 1. 手动创建logs目录
mkdir -p logs

# 2. 修改 train_lovink_questionnaire.sbatch 中的参数
vim train_lovink_questionnaire.sbatch

# 3. 提交作业
sbatch train_lovink_questionnaire.sbatch
```

## 📊 步骤3: 监控作业

### 查看作业状态

```bash
# 查看所有作业
squeue -u $USER

# 查看特定作业
squeue -j <JOB_ID>

# 查看作业详情
scontrol show job <JOB_ID>
```

### 查看实时日志

```bash
# 标准输出
tail -f logs/lovink_questionnaire_<JOB_ID>.out

# 错误输出
tail -f logs/lovink_questionnaire_<JOB_ID>.err
```

### 取消作业

```bash
# 取消单个作业
scancel <JOB_ID>

# 取消所有作业
scancel -u $USER
```

## 🔧 配置参数

### Slurm资源配置

在 `train_lovink_questionnaire.sbatch` 中修改：

```bash
#SBATCH --gres=gpu:8              # GPU数量
#SBATCH --mem=200G                # 内存
#SBATCH --time=48:00:00           # 最长运行时间
#SBATCH --partition=debug         # 分区（根据你的集群）
```

### 训练参数配置

在脚本中修改这些变量：

```bash
ABLATION_CONFIG="profile_and_history"    # 消融配置
HISTORY_STRATEGY="fixed_ratio"           # 历史策略
HISTORY_RATIO=0.5                        # 历史比例
```

**可选的消融配置：**
- `profile_and_history_and_context`
- `profile_and_history`
- `profile_and_context`
- `history_and_context`
- `profile_only`
- `history_only`
- `context_only`

**可选的历史策略：**
- `all_previous` - 使用所有之前的问答
- `fixed_ratio` - 前N%问题作为先验
- `fixed_count` - 固定N个问答
- `random` - 随机选择
- `none` - 不使用历史

## 📁 文件结构

```
CompactSubset_experiement/
├── pack_lingyu_env.sh                    # 环境打包脚本
├── train_lovink_questionnaire.sbatch    # Slurm作业脚本
├── submit_lovink_job.sh                  # 便捷提交脚本
├── batch_submit_lovink_experiments.sh   # 批量提交脚本
├── logs/                                 # 日志目录（自动创建）
│   ├── lovink_questionnaire_<JOB_ID>.out
│   └── lovink_questionnaire_<JOB_ID>.err
└── outputs/                              # 训练输出
    └── Lovink_<config>_<JOB_ID>/
```

## 🐛 常见问题

### Q1: 提交作业时提示 "logs 目录不存在"

**解决方案：**
```bash
mkdir -p /mnt/parallel/CompactSubset_experiement/logs
```

或使用 `submit_lovink_job.sh`，它会自动创建。

### Q2: 作业运行失败，提示环境文件不存在

**解决方案：**
```bash
# 重新打包环境
bash pack_lingyu_env.sh

# 检查文件是否存在
ls -lh /mnt/parallel/slurm_try/lingyu_env.tar.gz
```

### Q3: GPU不可用或数量不对

**检查：**
1. 查看 `logs/lovink_questionnaire_<JOB_ID>.out` 中的 GPU 信息
2. 确认 `#SBATCH --gres=gpu:8` 与实际匹配
3. 检查 CUDA_VISIBLE_DEVICES 环境变量

**调试：**
```bash
# 在计算节点上测试
srun --gres=gpu:2 nvidia-smi
```

### Q4: 如何修改输出目录？

在 `train_lovink_questionnaire.sbatch` 中修改：
```bash
OUTPUT_DIR="outputs/custom_name_${SLURM_JOB_ID}"
```

### Q5: 如何使用DeepSpeed？

添加 `--deepspeed` 参数到训练命令：
```bash
"$TMP_DIR/$ENV_NAME/bin/torchrun" \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_distributed_LovinkQuestionnaire.py \
    --config $CONFIG_FILE \
    --deepspeed ds_config_zero2.json \
    ...
```

## 📝 示例工作流

### 完整的训练流程

```bash
# Step 1: 打包环境（只需做一次）
bash pack_lingyu_env.sh

# Step 2: 提交作业
bash submit_lovink_job.sh profile_and_history fixed_ratio 0.5

# Step 3: 查看作业状态
squeue -u $USER

# Step 4: 监控训练
tail -f logs/lovink_questionnaire_<JOB_ID>.out

# Step 5: 查看结果
ls -lh outputs/Lovink_profile_and_history_<JOB_ID>/
```

### 批量消融实验

```bash
# Step 1: 确保环境已打包
bash pack_lingyu_env.sh

# Step 2: 批量提交
bash batch_submit_lovink_experiments.sh

# Step 3: 监控所有作业
watch -n 5 'squeue -u $USER'

# Step 4: 对比结果
for dir in outputs/Lovink_*/; do
    echo "$dir: $(tail -1 $dir/training_logs/training_progress.txt)"
done
```

## 💡 优化建议

### 1. 选择合适的分区

```bash
# 开发测试
#SBATCH --partition=debug
#SBATCH --time=2:00:00

# 正式训练
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
```

### 2. 合理分配资源

- **小模型（3B）**：4-8 GPU，100-200GB内存
- **大模型（30B）**：8 GPU，200-400GB内存

### 3. 使用W&B监控

确保在环境中设置：
```bash
export WANDB_API_KEY="your_key"
```

### 4. 定期检查日志

```bash
# 检查是否有错误
grep -i "error\|fail\|exception" logs/lovink_questionnaire_*.err

# 查看训练进度
grep "Step\|Epoch" logs/lovink_questionnaire_*.out | tail -20
```

## 🎉 快速开始命令

```bash
# 一键完成所有步骤
bash pack_lingyu_env.sh && \
bash submit_lovink_job.sh profile_and_history fixed_ratio 0.5
```

祝训练顺利！🚀
