# 豆瓣影评模型 - 快速开始指南

## 📦 已创建的文件

### 核心文件
- **`data_loader_movie_review.py`** - 影评数据加载器
- **`train_distributed_MovieReview.py`** - 分布式训练脚本（支持DeepSpeed）
- **`config_MovieReview.json`** - 训练配置文件
- **`example_movie_review_data.json`** - 示例数据（用户Y的28条影评）

### 启动脚本
- **`run_train_movie_review.sh`** - 正式训练脚本
- **`quick_test_movie_review.sh`** - 快速测试脚本
- **`test_movie_review_setup.py`** - 环境和配置验证

### 文档
- **`README_MovieReview.md`** - 完整文档
- **`USAGE_MovieReview.md`** - 详细使用说明
- **`QUICKSTART_MovieReview.md`** (本文件) - 快速开始

## 🚀 立即开始（3步）

### 步骤1: 准备数据

将你的影评数据保存为JSON，放在当前目录，例如 `my_reviews.json`：

```json
[{
  "user": {"profile": {"name": "Y"}},
  "task": {
    "description": "模拟用户影评风格",
    "task_behavior_collections": [{
      "type": "movie_review",
      "data": [
        {
          "continuation": "影评内容",
          "continuation_prefix": "电影名: ",
          "timestamp": "2024-01-01"
        }
      ]
    }]
  }
}]
```

### 步骤2: 修改配置

编辑 `config_MovieReview.json`：

```json
{
  "data": {
    "train_path": "my_reviews.json"  // 改成你的数据文件
  },
  "model": {
    "path": "/your/model/path"  // 改成你的模型路径
  }
}
```

### 步骤3: 开始训练

```bash
# 选择一种方式：

# 方式A: 使用脚本（推荐）
bash run_train_movie_review.sh

# 方式B: 直接命令
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config profile_and_history \
    --output_dir outputs/movie_review_0210 \
    --max_epochs 50 \
    --val_ratio 0.15 \
    --wandb_project MovieReview \
    --wandb_run_name exp_0210
```

## 🧪 测试你的配置

在正式训练前，先测试配置是否正确：

```bash
# 测试数据加载和环境
python test_movie_review_setup.py

# 快速训练测试（只训练1个epoch）
bash quick_test_movie_review.sh
```

## 🎯 消融实验

修改 `--ablation_config` 参数进行不同实验：

```bash
# 实验1: 完整模型（Profile + History）
--ablation_config profile_and_history

# 实验2: 仅用户Profile
--ablation_config profile_only

# 实验3: 仅历史影评
--ablation_config history_only

# 实验4: 无上下文（Baseline）
--ablation_config baseline
```

## 💡 常用命令示例

### 8卡训练（标准配置）

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config profile_and_history \
    --output_dir outputs/MovieReview_full_0210 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --val_ratio 0.15 \
    --wandb_project MovieReview \
    --wandb_run_name full_0210 \
    --prompt_style simple
```

### 使用DeepSpeed加速

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_history \
    --output_dir outputs/MovieReview_deepspeed \
    --max_epochs 50 \
    --val_ratio 0.15
```

### 单卡调试

```bash
python train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config baseline \
    --output_dir outputs/debug_single_gpu \
    --max_epochs 5
```

## 📊 输出文件

训练后输出目录包含：

```
outputs/your_output_dir/
├── pytorch_model.bin              # 模型权重
├── config.json                    # 模型配置
├── tokenizer_config.json          # Tokenizer配置
├── training_config.json           # 训练配置
├── test_samples.json              # 测试集（用于评估）
├── training_samples_preview.txt   # 样本预览
└── checkpoint-*/                  # 训练检查点
```

## ⚙️ 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件 | config_MovieReview.json |
| `--ablation_config` | 消融实验类型 | **必需** |
| `--output_dir` | 输出目录 | 自动生成 |
| `--max_epochs` | 训练轮次 | 50 |
| `--val_ratio` | 验证集比例 | 0.15 |
| `--prompt_style` | Prompt风格 | simple |
| `--deepspeed` | DeepSpeed配置 | 无 |
| `--wandb_project` | W&B项目名 | MovieReview |

## 🔧 故障排查

### NCCL/CUDA错误

```bash
# 方法1: 确保GPU可见
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 方法2: 更换端口
--master_port=29502

# 方法3: 检查GPU占用
nvidia-smi

# 方法4: 禁用FlashAttention
--disable_flash_attn
```

### 显存不足

修改 `config_MovieReview.json`：

```json
{
  "training": {
    "batch_size": 1,                 // 降低batch size
    "gradient_accumulation_steps": 16, // 增加梯度累积
    "max_length": 2048               // 减小序列长度
  }
}
```

### 数据格式错误

运行验证脚本检查数据：

```bash
python data_loader_movie_review.py your_data.json
```

## 📞 获取帮助

- 查看详细文档：`README_MovieReview.md`
- 查看使用说明：`USAGE_MovieReview.md`
- 测试环境：`python test_movie_review_setup.py`

## 🎓 对比DMSC训练方式

你习惯的DMSC命令：
```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    train_distributed_DMSC.py \
    --config config_DMSC.json \
    --ablation_config context_only \
    --output_dir outputs/DMSC_context_0210
```

等效的影评命令：
```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config history_only \
    --output_dir outputs/MovieReview_history_0210
```

**关键相似点**：
- ✅ 使用相同的`torchrun`启动方式
- ✅ 支持相同的`--config`配置文件格式
- ✅ 支持相同的`--ablation_config`消融实验
- ✅ 支持相同的`--deepspeed`加速
- ✅ 支持相同的`--wandb`监控
- ✅ 使用相同的动态Padding优化
- ✅ 使用相同的FlashAttention 2

**唯一区别**：
- 数据格式：影评数据使用时间序列格式
- 消融配置名：`profile_and_history` vs `profile_and_context`

## 🎉 完成！

现在你可以开始训练豆瓣影评模型了！祝训练顺利！
