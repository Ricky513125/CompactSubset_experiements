# 豆瓣影评模型 - 快速使用指南

## 快速开始

### 1. 准备数据文件

将你的影评数据保存为 `my_movie_reviews.json`，格式参考 `example_movie_review_data.json`。

### 2. 修改配置文件

编辑 `config_MovieReview.json`，设置正确的数据路径：

```json
{
  "data": {
    "train_path": "my_movie_reviews.json"
  },
  "model": {
    "path": "/path/to/your/model"
  }
}
```

### 3. 运行训练

**方式1：使用训练脚本（推荐）**

```bash
# 1. 修改 run_train_movie_review.sh 中的参数
# 2. 运行
bash run_train_movie_review.sh
```

**方式2：直接使用torchrun**

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config profile_and_history \
    --output_dir outputs/my_movie_review_model \
    --max_epochs 50 \
    --val_ratio 0.15 \
    --wandb_project MovieReview \
    --prompt_style simple
```

## 消融实验配置

在 `--ablation_config` 参数中选择：

| 配置 | 说明 |
|------|------|
| `profile_and_history` | 使用用户profile + 历史影评（完整模型）|
| `profile_only` | 仅使用用户profile |
| `history_only` | 仅使用历史影评 |
| `baseline` | 不使用任何上下文（baseline）|

## 完整的消融实验流程

运行所有4个消融实验：

```bash
# 实验1: 完整模型
torchrun --nproc_per_node=8 --master_port=29501 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config profile_and_history \
    --output_dir outputs/MovieReview_exp1_full \
    --wandb_run_name exp1_full

# 实验2: 仅Profile
torchrun --nproc_per_node=8 --master_port=29502 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config profile_only \
    --output_dir outputs/MovieReview_exp2_profile \
    --wandb_run_name exp2_profile

# 实验3: 仅History
torchrun --nproc_per_node=8 --master_port=29503 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config history_only \
    --output_dir outputs/MovieReview_exp3_history \
    --wandb_run_name exp3_history

# 实验4: Baseline
torchrun --nproc_per_node=8 --master_port=29504 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --ablation_config baseline \
    --output_dir outputs/MovieReview_exp4_baseline \
    --wandb_run_name exp4_baseline
```

## DeepSpeed 加速

如果需要使用DeepSpeed优化，添加 `--deepspeed` 参数：

```bash
torchrun --nproc_per_node=8 --master_port=29501 \
    train_distributed_MovieReview.py \
    --config config_MovieReview.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config profile_and_history \
    --output_dir outputs/MovieReview_deepspeed
```

需要先创建 `ds_config_zero2.json`（参考DMSC的配置）。

## 常见命令行参数

### 必需参数

- `--config`: 配置文件路径（默认：config_MovieReview.json）
- `--ablation_config`: 消融实验配置（必需）
- `--output_dir`: 输出目录（可选，默认自动生成）

### 可选参数

- `--data_file`: 数据文件（覆盖配置文件中的设置）
- `--val_ratio`: 验证集比例（默认：配置文件中的设置）
- `--max_epochs`: 最大训练轮次（默认：50）
- `--early_stopping_patience`: 早停耐心值（默认：3）
- `--prompt_style`: Prompt风格，simple或detailed（默认：simple）
- `--disable_flash_attn`: 禁用FlashAttention 2
- `--wandb_project`: W&B项目名称（默认：MovieReview）
- `--wandb_run_name`: W&B运行名称
- `--deepspeed`: DeepSpeed配置文件

## 训练监控

### 使用Weights & Biases

```bash
# 1. 安装并登录
pip install wandb
wandb login

# 2. 训练时会自动上传到W&B
# 查看实时训练曲线: https://wandb.ai/your-username/MovieReview
```

### 查看日志文件

训练期间会生成以下日志：

```
outputs/your_output_dir/
├── training_samples_preview.txt   # 训练样本预览
├── test_samples.json              # 测试集数据
├── training_config.json           # 训练配置
└── checkpoint-xxx/                # 检查点
```

## 故障排查

### 问题1: NCCL错误

```
torch.distributed.DistBackendError: NCCL error
```

**解决方法**：
1. 确保所有GPU可见：`export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
2. 更改master_port：`--master_port=29502`
3. 检查GPU是否被其他进程占用：`nvidia-smi`

### 问题2: 显存不足

```
CUDA out of memory
```

**解决方法**：
1. 在 `config_MovieReview.json` 中减小 `batch_size`
2. 增加 `gradient_accumulation_steps`
3. 减小 `max_length`
4. 使用DeepSpeed ZeRO优化

### 问题3: 数据加载失败

```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方法**：
1. 检查配置文件中的路径是否正确
2. 使用绝对路径：`--data_file /absolute/path/to/data.json`
3. 确保数据文件格式正确

## 评估模型

训练完成后，可以使用测试集评估：

```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model_path = "outputs/my_movie_review_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 加载测试集
with open(f"{model_path}/test_samples.json", 'r') as f:
    test_samples = json.load(f)

# 生成影评
for sample in test_samples[:5]:
    movie = sample['movie_name']
    prompt = f"请为电影《{movie}》写一条影评："
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"电影: {movie}")
    print(f"生成: {generated_review}")
    print(f"真实: {sample['next_question']}")
    print("-" * 80)
```

## 进阶使用

### 自定义Prompt格式

修改 `train_distributed_MovieReview.py` 中的 `MovieReviewDataset.format_prompt()` 方法。

### 修改训练超参数

编辑 `config_MovieReview.json` 中的 `training` 部分：

```json
{
  "training": {
    "max_length": 4096,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-5,
    "warmup_steps": 100
  }
}
```

## 技术支持

遇到问题？
1. 查看 `README_MovieReview.md` 了解详细文档
2. 检查错误日志文件
3. 提交Issue说明问题
