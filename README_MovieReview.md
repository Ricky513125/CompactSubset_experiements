# 豆瓣影评模型训练指南

本项目提供了一套完整的工具链，用于训练基于用户历史影评数据的影评风格模拟模型。

## 文件说明

### 核心文件

1. **data_loader_movie_review.py** - 影评数据加载器
   - 加载JSON格式的影评数据
   - 按时间顺序划分训练/验证/测试集
   - 提取历史影评作为上下文

2. **train_distributed_MovieReview.py** - 分布式训练脚本
   - 支持单卡/多卡训练
   - FlashAttention 2 加速
   - 动态Batch Padding优化

3. **example_movie_review_data.json** - 示例数据
   - 用户Y的28条影评（2010-2017）
   - 标准JSON格式

## 数据格式

影评数据应遵循以下JSON格式：

```json
[
  {
    "user": {
      "profile": {
        "name": "用户名"
      }
    },
    "task": {
      "description": "任务描述",
      "task_behavior_collections": [
        {
          "type": "movie_review",
          "type_desc": "豆瓣影评数据",
          "data": [
            {
              "context": [],
              "continuation": "影评内容",
              "continuation_prefix": "电影名: ",
              "timestamp": "2010-08-22"
            }
          ]
        }
      ]
    }
  }
]
```

### 关键字段

- `continuation`: 影评文本内容
- `continuation_prefix`: 电影名称（格式："电影名: "）
- `timestamp`: 发布时间（格式："YYYY-MM-DD"）
- `context`: 保持为空列表即可

## 使用方法

### 1. 测试数据加载器

```bash
# 测试数据加载和划分
python data_loader_movie_review.py example_movie_review_data.json
```

输出将显示：
- 样本总数
- 训练/验证/测试集划分
- 时间范围
- 样本示例

### 2. 单卡训练

```bash
python train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/movie_review_model_Y \
    --max_epochs 30 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5
```

### 3. 多卡训练（推荐）

```bash
# 4卡训练
torchrun --nproc_per_node=4 train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/movie_review_model_Y_4gpu \
    --max_epochs 30 \
    --batch_size 2 \
    --gradient_accumulation_steps 4
```

```bash
# 8卡训练
torchrun --nproc_per_node=8 train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/movie_review_model_Y_8gpu \
    --max_epochs 30 \
    --batch_size 2 \
    --gradient_accumulation_steps 2
```

## 参数说明

### 数据参数

- `--data_file`: 影评数据JSON文件（必需）
- `--train_ratio`: 训练集比例（默认0.7）
- `--val_ratio`: 验证集比例（默认0.15）
- `--test_ratio`: 测试集比例（默认0.15）

### 模型参数

- `--model_path`: 预训练模型路径（默认Qwen2.5-3B）
- `--output_dir`: 输出目录（必需）
- `--max_length`: 最大序列长度（默认4096）

### 训练参数

- `--batch_size`: 每个GPU的batch size（默认2）
- `--gradient_accumulation_steps`: 梯度累积步数（默认8）
- `--learning_rate`: 学习率（默认1e-5）
- `--max_epochs`: 最大训练轮次（默认50）
- `--warmup_steps`: warmup步数（默认100）

### 消融实验

- `--use_profile`: 使用用户profile（默认启用）
- `--use_history`: 使用历史影评（默认启用）
- `--no_profile`: 禁用profile
- `--no_history`: 禁用历史影评

### 其他选项

- `--disable_flash_attn`: 禁用FlashAttention 2
- `--wandb_project`: W&B项目名称（默认'MovieReview'）
- `--early_stopping_patience`: 早停耐心值（默认3）

## 数据划分策略

**重要**：本项目采用**时间顺序划分**，而非随机划分！

- ✅ **正确做法**：用早期影评训练，后期影评测试
  - 训练集：2010-2015的影评
  - 验证集：2015-2016的影评
  - 测试集：2016-2017的影评

- ❌ **错误做法**：随机打乱后划分
  - 会导致数据泄漏
  - 无法反映真实的时间演化

默认比例：70% 训练 / 15% 验证 / 15% 测试

## 输出文件

训练完成后，输出目录包含：

```
outputs/movie_review_model_Y/
├── pytorch_model.bin              # 模型权重
├── config.json                    # 模型配置
├── training_config.json           # 训练配置
├── test_samples.json              # 测试集数据
├── training_samples_preview.txt   # 样本预览
└── checkpoint-xxx/                # 训练检查点
```

## 消融实验示例

### 实验1：完整模型（Profile + History）

```bash
python train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/exp1_full \
    --use_profile --use_history
```

### 实验2：仅Profile（无History）

```bash
python train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/exp2_profile_only \
    --use_profile --no_history
```

### 实验3：仅History（无Profile）

```bash
python train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/exp3_history_only \
    --no_profile --use_history
```

### 实验4：无上下文（Baseline）

```bash
python train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/exp4_baseline \
    --no_profile --no_history
```

## 训练监控

### 使用Weights & Biases

```bash
# 安装wandb
pip install wandb

# 登录
wandb login

# 训练时启用监控
python train_distributed_MovieReview.py \
    --data_file example_movie_review_data.json \
    --output_dir outputs/movie_review_wandb \
    --wandb_project "MovieReviewExperiment" \
    --wandb_run_name "user_Y_full_model"
```

## 常见问题

### Q1: 数据太少怎么办？

28条影评对于深度学习来说确实较少。建议：

1. **增加训练轮次**：`--max_epochs 100`
2. **降低学习率**：`--learning_rate 5e-6`
3. **使用更小的模型**：如Qwen2.5-1.5B
4. **收集更多数据**：多个用户的影评

### Q2: 如何评估模型效果？

训练完成后，使用`test_samples.json`进行评估：

```python
# 加载测试集
with open('outputs/movie_review_model_Y/test_samples.json', 'r') as f:
    test_data = json.load(f)

# 使用训练好的模型生成影评
# 与真实影评对比（BLEU, ROUGE等指标）
```

### Q3: GPU显存不足？

尝试以下优化：

1. 减小batch size：`--batch_size 1`
2. 增加梯度累积：`--gradient_accumulation_steps 16`
3. 减小最大长度：`--max_length 2048`
4. 禁用FlashAttention：`--disable_flash_attn`

## 性能优化

### FlashAttention 2

需要CUDA 11.6+和支持的GPU（A100/H100等）：

```bash
pip install flash-attn --no-build-isolation
```

### 多卡并行

有效batch size = `batch_size × gradient_accumulation_steps × num_gpus`

建议配置：
- 4卡：`--batch_size 2 --gradient_accumulation_steps 4` → 有效batch=32
- 8卡：`--batch_size 2 --gradient_accumulation_steps 2` → 有效batch=32

## 技术细节

### 模型输入格式

```
用户: Y
任务: 基于用户在豆瓣上的历史影评数据，模拟该用户的影评风格和行为模式

历史影评记录 (19条):
  电影《钢铁侠1》: boring
  电影《复仇者联盟》: Again！Again！Again！
  ...
  电影《寻龙诀》: 坤哥舒淇黄渤铁三角神还原了小说里的摸金校尉...

请为电影《美人鱼》写一条影评：
```

### 模型输出（要学习的部分）

```
功夫之后再无周星驰。
```

## 引用

如果使用本项目，请引用：

```bibtex
@software{movie_review_trainer,
  title = {豆瓣影评风格模拟模型训练工具},
  year = {2026},
  note = {基于时间顺序的影评数据训练框架}
}
```

## 许可

本项目基于MIT许可开源。

## 联系方式

如有问题，请提Issue或联系项目维护者。
