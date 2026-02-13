# LovinkDialogue 独立训练脚本说明

## ✅ 修改完成

`train_distributed_LovinkDialogue.py` 现在是一个**完全独立**的训练脚本，不依赖外部模块。

---

## 🔧 主要修改

### 1. 注释掉了外部导入（第44-46行）

```python
# 已注释：
# from data_loader import load_train_data, extract_training_samples, get_user_only_history
# from sample_per_user import sample_per_user
# from train_with_dynamic_padding_Lovink import DynamicPaddingDataset, dynamic_padding_collate_fn, split_train_val, add_history_to_samples
```

### 2. 内联了所有需要的函数（第43行后插入）

- ✅ `sample_per_user()` - 每用户采样
- ✅ `split_train_val()` - 训练/验证集划分
- ✅ `add_history_to_samples()` - 添加历史信息
- ✅ `DynamicPaddingDataset` - 动态Padding数据集类
- ✅ `dynamic_padding_collate_fn()` - 动态Padding整理函数

### 3. 注释掉了旧的 main() 函数（第1897行）

只保留了支持所有参数的新 main() 函数（第1957行）

---

## 🚀 使用方法

### 您的命令现在可以直接运行

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

或使用脚本：

```bash
./run_lovink_standalone.sh
```

---

## 📊 关键参数说明

### 数据采样参数

#### `--max_samples_per_user N`
每个用户最多采样 N 个样本

- 大幅减少训练数据量
- 加快训练速度
- 使用固定随机种子保证可复现

```bash
--max_samples_per_user 2 \
--sample_seed 42
```

### Prompt 风格参数

#### `--prompt_style simple`
使用简洁的 Prompt 格式（推荐）

- `simple`: 简洁标签格式（只预测 continuation）
- `detailed`: 详细模板格式
- `lovink`: Lovink 专用格式

```bash
--prompt_style simple
```

### DeepSpeed 参数

#### `--deepspeed ds_config_zero3_optimized.json`
使用 DeepSpeed ZeRO-3 优化

- 支持 30B 模型训练
- 优化过的配置，无 CPU offload

```bash
--deepspeed ds_config_zero3_optimized.json
```

---

## 📁 文件结构

```
train_distributed_LovinkDialogue.py
├── 数据加载函数 (第48-1000行)
│   ├── load_train_data()
│   ├── extract_training_samples()
│   ├── get_user_only_history()
│   └── build_simple_training_prompt()
│
├── 工具函数 (第44-300行，新增)
│   ├── sample_per_user()
│   ├── split_train_val()
│   ├── add_history_to_samples()
│   ├── DynamicPaddingDataset
│   └── dynamic_padding_collate_fn()
│
├── 分布式训练工具 (第1700-1900行)
│   ├── setup_distributed()
│   └── cleanup_distributed()
│
└── 主函数 (第1957-2870行)
    └── main() - 支持所有参数
```

---

## 🎯 训练流程

### 1. 数据加载

```python
train_data = load_train_data(train_path)
all_samples = extract_training_samples(train_data, debug=is_main_process)
```

### 2. 用户采样（如果启用）

```python
if args.max_samples_per_user is not None:
    all_samples = sample_per_user(
        all_samples,
        max_samples_per_user=args.max_samples_per_user,
        random_seed=args.sample_seed
    )
```

### 3. 添加历史信息

```python
all_samples = add_history_to_samples(all_samples, all_samples)
```

### 4. 划分训练/验证集

```python
train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
```

### 5. 创建数据集

```python
train_dataset = DynamicPaddingDataset(
    samples=train_samples,
    tokenizer=tokenizer,
    max_length=train_config.get('max_length', 4096),
    use_profile=use_profile,
    use_history=use_history,
    use_context=use_context,
    verbose=is_main_process,
    use_detailed_template=(args.prompt_style != 'simple')
)
```

---

## ⚙️ 支持的所有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config_LovinkDialogue.json` |
| `--ablation_config` | 消融实验配置 | **必需** |
| `--val_ratio` | 验证集比例 | `0.1` |
| `--max_epochs` | 最大训练轮次 | `50` |
| `--early_stopping_patience` | 早停耐心值 | `3` |
| `--early_stopping_threshold` | 早停阈值 | `0.001` |
| `--output_dir` | 输出目录 | 自动生成 |
| `--wandb_project` | W&B 项目名 | `Qwen3-LovinkDialogue` |
| `--wandb_run_name` | W&B 运行名 | 自动生成 |
| `--deepspeed` | DeepSpeed 配置 | `None` |
| `--prompt_style` | Prompt 风格 | `simple` |
| `--max_samples_per_user` | 每用户最大样本数 | `None` (不采样) |
| `--sample_seed` | 采样随机种子 | `42` |
| `--disable_flash_attn` | 禁用 FlashAttention 2 | `False` |
| `--local_rank` | 本地进程 rank | `-1` (自动) |

---

## 📈 预期效果

### 使用 `--max_samples_per_user 2`

假设原始数据：
- 用户数：500
- 每用户平均样本数：20
- 总样本数：10,000

采样后：
- 总样本数：1,000 (每用户最多2个)
- 训练时间缩短：**10x**
- 采样比例：10%

---

## ⚠️ 重要提示

### 1. 不会打印 "使用详细 Prompt 模板"

当 `--prompt_style simple` 时，会打印：
```
✅ 使用简短 Prompt 模板 (data_loader.build_simple_training_prompt - 只预测continuation)
```

### 2. Prompt 格式

简洁格式示例：
```
[USER_HASH=user_001]
[PROFILE]
用户: 张三

[HISTORY]
历史消息1
历史消息2

[CONTEXT]
User: 你好
Assistant: 你好！

预测用户的下一条消息:
```

### 3. 数据不扩充

使用 `data_loader.extract_training_samples()`，只预测 continuation，不进行数据扩充。

---

## 🔧 故障排查

### 如果遇到问题

```bash
# 1. 清理 Python 缓存
find /mnt/parallel/CompactSubset_experiement -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /mnt/parallel/CompactSubset_experiement -name "*.pyc" -delete

# 2. 验证文件语法
python3 -m py_compile train_distributed_LovinkDialogue.py

# 3. 检查活跃的 main() 函数
grep -n "^if __name__ == '__main__':" train_distributed_LovinkDialogue.py
# 应该只显示一行：2873:if __name__ == '__main__':

# 4. 重新运行
./run_lovink_standalone.sh
```

---

## 🎉 完成状态

✅ **所有代码都在一个文件中**  
✅ **不依赖外部模块**  
✅ **支持所有命令行参数**  
✅ **支持 DeepSpeed ZeRO-3**  
✅ **支持用户采样**  
✅ **支持简洁 Prompt 格式**  
✅ **8卡分布式训练**  
✅ **FlashAttention 2 支持**  

---

## 📞 快速命令

```bash
# 运行训练
torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_distributed_LovinkDialogue.py \
    --config config_LovinkDialogue_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_context \
    --output_dir outputs/LovinkDialogue_profile_context_sampled_seed42 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-LovinkDialogue \
    --wandb_run_name profile_context_sampled_seed42 \
    --prompt_style simple \
    --max_samples_per_user 2 \
    --sample_seed 42
```

🚀 **Ready to train!**
