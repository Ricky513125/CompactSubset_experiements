# MovieReview 独立训练脚本说明

## ✅ 修改完成

`train_distributed_MovieReview.py` 现在是一个**完全独立**的训练脚本，不依赖任何外部模块。

---

## 📝 主要修改

### 1. 添加了 `sample_per_user` 函数（第40-88行）

```python
def sample_per_user(
    samples: List[Dict[str, Any]],
    max_samples_per_user: int,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """对每个用户的样本进行随机采样"""
```

**功能**：
- 按 `user_hash` 分组
- 每个用户最多保留 `max_samples_per_user` 个样本
- 使用固定随机种子保证可复现

---

### 2. 简化了 `DynamicPaddingDataset`（第431-491行）

**移除了外部导入**：
- ❌ 不再导入 `prompt_builder_LovinkDialogue`
- ❌ 不再导入 `data_loader` 或 `data_loader_more_data`

**使用模板方法模式**：
- 添加了 `format_prompt()` 方法（应该被子类覆盖）
- `__getitem__()` 调用 `format_prompt()`
- `MovieReviewDataset` 覆盖 `format_prompt()` 实现影评专用格式

```python
class DynamicPaddingDataset(Dataset):
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """应该被子类覆盖"""
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        prompt_text = self.format_prompt(sample)  # 调用子类的实现
        # ...
```

---

### 3. MovieReviewDataset 实现了 format_prompt（第1177-1218行）

```python
class MovieReviewDataset(DynamicPaddingDataset):
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """影评专用格式"""
        parts = []
        
        # 1. 用户Profile
        if self.use_profile and sample.get('user_profile'):
            profile = sample['user_profile']
            parts.append(f"用户: {profile.get('name', 'Unknown')}")
            if sample.get('task_description'):
                parts.append(f"任务: {sample['task_description']}")
            parts.append("")
        
        # 2. 历史影评
        if self.use_history and sample.get('history'):
            history = sample['history']
            parts.append(f"历史影评记录 ({len(history)}条):")
            for h in history:
                parts.append(f"  电影《{h['movie']}》: {h['review']}")
            parts.append("")
        
        # 3. 当前电影
        movie_name = sample.get('movie_name', '')
        parts.append(f"模仿用户风格为电影《{movie_name}》写一条影评：")
        
        return "\n".join(parts)
```

---

## 🚀 使用方法

### 基本命令（您的原始命令）

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_one_per_user_0213 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name one_per_user_0213 \
    --prompt_style simple \
    --one_sample_per_user
```

---

## 🔍 关键参数说明

### 数据模式参数

#### `--one_sample_per_user` (推荐)
**每个用户只生成1个训练样本**

- 使用前 n-1 条影评作为历史
- 预测第 n 条影评
- **大幅减少训练时间**（例如：5054个样本 → 150个样本）

```bash
--one_sample_per_user
```

#### `--max_samples_per_user N`
**每个用户最多采样N个样本**（与 `--one_sample_per_user` 互斥）

- 用于进一步控制数据量
- 适用于默认模式（每条影评一个样本）

```bash
--max_samples_per_user 10 \
--sample_seed 42
```

---

## 📊 训练流程

### 1. 数据加载

```python
# 加载原始数据
raw_data = load_movie_review_data(data_file)

# 提取样本
all_samples = extract_movie_review_samples(
    raw_data,
    one_sample_per_user=args.one_sample_per_user,  # 🔥 控制模式
    debug=is_main_process
)
```

### 2. 采样（如果需要）

```python
if args.max_samples_per_user is not None and not args.one_sample_per_user:
    all_samples = sample_per_user(
        all_samples,
        max_samples_per_user=args.max_samples_per_user,
        random_seed=args.sample_seed
    )
```

### 3. 时间划分

```python
train_samples, val_samples, test_samples = split_movie_reviews_by_time(
    all_samples,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 4. 创建数据集

```python
train_dataset = MovieReviewDataset(
    samples=train_samples,
    tokenizer=tokenizer,
    max_length=train_config.get('max_length', 4096),
    use_profile=use_profile,
    use_history=use_history,
    use_context=False,
    verbose=is_main_process,
    use_detailed_template=False  # 使用简单格式
)
```

### 5. Token 长度统计

脚本会自动打印 token 长度统计信息，帮助您配置 `max_length`：

```
================================================================================
📊 Token 长度统计（训练集）
================================================================================
样本总数: 3538
配置的 max_length: 1024

Token 长度分布:
  最小长度: 82 tokens
  最大长度: 1015 tokens
  平均长度: 265.3 tokens
  中位数: 201 tokens

分位数:
  25%: 136 tokens
  50%: 201 tokens
  75%: 315 tokens
  90%: 503 tokens
  95%: 657 tokens
  99%: 892 tokens

✅ 所有样本都在 max_length=1024 范围内
================================================================================
```

---

## ⚠️ 提示

### 不会再打印 "使用详细 Prompt 模板"

之前的误导性打印已被移除：
- ❌ 旧版：`ℹ️  使用详细 Prompt 模板 (prompt_builder_LovinkDialogue)`
- ✅ 新版：直接使用 `MovieReviewDataset.format_prompt()` （无打印）

### 实际使用的 Prompt 格式

```
用户: user_13162
任务: 基于用户在 MovieLens 上的历史评分和标签数据，模拟该用户的电影偏好和行为模式

历史影评记录 (21条):
  电影《钢铁侠》: boring
  电影《复仇者联盟》: Again！Again！Again！
  ...

模仿用户风格为电影《美国队长3》写一条影评：
```

---

## 🎯 完成状态

✅ **所有代码都在一个文件中**  
✅ **不依赖外部模块**（`data_loader.py`, `prompt_builder_LovinkDialogue.py` 等）  
✅ **支持 `--one_sample_per_user` 模式**  
✅ **支持 `--max_samples_per_user` 采样**  
✅ **Token 长度统计功能**  
✅ **简洁的 Prompt 格式**  
✅ **8卡分布式训练**  
✅ **DeepSpeed Zero-3 支持**  

---

## 🔧 故障排查

### 如果遇到缓存问题

```bash
# 停止当前训练
# Ctrl+C 或 kill 进程

# 清理 Python 缓存
find /mnt/parallel/CompactSubset_experiement -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /mnt/parallel/CompactSubset_experiement -name "*.pyc" -delete

# 重新运行训练命令
```

### 验证代码是否正确

```bash
# 检查 extract_movie_review_samples 函数
grep -A 5 "if one_sample_per_user:" train_distributed_MovieReview.py

# 检查 sample_per_user 函数
grep -A 3 "def sample_per_user" train_distributed_MovieReview.py

# 检查 MovieReviewDataset.format_prompt
grep -A 5 "class MovieReviewDataset" train_distributed_MovieReview.py
```

---

## 📞 总结

您现在有一个**完全独立**的训练脚本，可以直接运行您的命令：

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    train_distributed_MovieReview.py \
    --config config_DMSC_30B.json \
    --deepspeed ds_config_zero3_optimized.json \
    --ablation_config profile_and_history \
    --output_dir outputs/DMSC_one_per_user_0213 \
    --max_epochs 50 \
    --val_ratio 0.1 \
    --wandb_project Qwen3_30B-DMSC \
    --wandb_run_name one_per_user_0213 \
    --prompt_style simple \
    --one_sample_per_user
```

**预期结果**：
- 每个用户生成 1 个训练样本
- 训练样本数 ≈ 用户数（约150-200个）
- 不会再出现 "使用详细 Prompt 模板" 的打印
- Token 长度统计会显示在训练开始前

🎉 **Ready to train!**
