# 独立训练脚本改造总结

## 🎯 目标

将训练脚本改造为**完全独立**的版本，不依赖任何外部模块，所有代码都在一个文件中。

---

## ✅ 已完成的改造

### 1. train_distributed_MovieReview.py

**改造内容**：
- ✅ 添加了 `sample_per_user()` 函数
- ✅ 简化了 `DynamicPaddingDataset`，移除外部导入
- ✅ `MovieReviewDataset` 实现了 `format_prompt()`
- ✅ 注释掉了 3 个旧的 `if __name__ == '__main__':` 块
- ✅ 只保留一个真正的 main() 函数

**使用命令**：
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

**文档**：`MOVIEREVIEW_STANDALONE_SUMMARY.md`

---

### 2. train_distributed_LovinkDialogue.py

**改造内容**：
- ✅ 注释掉了外部导入（第44-46行）
- ✅ 内联了所有需要的函数：
  - `sample_per_user()`
  - `split_train_val()`
  - `add_history_to_samples()`
  - `DynamicPaddingDataset`
  - `dynamic_padding_collate_fn()`
- ✅ 注释掉了旧的 `if __name__ == '__main__':` 块（第1897行）
- ✅ 保留了支持所有参数的 main() 函数（第1957行）

**使用命令**：
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

**文档**：`LOVINKDIALOGUE_STANDALONE_SUMMARY.md`

---

## 📝 改造方法总结

### 步骤 1：备份原文件

```bash
cp train_distributed_XXX.py train_distributed_XXX.py.backup
```

### 步骤 2：识别外部导入

查找所有外部导入：
```bash
grep -n "^from (data_loader|sample_per_user|train_with_dynamic_padding)" train_distributed_XXX.py
```

### 步骤 3：注释外部导入

```bash
sed -i '行号s/^/# /' train_distributed_XXX.py
```

### 步骤 4：内联缺失的函数

创建包含所有需要函数的文件，然后插入：
```bash
sed -i '插入位置r functions_to_insert.txt' train_distributed_XXX.py
```

### 步骤 5：处理多个 main() 函数

识别所有 `if __name__ == '__main__':` 块：
```bash
grep -n "^if __name__ == '__main__':" train_distributed_XXX.py
```

注释掉旧的块，只保留最新的：
```bash
sed -i '行号s/^/# /' train_distributed_XXX.py
```

### 步骤 6：验证语法

```bash
python3 -m py_compile train_distributed_XXX.py
```

### 步骤 7：测试运行

```bash
torchrun --nproc_per_node=8 train_distributed_XXX.py --config ... --ablation_config ...
```

---

## 🛠️ 通用工具函数模板

### sample_per_user()

```python
def sample_per_user(
    all_samples: List[Dict[str, Any]],
    max_samples_per_user: int = 2,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """对每个用户的样本进行随机采样"""
    random.seed(random_seed)
    
    user_samples = {}
    for sample in all_samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    sampled_samples = []
    for user_hash, samples in user_samples.items():
        if len(samples) <= max_samples_per_user:
            sampled_samples.extend(samples)
        else:
            sampled = random.sample(samples, max_samples_per_user)
            sampled_samples.extend(sampled)
    
    return sampled_samples
```

### split_train_val()

```python
def split_train_val(samples, val_ratio=0.15, seed=42):
    """划分训练集和验证集（用户内划分）"""
    random.seed(seed)
    
    user_samples = {}
    for sample in samples:
        user_hash = sample['user_hash']
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    train_samples = []
    val_samples = []
    
    for user_hash, user_data in user_samples.items():
        random.shuffle(user_data)
        split_idx = int(len(user_data) * (1 - val_ratio))
        if split_idx == 0 and len(user_data) > 0:
            split_idx = 1
        train_samples.extend(user_data[:split_idx])
        val_samples.extend(user_data[split_idx:])
    
    return train_samples, val_samples
```

### dynamic_padding_collate_fn()

```python
def dynamic_padding_collate_fn(examples, tokenizer):
    """动态Padding的collate函数"""
    max_length_in_batch = max(ex['input_ids'].shape[0] for ex in examples)
    
    batch = {}
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ex in examples:
        seq_len = ex['input_ids'].shape[0]
        pad_len = max_length_in_batch - seq_len
        
        padded_input_ids.append(
            torch.cat([
                ex['input_ids'],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
        )
        
        padded_attention_mask.append(
            torch.cat([
                ex['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        
        padded_labels.append(
            torch.cat([
                ex['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )
    
    batch['input_ids'] = torch.stack(padded_input_ids)
    batch['attention_mask'] = torch.stack(padded_attention_mask)
    batch['labels'] = torch.stack(padded_labels)
    
    return batch
```

---

## 🔍 故障排查清单

### 问题 1：找不到模块

**症状**：`ModuleNotFoundError: No module named 'xxx'`

**解决**：
1. 检查是否还有未注释的外部导入
2. 确保所有需要的函数都已内联

### 问题 2：多个 main() 函数冲突

**症状**：`FileNotFoundError: [Errno 2] No such file or directory: '--config'`

**解决**：
1. 查找所有 `if __name__ == '__main__':` 块
2. 注释掉旧的块，只保留最新的

### 问题 3：缺少命令行参数

**症状**：`error: unrecognized arguments: --deepspeed ...`

**解决**：
1. 确保活跃的 main() 函数有完整的参数定义
2. 检查 `parser.add_argument()` 是否包含所有需要的参数

### 问题 4：Python 缓存问题

**症状**：修改代码后，旧代码仍在运行

**解决**：
```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
```

---

## 📊 改造前后对比

| 方面 | 改造前 | 改造后 |
|------|--------|--------|
| **文件数量** | 主脚本 + 3-5个依赖模块 | 单个主脚本 |
| **导入依赖** | 依赖外部模块 | 完全独立 |
| **可移植性** | 需要复制多个文件 | 只需一个文件 |
| **调试难度** | 需要跨文件查找 | 所有代码在一个文件 |
| **维护成本** | 多文件同步更新 | 单文件更新 |
| **运行风险** | 模块版本冲突 | 无外部依赖 |

---

## 🎓 最佳实践

### 1. 保持代码结构清晰

使用注释分隔不同功能模块：
```python
# ============================================================================
# 工具函数：用户采样
# ============================================================================

def sample_per_user(...):
    ...

# ============================================================================
# 数据加载模块
# ============================================================================

def load_train_data(...):
    ...
```

### 2. 保留原文件备份

```bash
cp original_file.py original_file.py.backup
```

### 3. 分步验证

每次修改后都验证语法：
```bash
python3 -m py_compile modified_file.py
```

### 4. 文档同步更新

创建对应的 `XXX_STANDALONE_SUMMARY.md` 文档

---

## 📦 交付清单

### train_distributed_MovieReview.py
- [x] 独立训练脚本
- [x] 使用说明文档
- [x] 快速启动脚本 `run_dmsc_one_sample_per_user.sh`

### train_distributed_LovinkDialogue.py
- [x] 独立训练脚本
- [x] 使用说明文档
- [x] 快速启动脚本 `run_lovink_standalone.sh`

### 通用文档
- [x] 改造方法总结（本文档）
- [x] 故障排查指南
- [x] 最佳实践建议

---

## 🎉 成功标志

✅ **语法验证通过**  
✅ **无外部导入**  
✅ **命令行参数完整**  
✅ **实际运行成功**  
✅ **文档完整清晰**  

---

## 📞 快速参考

### MovieReview 训练

```bash
./run_dmsc_one_sample_per_user.sh
```

### LovinkDialogue 训练

```bash
./run_lovink_standalone.sh
```

### 清理缓存

```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
```

### 验证语法

```bash
python3 -m py_compile train_distributed_*.py
```

---

🚀 **所有训练脚本现已独立，可以直接使用！**
