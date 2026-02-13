# MovieReview Prompt 格式说明

## 🎯 核心结论

**`MovieReviewDataset` 使用自定义的简单 Prompt 格式，不受 `use_detailed_template` 影响！**

---

## 📝 实际使用的 Prompt 格式

### MovieReviewDataset.format_prompt()

```python
def format_prompt(self, sample: Dict[str, Any]) -> str:
    """
    覆盖父类方法，使用影评专用格式
    """
    parts = []
    
    # 1. 用户Profile
    if self.use_profile and sample.get('user_profile'):
        profile = sample['user_profile']
        parts.append(f"用户: {profile.get('name', 'Unknown')}")
        if sample.get('task_description'):
            parts.append(f"任务: {sample['task_description']}")
        parts.append("")
    
    # 2. 历史影评（如果启用）
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

## 📋 Prompt 示例

### profile_and_history 模式

```
用户: user_13162
任务: 基于用户在 MovieLens 上的历史评分和标签数据，模拟该用户的电影偏好和行为模式

历史影评记录 (21条):
  电影《钢铁侠1》: boring
  电影《复仇者联盟》: Again！Again！Again！
  电影《泰囧》: 好笑又有启发性，难得的国产电影
  电影《十二生肖》: 权相佑纯打酱油的啊。
  电影《霍比特人1》: 还算不错，中土的景色很美...
  ...（共21条）

模仿用户风格为电影《美国队长3》写一条影评：
```

### profile_only 模式

```
用户: user_13162
任务: 基于用户在 MovieLens 上的历史评分和标签数据，模拟该用户的电影偏好和行为模式

模仿用户风格为电影《美国队长3》写一条影评：
```

---

## ⚠️ 关于 "使用详细 Prompt 模板" 的打印

### 为什么会打印？

```python
# train_distributed_MovieReview.py
train_dataset = MovieReviewDataset(
    samples=train_samples,
    tokenizer=tokenizer,
    # ... 其他参数
    use_detailed_template=False  # 🔥 新增：避免打印误导信息
)
```

### 父类 DynamicPaddingDataset.__init__()

```python
def __init__(self, ..., use_detailed_template=True, ...):
    if use_detailed_template:
        from prompt_builder_LovinkDialogue import build_training_prompt
        print("ℹ️  使用详细 Prompt 模板 (prompt_builder_LovinkDialogue)")  # ⬅️ 这里打印
        self.build_training_prompt = build_training_prompt
    else:
        from data_loader import build_simple_training_prompt
        print("ℹ️  使用简洁标签格式 (data_loader)")
        self.build_training_prompt = build_simple_training_prompt
```

### 实际行为

虽然父类会打印，但 `MovieReviewDataset` **覆盖了 `format_prompt` 方法**，所以：

1. ❌ **不会使用** `prompt_builder_LovinkDialogue`
2. ❌ **不会使用** `data_loader.build_simple_training_prompt`
3. ✅ **只会使用** `MovieReviewDataset.format_prompt()`

**打印信息是误导性的，但不影响实际功能！**

---

## 🔧 修复方案

### 方案 1: 传递 use_detailed_template=False（已修复）

```python
train_dataset = MovieReviewDataset(
    samples=train_samples,
    tokenizer=tokenizer,
    max_length=train_config.get('max_length', 4096),
    use_profile=use_profile,
    use_history=use_history,
    use_context=False,
    verbose=is_main_process,
    use_detailed_template=False  # ⬅️ 避免误导性打印
)
```

**效果**：打印信息会变成
```
ℹ️  使用简洁标签格式 (data_loader)
```

但实际上还是使用 `MovieReviewDataset.format_prompt()`。

### 方案 2: 不打印（更彻底）

修改 `DynamicPaddingDataset.__init__()` 添加条件：

```python
def __init__(self, ..., verbose=False, ...):
    if use_detailed_template:
        from prompt_builder_LovinkDialogue import build_training_prompt
        if verbose:  # ⬅️ 只在 verbose 时打印
            print("ℹ️  使用详细 Prompt 模板")
        self.build_training_prompt = build_training_prompt
```

---

## ✅ 验证实际使用的 Prompt

### 查看训练样本预览

```bash
cat outputs/DMSC_one_per_user_0213/training_samples_preview.txt
```

应该看到类似：
```
================================================================================
样本 1
================================================================================

电影: 釜山行
时间: 2016-09-12
历史影评: 21条
目标影评: 部分情节弱智得想骂街。。...
编码长度: 264 tokens
```

**不会**看到复杂的 markdown 格式或 `{VAR_NAME}` 占位符。

---

## 🎯 总结

| 方面 | 实际情况 |
|------|---------|
| **打印信息** | "使用详细 Prompt 模板" (误导) |
| **实际使用** | `MovieReviewDataset.format_prompt()` (简单格式) |
| **Prompt 风格** | 简单文本拼接，不是模板 |
| **是否受影响** | ❌ 不受 `use_detailed_template` 影响 |
| **是否受影响** | ❌ 不受 `--prompt_style` 影响 |

**结论**：打印信息可以忽略，实际功能正确！
