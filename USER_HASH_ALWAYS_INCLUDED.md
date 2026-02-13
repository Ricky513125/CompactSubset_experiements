# USER_HASH 始终包含在训练 Prompt 中

## 修改说明

为了确保模型能够识别不同用户，现在 **`user_hash` 始终包含在训练 prompt 中**，无论 `use_profile` 是否启用。

## 修改的文件

### 1. `data_loader.py`
- ✅ 在 `build_simple_training_prompt` 中添加 `user_hash` 参数
- ✅ 在 prompt 开头添加 `[USER_HASH=xxx]` 标签（始终包含）

### 2. `data_loader_more_data.py`
- ✅ 在 `build_simple_training_prompt` 中添加 `user_hash` 参数
- ✅ 在 prompt 开头添加 `[USER_HASH=xxx]` 标签（始终包含）

### 3. `train_with_dynamic_padding_Lovink.py`
- ✅ 在调用 `build_training_prompt` 时传递 `user_hash=sample.get('user_hash')`

## Prompt 格式变化

### 修改前（context_only）
```
<|im_start|>system
[TASK]
Given the historical dialogue of a character in a movie...

[RECENT_DIALOGUE]
User: ...
Assistant: ...

Predict the user's next message:<|im_end|>
```

### 修改后（context_only）
```
<|im_start|>system
[USER_HASH=2dc9c56e09d7479f8b6d69dfc4a5e6e4]

[TASK]
Given the historical dialogue of a character in a movie...

[RECENT_DIALOGUE]
User: ...
Assistant: ...

Predict the user's next message:<|im_end|>
```

### 如果启用 profile（profile_and_context）
```
<|im_start|>system
[USER_HASH=2dc9c56e09d7479f8b6d69dfc4a5e6e4]

[USER_PROFILE]
[USER_NAME=DANTE]
[USER_GENDER=m]

[TASK]
Given the historical dialogue of a character in a movie...

[RECENT_DIALOGUE]
User: ...
Assistant: ...

Predict the user's next message:<|im_end|>
```

## 为什么这样设计？

### 1. 用户标识始终可见
- ✅ **训练时**：模型知道当前在预测哪个用户
- ✅ **验证时**：相同的用户集，模型能识别
- ✅ **测试时**：相同的用户集，模型能识别

### 2. Profile 由消融配置控制
- ✅ `context_only`：只有 user_hash，没有 profile（name, gender 等）
- ✅ `profile_and_context`：user_hash + profile（name, gender 等）
- ✅ `profile_and_history_and_context`：user_hash + profile + history

### 3. 适合 In-User 评估
你提到：
> "我的训练验证和最终的测试都是一批user"

这意味着：
- 训练、验证、测试使用**相同的用户集**
- 每个用户有多个对话样本
- 数据划分是 **in-user splitting**（同一用户的不同对话分散在训练/验证/测试集中）

在这种设置下，`user_hash` 非常重要：
1. 模型可以学习**每个用户的特定模式**
2. 即使在 `context_only` 配置下，模型也知道是哪个用户
3. 测试时，模型可以利用训练时学到的用户特定知识

## 对比：Out-of-User 评估

如果是 **out-of-user splitting**（测试集的用户在训练集中从未见过）：
- 训练用户：User A, B, C
- 测试用户：User D, E, F（全新用户）

在这种情况下，`user_hash` 就不太有用，因为测试时的 hash 是新的。

但你的情况是 **in-user splitting**，所以 `user_hash` 很有用！

## 消融实验配置

### 现有配置（config_Chameleons.json）

```json
{
  "ablation_configs": {
    "history_and_context": {
      "use_profile": false,
      "use_history": true,
      "use_context": true
    },
    "history_only": {
      "use_profile": false,
      "use_history": true,
      "use_context": false
    },
    "context_only": {
      "use_profile": false,
      "use_history": false,
      "use_context": true
    }
  }
}
```

### 各配置的 Prompt 内容

| 配置 | user_hash | profile | history | context |
|------|-----------|---------|---------|---------|
| `context_only` | ✅ | ❌ | ❌ | ✅ |
| `history_only` | ✅ | ❌ | ✅ | ❌ |
| `history_and_context` | ✅ | ❌ | ✅ | ✅ |
| `profile_and_context` | ✅ | ✅ | ❌ | ✅ |
| `profile_and_history_and_context` | ✅ | ✅ | ✅ | ✅ |

**所有配置都包含 `user_hash`！**

## 预期效果

### 训练时
模型会学习：
- `[USER_HASH=aaa...]` → 某种对话风格
- `[USER_HASH=bbb...]` → 另一种对话风格

### 测试时（相同用户集）
模型会利用训练时学到的用户特定模式：
- 看到 `[USER_HASH=aaa...]` → 回忆起该用户的风格
- 生成更符合该用户特点的回复

## 验证

重新训练后，检查日志中的第一个样本：

```
【完整的输入文本】
<|im_start|>system
[USER_HASH=2dc9c56e09d7479f8b6d69dfc4a5e6e4]  ← ✅ 应该出现这一行

[TASK]
...
```

## 注意事项

⚠️ **重要**：
1. 旧的 checkpoint 没有见过 `[USER_HASH=...]` 标签，可能需要重新训练
2. 如果你的测试集是 **out-of-user**（新用户），考虑移除 user_hash
3. 对于 Chameleons（电影角色），user_hash 实际上是角色标识（character ID）

✅ **建议**：
- 对于 in-user 评估：保留 user_hash（当前配置）
- 对于 out-of-user 评估：可以考虑移除或使用特殊占位符
