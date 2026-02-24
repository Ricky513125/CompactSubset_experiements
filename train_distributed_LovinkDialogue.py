
import json
import argparse
import os
import sys
from pathlib import Path
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



# ============================================================================
# 导入必要的库（用于分布式训练）
# ============================================================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import torch.nn as nn


"""
数据加载模块 - 简短 Prompt 版本 用于训练，只使用continuation 
用于加载 LovinkDialogue 数据集，使用简短的 prompt 格式进行训练
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 导入缓存模块（现在在同一目录下）
try:
    from history_cache import save_history, load_history
except ImportError:
    # 如果失败，提供一个简单的占位实现
    print("⚠️ 无法导入 history_cache，使用占位实现")
    def save_history(user_hash, history):
        pass
    def load_history(user_hash):
        return None


def load_json_file(file_path: str) -> Any:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_train_data(train_path: str) -> List[Dict[str, Any]]:
    """加载训练数据"""
    if not os.path.exists(train_path):
        return []
    return load_json_file(train_path)


def get_user_profile(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    获取完整的用户 profile 信息
    合并 user.profile (基础信息) 和 user.personality.structured (心理测量数据)
    
     重要：保留完整的层级结构，以便 prompt_builder 可以正确提取和填充模板占位符
    
    数据结构：
    {
      'name': 'xxx',
      'dimensions': {
        'BIRI': {
          'explanation': '...',
          'dimensions': {
            'PerspectiveTaking': {'score': 65, 'description': '...'},
            ...
          }
        },
        ...
      }
    }
    """
    user = sample.get('user', {})
    
    # 获取基础 profile
    profile = user.get('profile', {})
    full_profile = dict(profile) if isinstance(profile, dict) else {}
    
    # 获取 personality 数据
    personality = user.get('personality', {})
    
    # ✅ 保留完整的 structured 数据结构（不扁平化）
    structured = personality.get('structured', {})
    if structured:
        # 直接保存完整的 structured 数据，包括 score, description, explanation 等
        full_profile['dimensions'] = structured
    
    # ✅ 同时保存 unstructured 文本分析（如果存在）
    unstructured = personality.get('unstructured', '')
    if unstructured:
        full_profile['unstructured'] = unstructured
    
    return full_profile if full_profile else None

def get_task_description(sample: Dict[str, Any]) -> str:
    """获取任务描述"""
    task = sample.get('task', {})
    return task.get('description', '')


def extract_training_samples(train_data: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    """
    从训练数据中提取训练样本
    目标：将'user'设为目标角色，'user_wxxxx'设为对话者(assistant)
    训练模式：[History...] -> Predict Next 'user' Turn
    """
    samples = []
    target_role_name = "user"  # 我们要预测的那个人的 source 标识符

    if debug:
        print(f"\n开始提取训练样本，总数据项数: {len(train_data)}\n" + "="*50)

    for item_idx, item in enumerate(train_data):
        user_hash = item.get('user_hash', '')
        user_profile = get_user_profile(item)
        # 获取完整的user对象（包含personality），用于人格映射
        user_object = item.get('user', {})
        task_description = get_task_description(item)
        
        # 获取用户的 name（用于识别 context 中的用户消息）
        user_name = None
        if user_profile and isinstance(user_profile, dict):
            user_name = str(user_profile.get('name', '')).strip()
        
        # 提取对话集合
        collections = item.get('task', {}).get('task_behavior_collections', [])
        
        for collection in collections:
            for data_item in collection.get('data', []):
                context = data_item.get('context', [])
                continuation = data_item.get('continuation', '').strip()
                
                # 跳过无效数据
                if not continuation:
                    continue
                
                # 构建对话上下文（转换为标准格式）
                full_dialogue = []
                for turn in context:
                    source = str(turn.get('source', '')).strip()
                    content = turn.get('content', '').strip()
                    
                    if not content:
                        continue
                    
                    # 判断是否是目标用户（我们要预测的人）说的话
                    is_target_user = False
                    if source.lower() == 'user':
                        is_target_user = True
                    elif user_name and source == user_name:
                        is_target_user = True
                    
                    # 目标用户的话映射为 assistant（模型要学习生成的）
                    # 对话者的话映射为 user（输入/上下文）
                    role = "assistant" if is_target_user else "user"
                    full_dialogue.append({"role": role, "content": content})
                
                # ✅ 简化逻辑：只预测 continuation，不进行数据扩充
                # 只创建一个样本：context -> continuation
                if len(full_dialogue) > 0 and full_dialogue[-1]['role'] == 'user':
                    # 确保 context 的最后一轮是 user (对话者)
                    # 这样符合 LLM "user输入 -> assistant生成" 的标准逻辑
                        samples.append({
                            'context': full_dialogue,           # 包含 role 和 content 的列表
                            'next_question': continuation,      # 目标文本（continuation）
                        'user_profile': user_profile,       # profile部分
                        'user_object': user_object,         # 完整的user对象（包含personality）
                            'task_description': task_description,
                            'user_hash': user_hash
                        })
                elif len(full_dialogue) == 0:
                    # 如果没有 context，直接预测 continuation（针对首次发言）
                                samples.append({
                        'context': [],
                        'next_question': continuation,
                        'user_profile': user_profile,
                        'user_object': user_object,
                                    'task_description': task_description,
                                    'user_hash': user_hash
                                })
    # --- 新增：保存样本逻辑 ---
    # 保存到当前项目目录下的 data 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "data", "extracted_samples")
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    
    # 建议保存为 .jsonl 格式，方便大规模数据处理
    save_path = os.path.join(save_dir, "extracted_samples.jsonl")
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✅ 成功将 {len(samples)} 个样本保存至: {save_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

        
    if debug:
        print(f"提取完成！有效样本总数: {len(samples)}\n" + "="*50)
    return samples


def get_user_only_history(
    all_samples: List[Dict[str, Any]], 
    user_hash: str,
    current_sample: Optional[Dict[str, Any]] = None,
    current_context: Optional[List[Dict[str, str]]] = None,
    max_history: int = 15,
    use_cache: bool = True
) -> List[str]:
    """
    获取用户历史发送的问题列表，用于 Few-shot 或 RAG 增强
    
    用户可能在数据中的 source 命名为 "user"，也可能是在 profile 里显示的 name（如 "HP", "AH" 等）
    需要从 context 中识别用户说的话，以及 next_question
    
    Args:
        all_samples: 所有训练样本
        user_hash: 用户哈希
        current_sample: 当前样本（用于排除）
        current_context: 当前context（用于排除和智能选择）
        max_history: 最大历史数量
        use_cache: 是否使用缓存
    
    Returns:
        用户历史对话列表（只包含用户说的话）
    """
    # 尝试从缓存加载
    if use_cache:
        cached_history = load_history(user_hash)
        if cached_history is not None:
            # 如果提供了current_context，需要排除其中的内容并智能选择
            if current_context:
                # 提取当前context中用户说的话
                current_user_contents = set()
                user_profile = None
                user_name = None
                
                # 获取用户名称
                for s in all_samples:
                    if s.get('user_hash') == user_hash:
                        user_profile = s.get('user_profile')
                        if user_profile and isinstance(user_profile, dict):
                            user_name = user_profile.get('name', '').strip()
                        break
                
                # 从current_context中提取预测目标用户（assistant）说的话
                for turn in current_context:
                    if isinstance(turn, dict):
                        # 如果turn是已处理格式（有role字段）
                        # 注意：预测目标用户的话被映射为assistant（模型要生成的）
                        if 'role' in turn and turn['role'] == 'assistant':
                            content = turn.get('content', '').strip()
                            if content:
                                current_user_contents.add(content)
                        # 如果turn是原始格式（有source字段）
                        elif 'source' in turn:
                            source = str(turn.get('source', '')).strip()
                            content = turn.get('content', '').strip()
                            is_user_turn = (source.lower() == 'user') or (user_name and source == user_name)
                            if is_user_turn and content:
                                current_user_contents.add(content)
                
                # 从缓存的历史中排除当前context中的内容
                filtered_history = [h for h in cached_history if h not in current_user_contents]
                
                # 如果长度过长，智能选择最相关的
                if len(filtered_history) > max_history:
                    return select_relevant_history(filtered_history, current_context, max_history)
                else:
                    return filtered_history[:max_history]
            else:
                # 没有current_context，直接返回缓存
                return cached_history[:max_history]
    
    # 缓存未命中或未启用缓存，重新计算
    user_history = []
    user_profile = None
    user_name = None
    
    # 获取用户 profile 名称
    for s in all_samples:
        if s.get('user_hash') == user_hash:
            user_profile = s.get('user_profile')
            if user_profile and isinstance(user_profile, dict):
                user_name = user_profile.get('name', '').strip()
            break
    
    # 提取当前context中预测目标用户（assistant）说的话（用于排除）
    current_user_contents = set()
    if current_context:
        for turn in current_context:
            if isinstance(turn, dict):
                # 已处理格式（注意：预测目标用户的话被映射为assistant）
                if 'role' in turn and turn['role'] == 'assistant':
                    content = turn.get('content', '').strip()
                    if content:
                        current_user_contents.add(content)
                # 原始格式
                elif 'source' in turn:
                    source = str(turn.get('source', '')).strip()
                    content = turn.get('content', '').strip()
                    is_user_turn = (source.lower() == 'user') or (user_name and source == user_name)
                    if is_user_turn and content:
                        current_user_contents.add(content)
    
    # 遍历所有样本
    for s in all_samples:
        if s.get('user_hash') != user_hash:
            continue  # 不同用户，跳过
        
        if current_sample is not None and s == current_sample:
            continue  # 跳过当前 sample
        
        context = s.get('context', [])
        if context:
            for turn in context:
                # 处理已处理格式（有role字段）
                # 注意：预测目标用户的话被映射为assistant（模型要生成的）
                if isinstance(turn, dict) and 'role' in turn:
                    if turn['role'] == 'assistant':
                        content = turn.get('content', '').strip()
                        if content and content not in current_user_contents:
                            user_history.append(content)
                # 处理原始格式（有source字段）
                elif isinstance(turn, dict) and 'source' in turn:
                    source = str(turn.get('source', '')).strip()
                    content = turn.get('content', '').strip()
                    
                    is_user_turn = (source.lower() == 'user') or (user_name and source == user_name)
                    if is_user_turn and content and content not in current_user_contents:
                        user_history.append(content)
        
        # next_question 也算用户说的话（但要排除当前context中的内容）
        q = s.get('next_question', '').strip()
        if q and q not in current_user_contents:
            user_history.append(q)
    
    # 去重并限制数量
    seen = set()
    unique_history = []
    for item in reversed(user_history):
        if item and item not in seen:
            seen.add(item)
            unique_history.append(item)
            if len(unique_history) >= max_history * 2:  # 先收集更多，用于智能选择
                break
    
    result = list(reversed(unique_history))
    
    # 如果长度过长，智能选择最相关的
    if len(result) > max_history and current_context:
        result = select_relevant_history(result, current_context, max_history)
    else:
        result = result[:max_history]
    
    # 保存到缓存（如果启用缓存）
    if use_cache and result:
        save_history(user_hash, result)
    
    return result


# ============================================================================
# 简短 Prompt 构建函数 - 用于训练
# ============================================================================

def build_simple_training_prompt(
    context: List[Dict[str, str]],
    next_question: str,
    user_profile: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    history: Optional[List[Any]] = None,
    use_profile: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    tokenizer = None,
    max_length: int = 8192,
    min_target_tokens: int = 64,
    user_hash: Optional[str] = None  # 新增：用户哈希
) -> tuple[List[Dict[str, str]], str]:
    """
    构建简短的训练 prompt，带动态长度调整
    
    格式:
    [USER_PROFILE]
    {...}  # JSON 格式的完整 profile（包括心理维度）
    
    [TASK]
    基于用户在 Lovink 对话中的历史数据，模拟该用户的对话行为模式
    
    [RECENT_DIALOGUE]
    User: ...
    Assistant: ...
    
    Predict the user's next message:
    
    Args:
        context: 对话上下文
        next_question: 目标回复（用户下一句话）
        user_profile: 用户画像（包括 dimensions 心理维度数据）
        task_description: 任务描述（可选，默认使用 Lovink 标准描述）
        history: 历史对话（可选）
        use_profile: 是否使用用户画像
        use_history: 是否使用历史
        use_context: 是否使用上下文
        tokenizer: tokenizer 实例（用于精确长度计算和动态截断）
        max_length: 最大序列长度（默认 8192）
        min_target_tokens: 为 target 预留的最小 token 数（默认 64）
    
    Returns:
        (messages, target_answer): messages 用于模型输入，target_answer 是预测目标
        
    特性:
        - ✅ 动态调整 context 长度，确保不超过 max_length
        - ✅ 优先保留最近的对话轮次
        - ✅ 预留足够空间给 target 生成（min_target_tokens + 实际 target 长度）
        - ✅ 如果发生截断，会在对话开头添加省略提示
    """
    messages = []
    
    # 构建 system message（简短格式）
    system_parts = []
    
    # 0. USER_HASH 部分 - 始终包含（无论 use_profile 是否启用）
    if user_hash:
        system_parts.append(f"[USER_HASH={user_hash}]")
    
    # 1. USER_PROFILE 部分 - 使用方括号标签格式（由 use_profile 控制）
    if use_profile and user_profile:
        profile_tags = []
        
        # 基础信息标签
        if 'name' in user_profile and user_profile['name']:
            profile_tags.append(f"[USER_NAME={user_profile['name']}]")
        if 'age' in user_profile and user_profile['age']:
            profile_tags.append(f"[USER_AGE={user_profile['age']}]")
        if 'gender' in user_profile and user_profile['gender']:
            profile_tags.append(f"[USER_GENDER={user_profile['gender']}]")
        
        # 心理维度标签（dimensions）
        # 支持两种格式：
        # 1. 扁平化格式: {"Ocean.Extraversion": 90, ...}
        # 2. 嵌套格式: {"BIRI": {"dimensions": {"PerspectiveTaking": {"score": 65}}}}
        if 'dimensions' in user_profile and isinstance(user_profile['dimensions'], dict):
            dims = user_profile['dimensions']
            
            # 检查是否是扁平化格式（包含 "." 的键）
            is_flat = any('.' in str(k) for k in dims.keys())
            
            if is_flat:
                # 扁平化格式：直接遍历
                for dim_key, dim_score in dims.items():
                    if dim_score is not None:
                        # dim_key 格式: "Ocean.Extraversion"
                        # 转换为: [DIM_OCEAN_EXTRAVERSION=90]
                        tag_name = f"DIM_{dim_key.upper().replace('.', '_')}"
                        profile_tags.append(f"[{tag_name}={dim_score}]")
            else:
                # 嵌套格式：需要遍历两层
                for scale_name, scale_data in dims.items():
                    if isinstance(scale_data, dict) and 'dimensions' in scale_data:
                        subdims = scale_data['dimensions']
                        for subdim_name, subdim_data in subdims.items():
                            if isinstance(subdim_data, dict) and 'score' in subdim_data:
                                score = subdim_data['score']
                                # 生成标签: [DIM_BIRI_PERSPECTIVETAKING=65]
                                tag_name = f"DIM_{scale_name.upper()}_{subdim_name.upper()}"
                                profile_tags.append(f"[{tag_name}={score}]")
        
        # 其他 profile 字段（排除 dimensions 和已处理的字段）
        excluded_keys = {'name', 'age', 'gender', 'dimensions', 'unstructured'}
        for key, value in user_profile.items():
            if key not in excluded_keys and value:
                # 将其他字段也转为标签格式
                tag_name = f"USER_{key.upper()}"
                profile_tags.append(f"[{tag_name}={value}]")
        
        if profile_tags:
            system_parts.append("[USER_PROFILE]\n" + "\n".join(profile_tags))
    English_flag = False
    Japanese_flag = False
    # 2. TASK 部分 - 任务描述
    task_text = task_description if task_description else "基于用户在 Lovink 对话中的历史数据，模拟该用户的对话行为模式"
    if task_text == "基于角色在电影中的历史对话数据，模拟该角色的对话风格和行为模式": # Chameleons
        English_flag = True 
        task_text = "Given the historical dialogue of a character in a movie, model the character's speaking style and behavioral patterns, and predict the next utterance the user would produce."
    elif task_text == "基于用户在 Reddit 上的历史对话数据，模拟该用户的对话风格和行为模式":
        # English_flag = True 
        task_text = "Given the historical dialogue of a user on Reddit, model the user's speaking style and behavioral patterns, and predict the next utterance the user would produce."
    elif task_text == "基于用户在 RealPersonaChat 数据集中的历史对话数据，模拟该用户的对话行为模式":
        Japanese_flag = True 
        task_text = "RealPersonaChatデータセットにおけるユーザーの過去の会話データに基づき、当該ユーザーの会話行動パターンをシミュレートする："
    elif task_text == "基于用户在 REALTALK 数据集中的历史对话数据，模拟该用户的对话风格和行为模式":
        task_text = "Given the historical dialogue of a user on REALTALK, model the user's speaking style and behavioral patterns, and predict the next utterance the user would produce."
    system_parts.append(f"[TASK]\n{task_text}")
    
    # 2.5. HISTORY 部分 - 历史信息（在 TASK 和 RECENT_DIALOGUE 之间）
    if use_history and history and len(history) > 0:
        history_parts = ["[HISTORY]"]
        # 限制历史条目数量，避免过长
        max_history_items = 15
        history_to_use = history[:max_history_items] if len(history) > max_history_items else history
        
        for i, item in enumerate(history_to_use, 1):
            # 支持多种格式的历史
            if isinstance(item, str):
                content = item
            elif isinstance(item, dict):
                content = item.get('next_question', '') or item.get('content', '') or item.get('continuation', '')
            else:
                content = str(item)
            
            if content:
                # 截断过长的历史项
                if len(content) > 200:
                    content = content[:197] + "..."
                history_parts.append(f"{i}. {content}")
        
        if len(history_parts) > 1:  # 确保有实际内容
            if task_text and "MovieLens" in task_text:
                # MovieLens 特殊标题
                system_parts.append("[HISTORICAL_RATINGS]")
                system_parts.append("\n".join(history_parts[1:]))  # 跳过 [HISTORY] 标题
            else:
                system_parts.append("\n".join(history_parts))
    
    # 3. RECENT_DIALOGUE 部分 - 动态调整长度
    recent_context = context.copy() if use_context and context else []
    
    # 如果提供了 tokenizer，进行动态长度检查
    if tokenizer and use_context and recent_context:
        # 构建初始的 dialogue 部分（不加入 system_parts）
        def build_dialogue_section(ctx):
            dialogue_parts = ["[RECENT_DIALOGUE]"]
            for turn in ctx:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                label = "User" if role == 'user' else "Assistant" if role == 'assistant' else "Unknown"
                dialogue_parts.append(f"{label}: {content}")
            return "\n".join(dialogue_parts)
        
        # 估算 target 的 token 数（包括 [ANSWER] 和 [/ANSWER] 标签）
        target_answer_with_tags = f"[ANSWER]\n{next_question}\n[/ANSWER]"
        target_tokens = len(tokenizer.encode(target_answer_with_tags, add_special_tokens=False))
        
        # 预留空间：target + min_target_tokens 的缓冲 + 特殊 tokens
        reserved_tokens = target_tokens + min_target_tokens + 50  # 50 for special tokens
        max_prompt_tokens = max_length - reserved_tokens
        
        # 从最近的对话开始，逐步增加，直到接近限制
        # 策略：从后往前添加对话轮次
        truncated_context = []
        removed_turns = 0
        
        for i in range(len(recent_context) - 1, -1, -1):
            # 尝试添加这一轮对话
            test_context = [recent_context[i]] + truncated_context
            
            # 构建临时 system message 测试长度
            temp_system_parts = system_parts.copy()
            temp_system_parts.append(build_dialogue_section(test_context))
            temp_system_parts.append("\nPredict the user's next message:")
            temp_system_content = "\n\n".join(temp_system_parts)
            
            # 构建临时 messages 测试 tokenization
            temp_messages = [{"role": "system", "content": temp_system_content}]
            
            try:
                # 使用 apply_chat_template 估算实际长度
                prompt_tokens = len(tokenizer.apply_chat_template(temp_messages, tokenize=True, add_generation_prompt=False))
                total_tokens = prompt_tokens + target_tokens
                
                if total_tokens <= max_length:
                    # 还有空间，添加这一轮
                    truncated_context = test_context
                else:
                    # 超出限制，停止添加
                    removed_turns += 1
                    break
            except:
                # 如果 tokenization 失败，保守估计
                truncated_context = test_context
                break
        
        recent_context = truncated_context
        
        if removed_turns > 0 and len(truncated_context) > 0:
            # 在对话开头添加省略提示
            dialogue_parts = [f"[RECENT_DIALOGUE]\n（前面省略了 {removed_turns} 轮对话，以下是最近的对话）"]
        else:
            dialogue_parts = ["[RECENT_DIALOGUE]"]
        
        for turn in recent_context:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            label = "User" if role == 'user' else "Assistant" if role == 'assistant' else "Unknown"
            dialogue_parts.append(f"{label}: {content}")
        
        system_parts.append("\n".join(dialogue_parts))
    elif use_context and recent_context:
        # 没有 tokenizer，使用简单的轮次限制（回退方案）
        recent_context = recent_context[-8:] if len(recent_context) > 8 else recent_context
        
        dialogue_parts = ["[RECENT_DIALOGUE]"]
        for turn in recent_context:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            label = "User" if role == 'user' else "Assistant" 
            dialogue_parts.append(f"{label}: {content}")
        
        system_parts.append("\n".join(dialogue_parts))
    
    # 4. 预测指令（中文提示）
    if English_flag:
        system_parts.append("\n预测用户的下一条消息：")
        system_parts.append("注意：请直接给出用户的下一条消息，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
    elif Japanese_flag:
        system_parts.append("\n预测用户的下一条消息：")
        system_parts.append("注意：请直接给出用户的下一条消息，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
    else:
        if task_text == "基于用户在 Lovink 问卷中的回答数据，模拟该用户的回答风格和行为模式":
            system_parts.append("\n预测用户针对该问题的回复：")
            system_parts.append("注意：请直接给出用户的回答，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
        elif task_text and "MovieLens" in task_text:
            system_parts.append("\n预测用户对该电影的评分：")
            system_parts.append("注意：请直接给出用户的评分，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
        elif task_text and "Reddit" in task_text:
            system_parts.append("\n预测用户对该评论的回复：")
            system_parts.append("注意：请直接给出用户的回复，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
        elif task_text and "REALTALK" in task_text:
            system_parts.append("\n预测用户的下一条消息：")
            system_parts.append("注意：请直接给出用户的下一条消息，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
        else:
            system_parts.append("\n预测用户的下一条消息：")
            system_parts.append("注意：请直接给出用户的下一条消息，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
    
    # 组合成 system message
    system_content = "\n\n".join(system_parts)
    messages.append({"role": "system", "content": system_content})
    
    # target_answer 用 [ANSWER] 和 [/ANSWER] 包裹 next_question（与训练时保持一致）
    target_answer = f"[ANSWER]\n{next_question}\n[/ANSWER]"
    
    return messages, target_answer


# ============================================================================
# 用户采样函数
# ============================================================================

def sample_per_user(
    all_samples: List[Dict[str, Any]],
    max_samples_per_user: int = 2,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    对每个用户的样本进行随机采样
    
    Args:
        all_samples: 所有训练样本
        max_samples_per_user: 每个用户最多保留多少个样本
        random_seed: 随机种子（保证可复现）
    
    Returns:
        采样后的样本列表
    """
    random.seed(random_seed)
    
    # 按用户分组
    user_samples = {}
    for sample in all_samples:
        user_hash = sample.get('user_hash', 'unknown')
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    # 对每个用户的样本进行采样
    sampled_samples = []
    for user_hash, samples in user_samples.items():
        if len(samples) <= max_samples_per_user:
            # 样本数不超过限制，全部保留
            sampled_samples.extend(samples)
        else:
            # 随机采样
            sampled = random.sample(samples, max_samples_per_user)
            sampled_samples.extend(sampled)
    
    # 打印统计信息
    print(f"\n{'='*50}")
    print(f"样本采样统计:")
    print(f"  原始样本数: {len(all_samples)}")
    print(f"  用户数: {len(user_samples)}")
    print(f"  每用户最大样本数: {max_samples_per_user}")
    print(f"  采样后样本数: {len(sampled_samples)}")
    print(f"  采样比例: {len(sampled_samples) / len(all_samples) * 100:.1f}%")
    print(f"  预期训练时间缩短: {len(all_samples) / len(sampled_samples):.1f}x")
    print(f"{'='*50}\n")
    
    return sampled_samples


# ============================================================================
# 训练/验证集划分和历史添加
# ============================================================================

def split_train_val(samples, val_ratio=0.15, seed=42):
    """
    划分训练集和验证集（用户内划分，保证每个用户在训练和验证集都有样本）
    
    策略：对每个用户的样本进行随机打乱后按比例划分
    - 适用场景：测试集中的用户也出现在训练集中
    - 目标：学习基于用户已有对话预测新对话（用户内泛化）
    
    Args:
        samples: 所有训练样本
        val_ratio: 验证集比例（默认0.15，即15%）
        seed: 随机种子
    
    Returns:
        (train_samples, val_samples)
    """
    random.seed(seed)
    
    # 按用户分组
    user_samples = {}
    for sample in samples:
        user_hash = sample['user_hash']
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    train_samples = []
    val_samples = []
    
    # 对每个用户的样本进行划分
    for user_hash, user_data in user_samples.items():
        # 随机打乱该用户的样本
        random.shuffle(user_data)
        
        # 计算划分点：(1 - val_ratio) 的样本用于训练
        split_idx = int(len(user_data) * (1 - val_ratio))
        
        # 确保至少有1个样本在训练集（如果该用户只有1个样本，全部给训练集）
        if split_idx == 0 and len(user_data) > 0:
            split_idx = 1
        
        # 划分
        train_samples.extend(user_data[:split_idx])
        val_samples.extend(user_data[split_idx:])
    
    return train_samples, val_samples


def add_history_to_samples(train_samples, all_samples):
    """为每个样本添加历史信息（只包含用户的问题，不包含assistant内容）"""
    samples_with_history = []
    for sample in train_samples:
        user_hash = sample['user_hash']
        history = get_user_only_history(
            all_samples, 
            user_hash,
            current_sample=sample,
            current_context=sample.get('context'),
            max_history=15,
            use_cache=True
        )
        sample['history'] = history
        samples_with_history.append(sample)
    return samples_with_history


class DynamicPaddingDataset(Dataset):
    """
    优化版数据集：不做padding，返回原始长度的tensor
    padding将在collate_fn中动态进行
    """
    def __init__(self, samples, tokenizer, max_length=32768, use_profile=True, use_history=True, use_context=True, verbose=False, use_detailed_template=True, max_context_turns=15, template_filename=None):
        # 使用绝对路径导入，确保使用当前目录的模块
        import sys
        from pathlib import Path
        current_dir = str(Path(__file__).parent.absolute())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # ✅ 根据 use_detailed_template 选择 prompt 构建函数
        if use_detailed_template:
            # 使用详细模板（标准 markdown 格式，使用 {VAR_NAME} 占位符）
            from prompt_builder_LovinkDialogue import build_training_prompt
            print("使用详细 Prompt 模板 (prompt_builder_LovinkDialogue)")
            self.build_training_prompt = build_training_prompt
        else:
            # 使用简短模板
            # 优先尝试从 data_loader.py 导入（新版本，只预测 continuation）
            # 如果失败，则从 data_loader_more_data.py 导入（旧版本，数据扩充）
            try:
                from data_loader import build_simple_training_prompt as build_training_prompt
                print("使用简短 Prompt 模板 (data_loader.build_simple_training_prompt - 只预测continuation)")
                self.build_training_prompt = build_training_prompt
            except ImportError:
                from data_loader_more_data import build_simple_training_prompt as build_training_prompt
                print("使用简短 Prompt 模板 (data_loader_more_data.build_simple_training_prompt)")
                self.build_training_prompt = build_training_prompt
        
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        self.use_detailed_template = use_detailed_template  # 是否使用详细模板
        self.max_context_turns = max_context_turns  # 新增：最大保留的 context 轮次数
        self.template_filename = template_filename  # 新增：模板文件名
        self.verbose = verbose  # 是否输出详细日志
        
        # 截断统计
        self.truncation_stats = {
            'total_samples': 0,
            'truncated_samples': 0,
            'truncated_turns': 0,
            # 历史记录统计
            'total_history_items': 0,
            'truncated_history_items': 0,
            'samples_with_history': 0,
            'samples_with_history_truncated': 0
        }
        
        # 用于记录第一次截断的样本信息（调试用）
        self.first_truncation_logged = False

    def __len__(self):
        return len(self.samples)
    
    def get_truncation_stats(self):
        """获取截断统计信息"""
        if self.truncation_stats['total_samples'] == 0:
            return {
                'truncation_rate': 0.0,
                'avg_truncated_turns': 0.0,
                'total_samples': 0,
                'truncated_samples': 0,
                # 历史记录统计
                'history_truncation_rate': 0.0,
                'total_history_items': 0,
                'truncated_history_items': 0,
                'samples_with_history': 0,
                'samples_with_history_truncated': 0
            }
        
        truncation_rate = self.truncation_stats['truncated_samples'] / self.truncation_stats['total_samples']
        avg_truncated_turns = (self.truncation_stats['truncated_turns'] / self.truncation_stats['truncated_samples'] 
                               if self.truncation_stats['truncated_samples'] > 0 else 0)
        
        # 计算历史记录截断率
        history_truncation_rate = 0.0
        if self.truncation_stats['total_history_items'] > 0:
            history_truncation_rate = self.truncation_stats['truncated_history_items'] / self.truncation_stats['total_history_items']
        
        return {
            'truncation_rate': truncation_rate,
            'avg_truncated_turns': avg_truncated_turns,
            'total_samples': self.truncation_stats['total_samples'],
            'truncated_samples': self.truncation_stats['truncated_samples'],
            # 历史记录统计
            'history_truncation_rate': history_truncation_rate,
            'total_history_items': self.truncation_stats['total_history_items'],
            'truncated_history_items': self.truncation_stats['truncated_history_items'],
            'samples_with_history': self.truncation_stats['samples_with_history'],
            'samples_with_history_truncated': self.truncation_stats['samples_with_history_truncated']
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 统计历史记录信息
        original_history = sample.get('history', []) if self.use_history else []
        has_history = len(original_history) > 0
        original_history_count = len(original_history)
        
        if has_history:
            self.truncation_stats['samples_with_history'] += 1
            self.truncation_stats['total_history_items'] += original_history_count
        
        # 1. 构建消息
        # ✅ 根据模板类型，传递不同的参数
        if self.use_detailed_template:
            # 详细模板需要额外的参数
            messages, target_answer = self.build_training_prompt(
                context=sample['context'],
                next_question=sample['next_question'],
                user_profile=sample.get('user_profile') if self.use_profile else None,
                task_description=sample.get('task_description'),
                history=original_history,
                use_profile=self.use_profile,
                use_history=self.use_history,
                use_context=self.use_context,
                use_detailed_template=self.use_detailed_template,
                max_context_turns=self.max_context_turns,
                tokenizer=self.tokenizer,
                template_filename=self.template_filename  # ✅ 传递模板文件名
            )
        else:
            # 简短模板 - ✅ 添加 tokenizer 和 max_length 用于动态长度调整
            messages, target_answer = self.build_training_prompt(
                context=sample['context'],
                next_question=sample['next_question'],
                user_profile=sample.get('user_profile') if self.use_profile else None,
                task_description=sample.get('task_description'),
                history=original_history,
                use_profile=self.use_profile,
                use_history=self.use_history,
                use_context=self.use_context,
                tokenizer=self.tokenizer,         # ✅ 传递 tokenizer
                max_length=self.max_length,       # ✅ 传递 max_length
                min_target_tokens=64,             # ✅ 预留 64 tokens 给 target
                user_hash=sample.get('user_hash')  # ✅ 传递 user_hash（始终包含）
            )


        # 检查历史记录是否被截断（在 prompt_builder 中限制为前5个）
        if has_history and original_history_count > 5:
            truncated_history_count = original_history_count - 5
            self.truncation_stats['truncated_history_items'] += truncated_history_count
            self.truncation_stats['samples_with_history_truncated'] += 1


        # 2. 生成完整文本
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # 修正：应该生成assistant角色的回复（目标用户）
        generation_suffix = "<|im_start|>assistant\n"
        full_prompt = full_prompt.strip() + generation_suffix
        im_end_token = "<|im_end|>"
        full_text = full_prompt + target_answer + im_end_token
        
        # ✅ 第二层保护：如果仍然超长，逐步从前往后删除对话轮次
        target_with_end = target_answer + im_end_token
        target_tokens = len(self.tokenizer.encode(target_with_end, add_special_tokens=False))
        min_buffer = 64
        
        full_length = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        is_truncated = False
        removed_turns = 0
        
        if full_length > self.max_length:
            is_truncated = True
            
            # 允许的最大 prompt 长度
            max_prompt_tokens = self.max_length - target_tokens - min_buffer
            
            # 如果有 RECENT_DIALOGUE 部分，逐步从前往后删除旧对话
            if len(messages) > 0 and messages[0].get('role') == 'system':
                system_content = messages[0]['content']
                
                if '[RECENT_DIALOGUE]' in system_content:
                    # 解析 dialogue 部分
                    parts = system_content.split('[RECENT_DIALOGUE]')
                    if len(parts) > 1:
                        prefix = parts[0].strip()  # Profile + Task
                        dialogue_section = parts[1].strip()
                        
                        # 提取对话行（跳过 "Predict the user's next message:"）
                        dialogue_lines = []
                        for line in dialogue_section.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('Predict') and not line.startswith('（前面省略'):
                                if line.startswith('User:') or line.startswith('Assistant:'):
                                    dialogue_lines.append(line)
                        
                        # 从前往后逐步删除对话轮次，直到长度合适
                        while dialogue_lines and full_length > self.max_length:
                            # 删除最旧的一轮（第一个）
                            dialogue_lines.pop(0)
                            removed_turns += 1
                            
                            # 重建 system message
                            if removed_turns > 0 and dialogue_lines:
                                new_dialogue = f"\n[RECENT_DIALOGUE]\n（前面省略了 {removed_turns} 轮对话）\n" + "\n".join(dialogue_lines)
                            elif dialogue_lines:
                                new_dialogue = "\n[RECENT_DIALOGUE]\n" + "\n".join(dialogue_lines)
                            else:
                                new_dialogue = ""
                            
                            new_system = prefix + new_dialogue + "\n\nPredict the user's next message:"
                            messages[0]['content'] = new_system
                            
                            # 重新生成并测试长度
                            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                            full_prompt = full_prompt.strip() + generation_suffix
                            full_text = full_prompt + target_answer + im_end_token
                            full_length = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        
        # 更新截断统计
        self.truncation_stats['total_samples'] += 1
        if is_truncated:
            self.truncation_stats['truncated_samples'] += 1
            self.truncation_stats['truncated_turns'] += removed_turns
            
            # 第一次遇到截断时输出日志
            if not self.first_truncation_logged and self.verbose:
                self.first_truncation_logged = True
                print(f"\n⚠️  第二层保护：逐步删除旧对话 (样本#{idx}):")
                print(f"  删除了 {removed_turns} 轮对话（从最旧的开始）")
                print(f"  调整后长度: {full_length} tokens")
                print(f"  最大长度: {self.max_length} tokens")
                print(f"  Target 长度: {target_tokens} tokens (已完整保留)")
                print(f"  (后续截断将不再输出详细信息)\n")

        # 3. 编码 - 关键：不做padding！
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 关键改动：不padding
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # 4. 计算labels
        target_ids = self.tokenizer.encode(target_answer, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        actual_prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        safe_prompt_len = min(actual_prompt_len, len(input_ids) - 1)
        labels[:safe_prompt_len] = -100
        
        # 屏蔽padding token（虽然现在没有padding，但为了兼容性保留）
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'actual_length': len(input_ids)  # 记录实际长度，用于调试
        }


def dynamic_padding_collate_fn(examples, tokenizer):
    """
    动态Padding的collate函数
    关键优化：只padding到batch内最长样本的长度，而不是固定的max_length
    """
    # 找到batch中最长的序列长度
    max_length_in_batch = max(ex['input_ids'].shape[0] for ex in examples)
    
    # 打印batch信息（用于调试）
    lengths = [ex['input_ids'].shape[0] for ex in examples]
    if random.random() < 0.05:  # 5%的概率打印，避免刷屏
        print(f"[Batch Info] Lengths: {lengths}, Max: {max_length_in_batch}, Avg: {sum(lengths)/len(lengths):.0f}")
    
    batch = {}
    
    # 动态padding每个字段
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ex in examples:
        seq_len = ex['input_ids'].shape[0]
        pad_len = max_length_in_batch - seq_len
        
        # Padding input_ids
        padded_input_ids.append(
            torch.cat([
                ex['input_ids'],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
        )
        
        # Padding attention_mask
        padded_attention_mask.append(
            torch.cat([
                ex['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        
        # Padding labels
        padded_labels.append(
            torch.cat([
                ex['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )
    
    batch['input_ids'] = torch.stack(padded_input_ids)
    batch['attention_mask'] = torch.stack(padded_attention_mask)
    batch['labels'] = torch.stack(padded_labels)
    
    # 添加其他元信息（如果有）
    if 'actual_length' in examples[0]:
        batch['actual_length'] = [ex['actual_length'] for ex in examples]
    
    return batch

def select_relevant_history(history: List[str], current_context: List[Dict[str, str]], max_history: int) -> List[str]:
    """
    智能选择最相关的历史记录（简单版本：返回最近的）
    
    Args:
        history: 历史记录列表
        current_context: 当前对话上下文
        max_history: 最大历史数量
    
    Returns:
        筛选后的历史记录
    """
    # 简单实现：返回最近的历史记录
    return history[-max_history:] if len(history) > max_history else history


def check_flash_attention_support():
    """检查系统是否支持 FlashAttention 2"""
    try:
        import flash_attn
        flash_version = getattr(flash_attn, '__version__', 'unknown')
        print(f"FlashAttention 已安装，版本: {flash_version}")
        return True
    except ImportError:
        print("警告: FlashAttention 未安装")
        return False


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print('警告: 未检测到分布式训练环境变量，使用单卡训练')
        rank = 0
        world_size = 1
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='分布式消融实验训练（FlashAttention 2 + 动态Padding）- LovinkDialogue')
    parser.add_argument('--config', type=str,
                       default='config_LovinkDialogue.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='最大训练轮次（默认：50）')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值（默认：3）')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='早停阈值（默认：0.001）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='本地进程rank（由 torch.distributed.launch 自动设置）')
    parser.add_argument('--wandb_project', type=str, default='Qwen3-LovinkDialogue',
                       help='Weights & Biases项目名称（默认：Qwen3-LovinkDialogue）')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases运行名称（默认：自动生成）')
    parser.add_argument('--disable_flash_attn', action='store_true',
                       help='禁用FlashAttention 2，使用标准attention')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed配置文件路径（可选）')
    
    # 新增：Prompt 模板控制参数
    parser.add_argument('--prompt_style', type=str, default='simple',
                       choices=['simple', 'detailed', 'lovink'],
                       help='Prompt 风格：simple=简洁标签格式（默认），detailed=详细模板，lovink=Lovink风格')
    parser.add_argument('--template_filename', type=str, default=None,
                       help='指定模板文件名（仅当 prompt_style=detailed 时生效）')
    
    # 新增：每用户采样参数
    parser.add_argument('--max_samples_per_user', type=int, default=None,
                       help='每个用户最多保留多少个样本（用于减少训练数据量）')
    parser.add_argument('--sample_seed', type=int, default=42,
                       help='采样随机种子（默认：42，保证可复现）')
    
    args = parser.parse_args()
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 只在主进程打印信息
    is_main_process = (rank == 0)
    
    # 配置 Weights & Biases (只在主进程)
    if args.wandb_project:
        try:
            import wandb
            os.environ['WANDB_PROJECT'] = args.wandb_project
            if args.wandb_run_name:
                os.environ['WANDB_NAME'] = args.wandb_run_name
            if is_main_process:
                print(f"✓ 已启用 Weights & Biases 监控")
        except ImportError:
            if is_main_process:
                print("警告: wandb 未安装")
            args.wandb_project = None
    
    if is_main_process:
        print(f"=" * 80)
        print(f"分布式训练设置（FlashAttention 2 + 动态Padding）:")
        print(f"  World Size (总进程数): {world_size}")
        print(f"  Rank (进程ID): {rank}")
        print(f"  Local Rank (本地GPU ID): {local_rank}")
        print(f"  使用 {world_size} 张GPU进行并行训练")
        print(f"  优化策略: FlashAttention 2 + 动态Batch Padding")
        if args.deepspeed:
            print(f"  DeepSpeed配置: {args.deepspeed}")
        print(f"=" * 80)
    
    # 检查 FlashAttention 支持
    use_flash_attn = False
    if not args.disable_flash_attn and is_main_process:
        use_flash_attn = check_flash_attention_support()
    
    # 广播 use_flash_attn 到所有进程
    if world_size > 1:
        use_flash_attn_tensor = torch.tensor([use_flash_attn], dtype=torch.bool, device=f'cuda:{local_rank}')
        dist.broadcast(use_flash_attn_tensor, src=0)
        use_flash_attn = use_flash_attn_tensor.item()
    
    # 验证GPU是否可用
    if torch.cuda.is_available():
        if is_main_process:
            print(f"CUDA 可用，总GPU数量: {torch.cuda.device_count()}")
            print(f"当前进程使用 GPU: {local_rank}")
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
            print(f"GPU 名称: {gpu_name}")
            print(f"GPU 总内存: {gpu_memory:.2f} GB")
            
            # 检查GPU计算能力
            compute_capability = torch.cuda.get_device_capability(local_rank)
            print(f"GPU 计算能力: {compute_capability[0]}.{compute_capability[1]}")
            if compute_capability[0] >= 8:  # A100/H100
                print("✓ GPU支持FlashAttention 2优化")
            else:
                print("GPU计算能力较低，FlashAttention 2性能可能受限")
    else:
        print("错误: CUDA 不可用")
        cleanup_distributed()
        return
    
    # 加载配置（优先使用当前目录，支持绝对路径）
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # 优先查找当前目录
        local_config = Path(__file__).parent / args.config
        if local_config.exists():
            config_path = str(local_config)
        else:
            # 回退到父目录（向后兼容）
            config_path = os.path.join(Path(__file__).parent.parent, args.config)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取消融配置
    ablation_config = config['ablation_configs'][args.ablation_config]
    use_profile = ablation_config.get('use_profile', True)
    use_history = ablation_config.get('use_history', True)
    use_context = ablation_config.get('use_context', True)
    config_name = ablation_config['name']
    
    if is_main_process:
        print("=" * 80)
        print(f"消融实验（FlashAttn2 + 动态Padding）: {config_name}")
        print(f"使用配置: profile={use_profile}, history={use_history}, context={use_context}")
        print(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}")
        print("=" * 80)
    
    # 加载训练数据
    if is_main_process:
        print("加载训练数据...")
    train_path = config['data']['train_path']
    train_data = load_train_data(train_path)
    
    if not train_data:
        print(f"错误: 无法加载训练数据")
        cleanup_distributed()
        return
    
    # 提取训练样本
    all_samples = extract_training_samples(train_data, debug=is_main_process)
    if is_main_process:
        print(f"提取了 {len(all_samples)} 个训练样本")
        print(f"  ✅ 使用 data_loader.py：不进行数据扩充，每个 data_item 生成 1 个样本")
    
    # 新增：每用户采样（如果指定了 max_samples_per_user）
    if args.max_samples_per_user is not None:
        if is_main_process:
            print(f"\n对每个用户进行采样（每用户最多 {args.max_samples_per_user} 个样本）...")
        all_samples = sample_per_user(
            all_samples,
            max_samples_per_user=args.max_samples_per_user,
            random_seed=args.sample_seed
        )
    
    # 添加历史信息
    if use_history:
        if is_main_process:
            print("添加历史信息...")
        all_samples = add_history_to_samples(all_samples, all_samples)
    
    # 划分训练集和验证集
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
    if is_main_process:
        print(f"训练集: {len(train_samples)} 个样本")
        print(f"验证集: {len(val_samples)} 个样本")
        print(f"每个GPU实际处理约 {len(train_samples) // world_size} 个训练样本")
    
    # 获取模型配置
    model_config = config['model']
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_dir = model_config['checkpoint_dir']
        dataset_name = os.path.basename(os.path.dirname(train_path))
        flash_suffix = "flashattn2" if use_flash_attn else "standard"
        output_dir = os.path.join(checkpoint_dir, f"{dataset_name}_ablation_{config_name}_{flash_suffix}_dynamic_distributed")
    
    # 只在主进程创建目录和日志文件
    training_log_path = None
    if is_main_process:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
            
            # 创建训练日志文件
            training_log_path = os.path.join(output_dir, "training_samples_log.txt")
            print(f"训练日志: {training_log_path}")
        except (OSError, IOError) as e:
            print(f"警告: 无法创建输出目录: {e}")
    
    # 等待主进程创建完目录
    if world_size > 1:
        dist.barrier()
    
    # 加载模型和tokenizer
    model_path = model_config['path']
    if is_main_process:
        print(f"加载模型: {model_path}")
        if use_flash_attn:
            print("  使用 FlashAttention 2 实现...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ✅ 先获取 train_config（在数据分析之前需要）
    train_config = config.get('training', {})
    
    # ============================================================================
    # 计算训练集的最大输入长度（在主进程中）
    # ============================================================================
    if is_main_process:
        print("\n" + "="*80)
        print("📊 分析训练数据长度分布")
        print("="*80)
        
        # 导入prompt构建函数
        if args.prompt_style == 'simple':
            from data_loader import build_simple_training_prompt
        else:
            from prompt_builder_LovinkDialogue import build_training_prompt as build_simple_training_prompt
        
        # 采样部分数据进行分析（避免太慢）
        sample_size = min(100, len(train_samples))
        sampled_indices = random.sample(range(len(train_samples)), sample_size)
        
        lengths = []
        max_length_sample = None
        max_length = 0
        
        print(f"正在分析 {sample_size} 个样本...")
        for idx in sampled_indices:
            sample = train_samples[idx]
            
            # 构建prompt
            try:
                messages, target_answer = build_simple_training_prompt(
                    context=sample['context'],
                    next_question=sample['next_question'],
                    user_profile=sample.get('user_profile') if use_profile else None,
                    task_description=sample.get('task_description'),
                    history=sample.get('history', []) if use_history else [],
                    use_profile=use_profile,
                    use_history=use_history,
                    use_context=use_context,
                    tokenizer=tokenizer,
                    max_length=train_config.get('max_length', 4096),
                    min_target_tokens=64,
                    user_hash=sample.get('user_hash')
                )
                
                # 生成完整文本
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                generation_suffix = "<|im_start|>assistant\n"
                full_prompt = full_prompt.strip() + generation_suffix
                im_end_token = "<|im_end|>"
                full_text = full_prompt + target_answer + im_end_token
                
                # 计算长度
                token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                length = len(token_ids)
                lengths.append(length)
                
                # 记录最长的样本
                if length > max_length:
                    max_length = length
                    max_length_sample = {
                        'idx': idx,
                        'length': length,
                        'user_hash': sample.get('user_hash', 'unknown'),
                        'context_turns': len(sample.get('context', [])),
                        'history_items': len(sample.get('history', []))
                    }
            except Exception as e:
                print(f"  警告: 样本 {idx} 处理失败: {e}")
                continue
        
        if lengths:
            import numpy as np
            lengths_array = np.array(lengths)
            
            print(f"\n训练数据长度统计（基于 {len(lengths)} 个样本）:")
            print(f"  最小长度: {lengths_array.min()}")
            print(f"  最大长度: {lengths_array.max()}")
            print(f"  平均长度: {lengths_array.mean():.1f}")
            print(f"  中位数长度: {np.median(lengths_array):.1f}")
            print(f"  标准差: {lengths_array.std():.1f}")
            print(f"\n长度分布:")
            print(f"  < 1024 tokens:  {(lengths_array < 1024).sum()} ({(lengths_array < 1024).sum()/len(lengths)*100:.1f}%)")
            print(f"  < 2048 tokens:  {(lengths_array < 2048).sum()} ({(lengths_array < 2048).sum()/len(lengths)*100:.1f}%)")
            print(f"  < 4096 tokens:  {(lengths_array < 4096).sum()} ({(lengths_array < 4096).sum()/len(lengths)*100:.1f}%)")
            print(f"  < 8192 tokens:  {(lengths_array < 8192).sum()} ({(lengths_array < 8192).sum()/len(lengths)*100:.1f}%)")
            print(f"  >= 8192 tokens: {(lengths_array >= 8192).sum()} ({(lengths_array >= 8192).sum()/len(lengths)*100:.1f}%)")
            
            if max_length_sample:
                print(f"\n最长样本信息:")
                print(f"  索引: {max_length_sample['idx']}")
                print(f"  长度: {max_length_sample['length']} tokens")
                print(f"  用户哈希: {max_length_sample['user_hash']}")
                print(f"  上下文轮次: {max_length_sample['context_turns']}")
                print(f"  历史条目数: {max_length_sample['history_items']}")
            
            # 根据数据分布给出配置建议
            configured_max_length = train_config.get('max_length', 4096)
            percentile_95 = np.percentile(lengths_array, 95)
            print(f"\n配置建议:")
            print(f"  当前配置的 max_length: {configured_max_length}")
            print(f"  95分位数长度: {percentile_95:.0f}")
            if percentile_95 > configured_max_length:
                print(f"  ⚠️  警告: 95%的数据超过配置的max_length，可能导致大量截断")
                print(f"  建议调整 max_length 至少到 {int(percentile_95)}")
            elif percentile_95 < configured_max_length * 0.7:
                print(f"  ℹ️  提示: 95%的数据长度远小于max_length，可以考虑降低以节省显存")
            else:
                print(f"  ✓ max_length 设置合理")
        
        print("="*80 + "\n")
    
    # 等待主进程完成分析
    if world_size > 1:
        dist.barrier()
    
    # 加载模型到指定GPU（使用FlashAttention 2）
    model_kwargs = {
        'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        'trust_remote_code': True,
    }
    
    # 如果支持且未禁用，则使用FlashAttention 2
    if use_flash_attn:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if is_main_process:
            if use_flash_attn:
                print("✓ 模型已加载（FlashAttention 2）")
            else:
                print("✓ 模型已加载（标准Attention）")
    except Exception as e:
        if is_main_process:
            print(f"加载FlashAttention 2失败: {e}")
            print("   回退到标准attention...")
        # 回退到标准attention
        model_kwargs.pop('attn_implementation', None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        use_flash_attn = False
    
    # 启用梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if is_main_process:
            print("✓ 梯度检查点已启用")
    
    # 将模型移到对应的GPU
    model = model.to(local_rank)
    
    # 创建数据集（使用动态Padding版本）
    # train_config 已在前面定义（数据分析阶段）
    if is_main_process:
        print("创建训练数据集（动态Padding模式）...")
    
    # ✅ 根据命令行参数决定使用哪种 prompt 风格
    use_detailed_template = (args.prompt_style != 'simple')
    template_filename = args.template_filename if args.prompt_style == 'detailed' else None
    
    if is_main_process:
        print(f"Prompt 风格: {args.prompt_style}")
        if args.prompt_style == 'simple':
            print("   使用简洁标签格式（[USER_PROFILE] [DIM_XXX=score] ...）")
        elif args.prompt_style == 'detailed':
            if template_filename:
                print(f"   使用详细模板: {template_filename} (标准 {{VAR_NAME}} 格式)")
            else:
                print("   使用详细模板（默认顺序查找）")
        elif args.prompt_style == 'lovink':
            print("   使用 Lovink 风格模板")
    
    train_dataset = DynamicPaddingDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=train_config.get('max_length', 4096),
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context,
        verbose=is_main_process,  # 只在主进程输出详细日志
        use_detailed_template=use_detailed_template,
        template_filename=template_filename
    )
    
    val_dataset = None
    if val_samples:
        if is_main_process:
            print("创建验证数据集（动态Padding模式）...")
        val_dataset = DynamicPaddingDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            max_length=train_config.get('max_length', 4096),
            use_profile=use_profile,
            use_history=use_history,
            use_context=use_context,
            use_detailed_template=use_detailed_template,
            template_filename=template_filename
        )
    
    # 数据整理器（使用动态Padding版本）
    def collate_fn(examples):
        return dynamic_padding_collate_fn(examples, tokenizer)
    
    # 在主进程中打印几个样本示例（用于调试和验证）
    if is_main_process and training_log_path:
        
        # 同时写入日志文件
        with open(training_log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write(f"训练配置: {config_name}\n")
            log_file.write(f"数据集: {train_path}\n")
            log_file.write(f"总样本数: {len(train_samples)}\n")
            log_file.write(f"Max Length: {train_config.get('max_length', 4096)}\n")
            log_file.write(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}\n")
            log_file.write("=" * 80 + "\n\n")
            
            num_samples_to_show = min(5, len(train_samples))
            for i in range(num_samples_to_show):
                sample = train_samples[i]
                
                # 控制台输出
                print(f"\n--- 样本 {i+1} ---")
                
                # 日志文件输出
                log_file.write(f"\n{'=' * 80}\n")
                log_file.write(f"样本 {i+1}\n")
                log_file.write(f"{'=' * 80}\n\n")
                
                # 显示角色映射的context
                context_info = f"Context ({len(sample['context'])}轮):"
                print(context_info)
                log_file.write(context_info + "\n")
                
                for j, turn in enumerate(sample['context']):
                    role = turn['role']
                    content = turn['content']
                    role_desc = "user(对话者)" if role == "user" else "assistant(目标用户)"
                    
                    # 控制台只显示前5轮，且截断
                    if j < 5:
                        print(f"  {j+1}. {role_desc:25s}: {content[:60]}...")
                    
                    # 日志文件显示完整内容
                    log_file.write(f"  {j+1}. {role_desc}:\n")
                    log_file.write(f"     {content}\n\n")
                
                if len(sample['context']) > 5:
                    print(f"  ... (还有 {len(sample['context']) - 5} 轮)")
                
                # 显示要预测的target
                target = sample['next_question']
                print(f"\nTarget (模型要生成的):")
                print(f"  assistant(目标用户): {target[:100]}...")
                
                log_file.write(f"\nTarget (模型要生成的):\n")
                log_file.write(f"  assistant(目标用户):\n")
                log_file.write(f"     {target}\n\n")
                
                # 显示profile信息
                if sample.get('user_profile'):
                    profile = sample['user_profile']
                    print(f"\nProfile:")
                    log_file.write(f"Profile:\n")
                    
                    for key in ['name', 'age', 'gender', 'profession', 'residence']:
                        if key in profile:
                            info = f"  {key.capitalize()}: {profile[key]}"
                            print(info)
                            log_file.write(info + "\n")
                
                # 使用dataset的__getitem__来获取编码后的信息
                try:
                    encoded_sample = train_dataset[i]
                    input_length = len(encoded_sample['input_ids'])
                    valid_labels = (encoded_sample['labels'] != -100).sum().item()
                    actual_length = encoded_sample.get('actual_length', input_length)
                    
                    encoding_info = [
                        f"\n编码信息:",
                        f"  输入长度: {input_length} tokens",
                        f"  实际长度: {actual_length} tokens",
                        f"  有效标签数: {valid_labels} tokens",
                        f"  训练比例: {valid_labels/input_length:.2%}"
                    ]
                    
                    for line in encoding_info:
                        print(line)
                        log_file.write(line + "\n")
                    
                    # 检查是否被截断
                    if hasattr(train_dataset, 'truncation_stats'):
                        stats = train_dataset.get_truncation_stats()
                        if stats['truncated_samples'] > 0:
                            truncation_info = f"  ⚠️  已有 {stats['truncated_samples']} 个样本被截断"
                            print(truncation_info)
                            log_file.write(truncation_info + "\n")
                    
                except Exception as e:
                    error_msg = f"\n编码信息: 无法获取 ({e})"
                    print(error_msg)
                    log_file.write(error_msg + "\n")
                
                log_file.write("\n")
        
        print(f"\n✓ 样本详情已保存到: {training_log_path}")
        print("=" * 80)
    
    # 计算训练步数
    steps_per_epoch = len(train_dataset) // (world_size * train_config.get('batch_size', 2) * train_config.get('gradient_accumulation_steps', 8))
    eval_steps_value = max(1, steps_per_epoch // 2) if val_dataset else None
    save_steps_value = train_config.get('save_steps', 500)
    
    if val_dataset and eval_steps_value and save_steps_value % eval_steps_value != 0:
        save_steps_value = ((save_steps_value + eval_steps_value - 1) // eval_steps_value) * eval_steps_value
        if is_main_process:
            print(f"调整 save_steps 为 {save_steps_value}（eval_steps={eval_steps_value} 的整数倍）")
    
    # 训练参数（分布式 + FlashAttention 2 + 动态Padding）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=train_config.get('batch_size', 2),
        per_device_eval_batch_size=train_config.get('eval_batch_size', 2),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 8),
        learning_rate=train_config.get('learning_rate', 1e-5),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 100),
        logging_steps=train_config.get('logging_steps', 10),
        save_steps=save_steps_value,
        eval_steps=eval_steps_value,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,  # FlashAttention 2 与 BF16 配合效果更好
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=0.5,
        report_to="wandb" if args.wandb_project else "none",
        # 分布式训练关键参数
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        dataloader_num_workers=2,
        save_on_each_node=False,
        logging_first_step=True,
        # DeepSpeed配置（可选）
        deepspeed=args.deepspeed,
    )
    
    # 创建早停回调
    callbacks = []
    if val_dataset:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping)
    
    # 创建自定义Trainer（带数值稳定性检查和详细日志）
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 保存 tokenizer 引用（用于损失权重计算）
            self.tokenizer = tokenizer
            # 创建训练进度日志文件
            if is_main_process:
                self.progress_log_file = os.path.join(output_dir, "training_logs", "training_progress.txt")
                os.makedirs(os.path.dirname(self.progress_log_file), exist_ok=True)
                with open(self.progress_log_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 100 + "\n")
                    f.write("训练进度日志\n")
                    f.write("=" * 100 + "\n\n")
            else:
                self.progress_log_file = None
        
        def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs) -> None:
            """
            重写log方法，修正梯度累积导致的train_loss显示问题，并添加详细日志
            """
            if "loss" in logs:
                # 修正train_loss：除以梯度累积步数
                logs["loss"] = logs["loss"] / self.args.gradient_accumulation_steps
            
            # 记录详细日志（前50步和每100步）
            if is_main_process and self.progress_log_file:
                step = self.state.global_step
                if step <= 50 or step % 100 == 0:
                    with open(self.progress_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'=' * 80}\n")
                        f.write(f"Step {step} | Epoch {self.state.epoch:.2f}\n")
                        f.write(f"{'=' * 80}\n")
                        for key, value in logs.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {key}: {value:.6f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                        f.write("\n")
            
            # 调用父类的log方法，传递所有额外参数
            if start_time is not None:
                super().log(logs, start_time, **kwargs)
            else:
                super().log(logs, **kwargs)
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """计算损失（带数值稳定性检查和batch日志）"""
            # 记录前3个batch的详细信息
            if is_main_process and self.state.global_step <= 3 and self.progress_log_file:
                with open(self.progress_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'=' * 100}\n")
                    f.write(f"Batch 详细信息 - Step {self.state.global_step}\n")
                    f.write(f"{'=' * 100}\n")
                    f.write(f"Batch size: {inputs['input_ids'].shape[0]}\n")
                    f.write(f"Sequence lengths: {inputs['input_ids'].shape[1]}\n")
                    
                    # 显示第一个样本的信息
                    if inputs['input_ids'].shape[0] > 0:
                        first_input_ids = inputs['input_ids'][0]
                        first_labels = inputs['labels'][0]
                        first_attention_mask = inputs['attention_mask'][0]
                        
                        f.write(f"\n第一个样本:\n")
                        f.write(f"  Input length: {len(first_input_ids)}\n")
                        f.write(f"  Valid labels: {(first_labels != -100).sum().item()}\n")
                        f.write(f"  Attention tokens: {first_attention_mask.sum().item()}\n")
                        
                        # 解码更多tokens以查看实际内容
                        try:
                            seq_len = len(first_input_ids)
                            
                            f.write(f"\n  解码的输入 (前500 tokens):\n")
                            f.write(f"  {tokenizer.decode(first_input_ids[:500], skip_special_tokens=False)}\n")
                            f.write(f"  ...\n")
                            
                            # 如果够长，打印中间部分
                            if seq_len > 1000:
                                f.write(f"\n  解码的输入 (第500-1000 tokens):\n")
                                f.write(f"  {tokenizer.decode(first_input_ids[500:1000], skip_special_tokens=False)}\n")
                                f.write(f"  ...\n")
                            
                            f.write(f"\n  解码的输入 (后500 tokens):\n")
                            f.write(f"  {tokenizer.decode(first_input_ids[-500:], skip_special_tokens=False)}\n\n")
                            
                            # 解码标签（完整显示，不截断）
                            valid_label_mask = first_labels != -100
                            if valid_label_mask.any():
                                valid_labels = first_labels[valid_label_mask]
                                f.write(f"  解码的标签 (完整有效部分，共{len(valid_labels)}个tokens):\n")
                                f.write(f"  {tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")
                        except Exception as e:
                            f.write(f"  解码失败: {e}\n")
                    
                    f.write("\n")
            
            # 移除actual_length字段（如果存在）
            actual_lengths = inputs.pop('actual_length', None)
            
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            
            # 检查并清理logits中的nan/inf
            if logits is not None and logits.numel() > 0:
                # 快速采样检查
                check_size = min(1000, logits.numel() // 2)
                if logits.numel() > check_size * 2:
                    head_values = logits.view(-1)[:check_size]
                    tail_values = logits.view(-1)[-check_size:]
                    has_issue = torch.isnan(head_values).any() or torch.isnan(tail_values).any() or \
                                torch.isinf(head_values).any() or torch.isinf(tail_values).any()
                else:
                    has_issue = torch.isnan(logits).any() or torch.isinf(logits).any()
                
                if has_issue:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} 检测到nan/inf，正在清理...")
                    logits = torch.where(
                        torch.isnan(logits) | torch.isinf(logits),
                        torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                        logits
                    )
                    logits = torch.clamp(logits, min=-50.0, max=50.0)
            
            # 计算损失（对 [ANSWER] 和 [/ANSWER] token 增加权重）
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            elif labels is not None:
                valid_labels_count = (labels != -100).sum().item()
                
                if valid_labels_count == 0:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} 没有有效的labels")
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # 创建损失权重：对 [ANSWER] 和 [/ANSWER] token 增加权重
                    # 获取 tokenizer 中的 [ANSWER] 和 [/ANSWER] 的所有 token IDs
                    answer_start_token_ids = set()
                    answer_end_token_ids = set()
                    
                    try:
                        # 尝试获取 [ANSWER] 和 [/ANSWER] 的所有 token IDs
                        if hasattr(self.tokenizer, 'encode'):
                            # 编码标签（可能被编码为多个 token）
                            answer_start_tokens = self.tokenizer.encode("[ANSWER]", add_special_tokens=False)
                            answer_end_tokens = self.tokenizer.encode("[/ANSWER]", add_special_tokens=False)
                            
                            # 保存所有相关的 token IDs（不仅仅是第一个）
                            if answer_start_tokens:
                                answer_start_token_ids = set(answer_start_tokens)
                            if answer_end_tokens:
                                answer_end_token_ids = set(answer_end_tokens)
                    except:
                        pass
                    
                    # 创建权重张量（默认权重为 1.0）
                    batch_size, seq_len = shift_labels.shape
                    loss_weights = torch.ones_like(shift_labels, dtype=torch.float32)
                    
                    # 对 [ANSWER] 和 [/ANSWER] 的所有 token 增加权重（权重设为 3.0）
                    if answer_start_token_ids:
                        for token_id in answer_start_token_ids:
                            loss_weights[shift_labels == token_id] = 3.0
                    if answer_end_token_ids:
                        for token_id in answer_end_token_ids:
                            loss_weights[shift_labels == token_id] = 3.0
                    
                    # 使用加权损失
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                    per_token_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # 应用权重并计算平均损失
                    per_token_loss = per_token_loss.view(batch_size, seq_len)
                    valid_mask = (shift_labels != -100)
                    weighted_loss = (per_token_loss * loss_weights * valid_mask.float()).sum()
                    valid_count = (valid_mask.float() * loss_weights).sum()
                    
                    if valid_count > 0:
                        loss = weighted_loss / valid_count
                    else:
                        loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
            else:
                loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
            
            # 检查损失值
            if loss is not None and torch.is_tensor(loss):
                if loss.dim() > 0:
                    loss = loss.mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} loss为nan/inf")
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                elif loss.item() > 1e6:
                    if rank == 0:
                        print(f"警告: [GPU {rank}] Step {self.state.global_step} loss过大")
                    loss = torch.clamp(loss, max=100.0)
            
            # 定期清理CUDA缓存
            if self.state.global_step % 10 == 0:
                torch.cuda.empty_cache()
            
            if return_outputs:
                return loss, outputs
            return loss
    
    # 创建 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  # 使用动态padding的collate_fn
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # 创建训练日志文件（主进程）
    if is_main_process:
        log_dir = os.path.join(output_dir, "training_logs")
        os.makedirs(log_dir, exist_ok=True)
        training_log_file = os.path.join(log_dir, "detailed_training_log.txt")
        
        print(f"\n📝 创建详细训练日志: {training_log_file}")
        
        with open(training_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("详细训练日志 - 前3个训练样本\n")
            f.write("=" * 100 + "\n\n")
            
            # 记录前3个训练样本的详细信息
            num_samples_to_log = min(3, len(train_dataset))
            for idx in range(num_samples_to_log):
                raw_sample = train_samples[idx]
                encoded_sample = train_dataset[idx]
                
                f.write(f"\n{'=' * 100}\n")
                f.write(f"训练样本 #{idx + 1}\n")
                f.write(f"{'=' * 100}\n\n")
                
                # 1. 原始样本信息
                f.write("【原始样本信息】\n")
                f.write(f"User Hash: {raw_sample.get('user_hash', 'N/A')}\n")
                if raw_sample.get('user_profile'):
                    profile = raw_sample['user_profile']
                    f.write(f"User Profile: {profile.get('name', 'N/A')} (age: {profile.get('age', 'N/A')})\n")
                f.write("\n")
                
                # 2. 对话上下文
                f.write("【对话上下文 Context】\n")
                context = raw_sample.get('context', [])
                for turn_idx, turn in enumerate(context[-5:], 1):  # 只显示最后5轮
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    f.write(f"  轮次{turn_idx} [{role}]: {content}\n")
                if len(context) > 5:
                    f.write(f"  ... (还有 {len(context) - 5} 轮对话)\n")
                f.write("\n")
                
                # 3. 目标输出（模型要学习生成的内容）
                f.write("【目标输出 Next Question】\n")
                next_question = raw_sample.get('next_question', '')
                f.write(f"{next_question}\n\n")
                
                # 4. 历史信息（如果有）
                if use_history and raw_sample.get('history'):
                    f.write("【历史信息 History】\n")
                    history = raw_sample['history']
                    for hist_idx, hist_item in enumerate(history[:3], 1):  # 只显示前3条
                        f.write(f"  历史{hist_idx}: {hist_item[:100]}...\n")
                    if len(history) > 3:
                        f.write(f"  ... (还有 {len(history) - 3} 条历史)\n")
                    f.write("\n")
                
                # 5. 编码后的信息
                f.write("【编码后的数据】\n")
                input_ids = encoded_sample['input_ids']
                labels = encoded_sample['labels']
                attention_mask = encoded_sample['attention_mask']
                
                f.write(f"Input IDs 长度: {len(input_ids)}\n")
                f.write(f"Attention Mask 长度: {len(attention_mask)}\n")
                f.write(f"Labels 长度: {len(labels)}\n")
                
                valid_labels = (labels != -100).sum().item()
                f.write(f"有效标签数: {valid_labels}\n")
                f.write(f"训练比例: {valid_labels / len(labels):.2%}\n")
                
                # 解码查看实际的文本（更详细的打印）
                total_length = len(input_ids)
                
                # 如果序列不太长（< 6000 tokens），直接打印完整内容
                if total_length <= 6000:
                    f.write("\n【完整的输入文本】\n")
                    f.write("-" * 100 + "\n")
                    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
                    f.write(decoded_full + "\n")
                    f.write("-" * 100 + "\n\n")
                    f.write(f"总序列长度: {total_length} tokens (已打印完整内容)\n\n")
                else:
                    # 序列太长，分段打印
                    f.write(f"\n【序列太长 ({total_length} tokens)，分段打印】\n\n")
                    
                    # 打印前2000个tokens
                    f.write("【第1-2000 tokens】\n")
                    f.write("-" * 100 + "\n")
                    decoded_input_start = tokenizer.decode(input_ids[:2000], skip_special_tokens=False)
                    f.write(decoded_input_start + "\n")
                    f.write("-" * 100 + "\n\n")
                    
                    # 打印中间部分（第2000-4000个tokens）
                    f.write("【第2001-4000 tokens】\n")
                    f.write("-" * 100 + "\n")
                    decoded_input_middle = tokenizer.decode(input_ids[2000:4000], skip_special_tokens=False)
                    f.write(decoded_input_middle + "\n")
                    f.write("-" * 100 + "\n\n")
                    
                    # 如果还有更多，打印第4000-6000
                    if total_length > 6000:
                        f.write("【第4001-6000 tokens】\n")
                        f.write("-" * 100 + "\n")
                        decoded_input_middle2 = tokenizer.decode(input_ids[4000:6000], skip_special_tokens=False)
                        f.write(decoded_input_middle2 + "\n")
                        f.write("-" * 100 + "\n\n")
                    
                    # 打印后2000个tokens
                    f.write("【后2000 tokens】\n")
                    f.write("-" * 100 + "\n")
                    decoded_input_end = tokenizer.decode(input_ids[-2000:], skip_special_tokens=False)
                    f.write(decoded_input_end + "\n")
                    f.write("-" * 100 + "\n\n")
                    
                    f.write(f"总序列长度: {total_length} tokens\n\n")
                
                # 解码标签（只显示有效的部分）
                valid_label_indices = (labels != -100).nonzero(as_tuple=True)[0]
                if len(valid_label_indices) > 0:
                    f.write("【解码后的标签文本 (模型要学习生成的部分)】\n")
                    f.write("-" * 100 + "\n")
                    valid_labels_ids = labels[valid_label_indices]
                    decoded_labels = tokenizer.decode(valid_labels_ids, skip_special_tokens=False)
                    f.write(decoded_labels + "\n")
                    f.write("-" * 100 + "\n\n")
                
                f.write("\n")
            
            f.write("=" * 100 + "\n")
            f.write("训练样本日志记录完成\n")
            f.write("=" * 100 + "\n")
        
        print(f"✓ 训练样本日志已保存到: {training_log_file}\n")
    
    # 开始训练
    if is_main_process:
        print("=" * 80)
        print("开始分布式训练（FlashAttention 2 + 动态Padding）")
        print("=" * 80)
        print(f"总样本数: {len(train_dataset)}")
        print(f"每个GPU处理约: {len(train_dataset) // world_size} 个样本")
        effective_batch = train_config.get('batch_size', 2) * train_config.get('gradient_accumulation_steps', 8) * world_size
        print(f"有效 batch size: {effective_batch}")
        print(f"预计每个epoch步数: {steps_per_epoch}")
        print(f"Max Length: {train_config.get('max_length', 4096)} (动态padding)")
        print(f"Attention: {'FlashAttention 2' if use_flash_attn else '标准Attention'}")
        if args.wandb_project:
            print(f"W&B 监控: 项目={args.wandb_project}, 运行={args.wandb_run_name or 'auto'}")
        
        # 输出初始截断统计（训练前）
        if hasattr(train_dataset, 'get_truncation_stats'):
            stats = train_dataset.get_truncation_stats()
            print(f"\n数据预处理截断统计:")
            print(f"  已处理样本: {stats['total_samples']}")
            print(f"  被截断样本: {stats['truncated_samples']}")
            if stats['total_samples'] > 0:
                print(f"  截断率: {stats['truncation_rate']:.2%}")
                if stats['truncated_samples'] > 0:
                    print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
        
        print("=" * 80)
    
    trainer.train()
    
    # 训练完成，输出日志汇总
    if is_main_process:
        print("\n" + "=" * 80)
        print("训练日志汇总")
        print("=" * 80)
        
        log_dir = os.path.join(output_dir, "training_logs")
        if os.path.exists(log_dir):
            print(f"详细日志文件:")
            for log_file_name in os.listdir(log_dir):
                log_path = os.path.join(log_dir, log_file_name)
                file_size = os.path.getsize(log_path) / 1024  # KB
                print(f"  - {log_path} ({file_size:.1f} KB)")
        print("=" * 80 + "\n")
    
    # 保存最终模型（只在主进程保存）
    if is_main_process:
        print(f"保存最终模型到 {output_dir}")
        try:
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            print("✓ 模型保存成功")
            
            # 保存训练配置信息
            config_info = {
                'flash_attention_2': use_flash_attn,
                'dynamic_padding': True,
                'gradient_checkpointing': True,
                'ablation_config': args.ablation_config,
                'config_name': config_name
            }
            with open(os.path.join(output_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            print("✓ 训练配置已保存")
            
        except Exception as e:
            print(f"警告: 保存模型时出错: {e}")
        
        # 输出截断统计
        if hasattr(train_dataset, 'get_truncation_stats'):
            stats = train_dataset.get_truncation_stats()
            print("\n" + "="*80)
            print(" 训练数据截断统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  被截断样本数: {stats['truncated_samples']}")
            print(f"  截断率: {stats['truncation_rate']:.2%}")
            print(f"  平均截断轮次: {stats['avg_truncated_turns']:.2f}")
            print("="*80)
            
            # 将截断统计写入日志文件
            if training_log_path:
                try:
                    with open(training_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write("\n" + "="*80 + "\n")
                        log_file.write("📊 最终训练数据截断统计\n")
                        log_file.write("="*80 + "\n")
                        log_file.write(f"总样本数: {stats['total_samples']}\n")
                        log_file.write(f"被截断样本数: {stats['truncated_samples']}\n")
                        log_file.write(f"截断率: {stats['truncation_rate']:.2%}\n")
                        log_file.write(f"平均截断轮次: {stats['avg_truncated_turns']:.2f}\n")
                        log_file.write(f"FlashAttention 2: {'启用' if use_flash_attn else '禁用'}\n")
                        log_file.write("="*80 + "\n")
                    print(f"✓ 截断统计已追加到: {training_log_path}")
                except Exception as e:
                    print(f"警告: 无法写入截断统计到日志文件: {e}")
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    if is_main_process:
        print(f"\n 训练完成！模型保存在: {output_dir}")
        if use_flash_attn:
            print(" 使用了 FlashAttention 2 加速训练")
    
    # 清理分布式环境
    cleanup_distributed()


if __name__ == '__main__':
    main()
