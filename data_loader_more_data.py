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


def load_test_data(test_path: str) -> List[Dict[str, Any]]:
    """加载测试数据"""
    if not os.path.exists(test_path):
        return []
    return load_json_file(test_path)


def get_user_profile(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    获取完整的用户 profile 信息
    合并 user.profile (基础信息) 和 user.personality.structured (心理测量数据)
    
    ⚠️ 重要：保留完整的层级结构，以便 prompt_builder 可以正确提取和填充模板占位符
    
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

def get_user_name(sample: Dict[str, Any]) -> Optional[str]:
    """获取用户 name（从 profile 中）"""
    user = sample.get('user', {})
    profile = user.get('profile', {})
    if isinstance(profile, dict):
        return profile.get('name', None)
    return None

def get_user_name(sample: Dict[str, Any]) -> Optional[str]:
    """获取用户 name（从 profile 中）"""
    user = sample.get('user', {})
    profile = user.get('profile', {})
    if isinstance(profile, dict):
        return profile.get('name', None)
    return None


def get_task_description(sample: Dict[str, Any]) -> str:
    """获取任务描述"""
    task = sample.get('task', {})
    return task.get('description', '')


def extract_context_from_sample(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """从样本中提取 context"""
    task = sample.get('task', {})
    collections = task.get('task_behavior_collections', [])
    
    if collections and len(collections) > 0:
        collection = collections[0]
        data_items = collection.get('data', [])
        if data_items and len(data_items) > 0:
            return data_items[0].get('context', [])
    
    return []


def extract_continuation_from_sample(sample: Dict[str, Any]) -> str:
    """从样本中提取 continuation"""
    task = sample.get('task', {})
    collections = task.get('task_behavior_collections', [])
    
    if collections and len(collections) > 0:
        collection = collections[0]
        data_items = collection.get('data', [])
        if data_items and len(data_items) > 0:
            return data_items[0].get('continuation', '')
    
    return ''


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
                full_dialogue = []
                for turn in context:
                    source = str(turn.get('source', '')).strip()
                    # 判断是否是目标用户（我们要预测的人）说的话：
                    # - source 是 "user"（通用标识）
                    # - source 是 profile 中的 name（如 "HP", "AH" 等）
                    is_target_user = False
                    if source.lower() == 'user':
                        is_target_user = True
                    elif user_name and source == user_name:
                        is_target_user = True
                    
                    # 关键修正：目标用户的话应该映射为assistant（模型要学习生成的）
                    # 对话者的话应该映射为user（输入/上下文）
                    role = "assistant" if is_target_user else "user"
                    full_dialogue.append({"role": role, "content": turn.get('content', '')})
                
                # --- 简化逻辑：只预测 continuation，不进行数据扩充 ---
                # 只创建一个样本：context -> continuation
                # if continuation and len(full_dialogue) > 0:
                #     # 确保 context 的最后一轮是 user (对话者)
                #     # 这样符合 LLM "user输入 -> assistant生成" 的标准逻辑
                #     if full_dialogue[-1]['role'] == 'user':
                #         samples.append({
                #             'context': full_dialogue,           # 包含 role 和 content 的列表
                #             'next_question': continuation,      # 目标文本（continuation）
                #             'user_profile': user_profile,       # profile部分（用于向后兼容）
                #             'user_object': user_object,         # 完整的user对象（包含personality，用于人格映射）
                #             'task_description': task_description,
                #             'user_hash': user_hash
                #         })
    
                # 构建完整对话列表用于切分
                # 注意：我们要预测的是用户说话的内容
                # 用户在数据中的 source 可能是 "user"，也可能是在 profile 里显示的 name
                full_dialogue = []
                for turn in context:
                    source = str(turn.get('source', '')).strip()
                    # 判断是否是目标用户（我们要预测的人）说的话：
                    # - source 是 "user"（通用标识）
                    # - source 是 profile 中的 name（如 "HP", "AH" 等）
                    is_target_user = False
                    if source.lower() == 'user':
                        is_target_user = True
                    elif user_name and source == user_name:
                        is_target_user = True
                    
                    # 关键修正：目标用户的话应该映射为assistant（模型要学习生成的）
                    # 对话者的话应该映射为user（输入/上下文）
                    role = "assistant" if is_target_user else "user"
                    full_dialogue.append({"role": role, "content": turn.get('content', '')})
                
                # 将 continuation 也加入队列（作为最后一个样本的目标）
                # continuation 是目标用户的回复，所以应该是 assistant
                if continuation:
                    full_dialogue.append({"role": "assistant", "content": continuation})

                # --- 样本切分逻辑 ---
                # 我们寻找每一个目标用户（assistant）回复的位置，将其作为 target，之前的作为 context
                for i in range(len(full_dialogue)):
                    if full_dialogue[i]['role'] == 'assistant':
                        # 只有当目标用户说话时，我们才可能创建一个样本
                        # context 是 0 到 i-1 轮
                        input_context = full_dialogue[:i]
                        target_text = full_dialogue[i]['content']

                        if target_text and len(input_context) > 0:
                            # 确保 context 的最后一轮是 user (对话者)
                            # 这样符合 LLM "user输入 -> assistant生成" 的标准逻辑
                            if input_context[-1]['role'] == 'user':
                                samples.append({
                                    'context': input_context,     # 包含 role 和 content 的列表
                                    'next_question': target_text, # 目标文本
                                    'user_profile': user_profile,  # profile部分（用于向后兼容）
                                    'user_object': user_object,   # 完整的user对象（包含personality，用于人格映射）
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

def extract_test_samples(test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """提取测试样本，逻辑同上，但保持 context 为原始格式以供模板转换"""
    samples = []

    for item in test_data:
        user_hash = item.get('user_hash', '')
        user_profile = get_user_profile(item)
        task_description = get_task_description(item)
        
        # 获取用户的 name（用于识别 context 中的用户消息）
        user_name = None
        if user_profile and isinstance(user_profile, dict):
            user_name = str(user_profile.get('name', '')).strip()
        
        collections = item.get('task', {}).get('task_behavior_collections', [])
        
        for collection in collections:
            for data_item in collection.get('data', []):
                context = data_item.get('context', [])
                reference = data_item.get('continuation', '')
                
                if context and reference:
                    # 转换格式
                    formatted_context = []
                    for turn in context:
                        source = str(turn.get('source', '')).strip()
                        # 判断是否是目标用户（我们要预测的人）说的话：
                        # - source 是 "user"（通用标识）
                        # - source 是 profile 中的 name（如 "HP", "AH" 等）
                        is_target_user = False
                        if source.lower() == 'user':
                            is_target_user = True
                        elif user_name and source == user_name:
                            is_target_user = True
                        
                        # 目标用户的话应该映射为assistant，对话者的话应该映射为user
                        role = "assistant" if is_target_user else "user"
                        formatted_context.append({"role": role, "content": turn.get('content', '')})
                    
                    samples.append({
                        'context': formatted_context,
                        'reference': reference,
                        'user_profile': user_profile,
                        'task_description': task_description,
                        'user_hash': user_hash
                    })
    return samples


def get_user_history_samples(
    all_samples: List[Dict[str, Any]],
    user_hash: str,
    current_sample: Optional[Dict[str, Any]] = None,
    max_history: int = 2000
) -> List[Dict[str, Any]]:
    """
    获取用户的历史样本（用于构建 H_{s,u}）
    
    Args:
        all_samples: 所有训练样本
        user_hash: 用户哈希
        current_sample: 当前样本（用于排除）
        max_history: 最大历史样本数
    
    Returns:
        历史样本列表
    """
    history = []
    for sample in all_samples:
        if sample['user_hash'] == user_hash:
            if current_sample is None or sample != current_sample:
                history.append(sample)
                if len(history) >= max_history:
                    break
    return history


def select_relevant_history(history: List[str], current_context: List[Dict[str, str]], max_items: int = 15) -> List[str]:
    """
    从历史中选择与当前context最相关的几条
    
    Args:
        history: 完整的历史列表
        current_context: 当前对话上下文
        max_items: 最大选择数量
    
    Returns:
        选择后的历史列表
    """
    if len(history) <= max_items:
        return history
    
    # 提取当前context中的关键词（从所有turn的content中）
    current_keywords = set()
    for turn in current_context:
        content = turn.get('content', '').strip()
        if content:
            # 简单的关键词提取：分词（按空格和标点）
            words = content.lower().split()
            current_keywords.update(words)
    
    # 计算每条历史与当前context的相似度
    history_scores = []
    for hist_item in history:
        # 提取历史项的关键词
        hist_words = set(hist_item.lower().split())
        # 计算交集（共同关键词）
        common_keywords = current_keywords & hist_words
        # 相似度 = 共同关键词数量 / (当前关键词数量 + 历史关键词数量 - 共同关键词数量)
        # 使用Jaccard相似度
        union_size = len(current_keywords | hist_words)
        similarity = len(common_keywords) / union_size if union_size > 0 else 0.0
        history_scores.append((hist_item, similarity))
    
    # 按相似度排序，选择最相关的
    history_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [item for item, _ in history_scores[:max_items]]
    
    # 保持原始顺序（按在原始history中的位置）
    # 但优先选择相似度高的
    result = []
    for hist_item in history:
        if hist_item in selected:
            result.append(hist_item)
            if len(result) >= max_items:
                break
    
    # 如果还没达到max_items，补充剩余的（按时间顺序）
    if len(result) < max_items:
        for hist_item in history:
            if hist_item not in result:
                result.append(hist_item)
                if len(result) >= max_items:
                    break
    
    return result


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


def build_all_user_history_cache(all_samples: List[Dict[str, Any]], max_history: int = 15) -> Dict[str, int]:
    """
    批量构建所有用户的历史缓存
    
    Args:
        all_samples: 所有训练样本
        max_history: 每个用户的最大历史数量
    
    Returns:
        统计信息：{user_hash: history_count}
    """
    user_hashes = set()
    for sample in all_samples:
        user_hash = sample.get('user_hash')
        if user_hash:
            user_hashes.add(user_hash)
    
    stats = {}
    print(f"开始构建 {len(user_hashes)} 个用户的历史缓存...")
    
    for idx, user_hash in enumerate(user_hashes, 1):
        # 获取该用户的所有历史（不排除任何样本，因为这是全量构建）
        history = get_user_only_history(
            all_samples,
            user_hash,
            current_sample=None,
            current_context=None,
            max_history=max_history * 2,  # 缓存时保存更多，使用时再智能选择
            use_cache=False  # 先不保存，等计算完再保存
        )
        
        # 保存到缓存
        if history:
            save_history(user_hash, history)
            stats[user_hash] = len(history)
        
        if idx % 100 == 0:
            print(f"  已处理 {idx}/{len(user_hashes)} 个用户...")
    
    print(f"✅ 缓存构建完成！共 {len(stats)} 个用户有历史数据")
    return stats


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
        
        # 估算 target 的 token 数
        target_tokens = len(tokenizer.encode(next_question, add_special_tokens=False))
        
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
    
    # 4. 预测指令
    if English_flag:
        system_parts.append("\nPredict the user's next message:")
    elif Japanese_flag:
        system_parts.append("\nユーザーの次のメッセージを予測する：")
    else:
        if task_text == "基于用户在 Lovink 问卷中的回答数据，模拟该用户的回答风格和行为模式":
            system_parts.append("\n预测用户针对该问题的回复：")
        elif task_text and "MovieLens" in task_text:
            system_parts.append("\n预测用户对该电影的评分：")
        elif task_text and "Reddit" in task_text:
            system_parts.append("\nPredict the user's response to the comment:")
        elif task_text and "REALTALK" in task_text:
            system_parts.append("\nPredict the user's next message:")
        else:
            system_parts.append("\n预测用户的下一条消息:")
    
    # 组合成 system message
    system_content = "\n\n".join(system_parts)
    messages.append({"role": "system", "content": system_content})
    
    # target_answer 用 [ANSWER] 和 [/ANSWER] 包裹 next_question
    target_answer = f"[ANSWER]\n{next_question}\n[/ANSWER]"
    
    return messages, target_answer


def build_simple_inference_prompt(
    context: List[Dict[str, str]],
    user_profile: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    history: Optional[List[Any]] = None,
    use_profile: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    max_context_turns: int = 10
) -> List[Dict[str, str]]:
    """
    构建简短的推理 prompt（与训练时格式一致）
    
    格式:
    [USER_PROFILE]
    [USER_NAME=xxx] [DIM_OCEAN_EXTRAVERSION=90] ...
    
    [TASK]
    基于用户在 Lovink 对话中的历史数据，模拟该用户的对话行为模式
    
    [RECENT_DIALOGUE]
    User: ...
    Assistant: ...
    
    Predict the user's next message:
    
    Args:
        context: 对话上下文
        user_profile: 用户画像（包括 dimensions 心理维度数据）
        task_description: 任务描述（可选）
        history: 历史对话（可选，当前版本未使用）
        use_profile: 是否使用用户画像
        use_history: 是否使用历史
        use_context: 是否使用上下文
        max_context_turns: 最大保留的对话轮次（默认10）
    
    Returns:
        messages: 消息列表，格式为 [{"role": "system", "content": "..."}]
    """
    messages = []
    
    # 构建 system message（简短格式）
    system_parts = []
    
    # 1. USER_PROFILE 部分 - 使用方括号标签格式（与训练时一致）
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
        # 2. 嵌套格式: {"OCEAN": {"dimensions": {"Extraversion": {"score": 90}}}}
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
                                # 生成标签: [DIM_OCEAN_EXTRAVERSION=90]
                                tag_name = f"DIM_{scale_name.upper()}_{subdim_name.upper()}"
                                profile_tags.append(f"[{tag_name}={score}]")
        
        # 其他 profile 字段（排除 dimensions 和已处理的字段）
        excluded_keys = {'name', 'age', 'gender', 'dimensions', 'unstructured', 'user_hash', 'description'}
        for key, value in user_profile.items():
            if key not in excluded_keys and value:
                # 将其他字段也转为标签格式
                tag_name = f"USER_{key.upper()}"
                profile_tags.append(f"[{tag_name}={value}]")
        
        if profile_tags:
            system_parts.append("[USER_PROFILE]\n" + "\n".join(profile_tags))
    
    # 2. TASK 部分 - 任务描述
    task_text = task_description if task_description else "基于用户在 Lovink 对话中的历史数据，模拟该用户的对话行为模式"
    system_parts.append(f"[TASK]\n{task_text}")
    
    # 3. RECENT_DIALOGUE 部分 - 限制轮次数
    if use_context and context:
        # 智能截断：如果对话轮次过多，只保留最近的N轮
        recent_context = context[-max_context_turns:] if len(context) > max_context_turns else context
        
        dialogue_parts = []
        if len(context) > max_context_turns:
            truncated_count = len(context) - max_context_turns
            dialogue_parts.append(f"[RECENT_DIALOGUE]\n（前面省略了 {truncated_count} 轮对话，以下是最近的 {max_context_turns} 轮）")
        else:
            dialogue_parts.append("[RECENT_DIALOGUE]")
        
        for turn in recent_context:
            role = turn.get('role', turn.get('source', 'unknown'))
            content = turn.get('content', '')
            
            # 统一角色映射
            if str(role).lower() == 'user':
                label = "User"
            elif str(role).lower() in ('assistant', 'other'):
                label = "Assistant"
            else:
                label = "User" if role == 'user' else "Assistant"
            
            dialogue_parts.append(f"{label}: {content}")
        
        system_parts.append("\n".join(dialogue_parts))
    
    # 4. 预测指令
    system_parts.append("\nPredict the user's next message:")
    
    # 组合成 system message
    system_content = "\n\n".join(system_parts)
    messages.append({"role": "system", "content": system_content})
    
    return messages


def format_user_profile_simple(user_profile: Dict[str, Any]) -> str:
    """
    将用户画像格式化为简短字符串
    
    Args:
        user_profile: 用户画像数据
    
    Returns:
        格式化的字符串
    """
    if not user_profile:
        return "user_name: Unknown"
    
    parts = []
    
    # 基础信息
    user_name = user_profile.get('name', 'Unknown')
    parts.append(f"user_name: {user_name}")
    
    # 如果有 unstructured 分析，添加简短摘要
    # unstructured = user_profile.get('unstructured', '')
    # parts.append(f"personality: {unstructured}")

    return "\n".join(parts)