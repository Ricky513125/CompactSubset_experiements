"""
使用 vLLM 进行高性能推理

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
# 单GPU
python inference_vllm.py \
    --checkpoint_dir outputs/Chameleons_8B_context_sampled_seed42 \
    --dataset Chameleons \
    --ablation_config context_only \
    --num_samples 5 \
    --output_dir outputs/leaderboards/Chameleons_8B_context_vllm

# 多GPU (Tensor Parallelism)
python inference_vllm.py \
    --checkpoint_dir outputs/Chameleons_8B_context_sampled_seed42 \
    --dataset Chameleons \a
    --ablation_config context_only \
    --num_samples 5 \
    --output_dir outputs/leaderboards/Chameleons_8B_context_vllm \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9

性能对比:
- HuggingFace Transformers (8 GPU, batch_size=1): ~100 samples/min
- vLLM (1 GPU, batch_size=auto): ~500-1000 samples/min
- vLLM (4 GPU TP, batch_size=auto): ~1500-2000 samples/min
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
from datetime import datetime

# 添加当前目录到 Python 路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 导入 vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    print("错误: vLLM 未安装")
    print("请运行: pip install vllm")
    sys.exit(1)

# 导入推理辅助函数
from inference import (
    load_test_leaderboard,
    get_user_info_from_leaderboard,
)
from data_loader_more_data import load_train_data
import re

# 日本语规范化（如果可用）
try:
    from japanese_text_normalizer import normalize_japanese_text
except ImportError:
    def normalize_japanese_text(text):
        return text


def is_too_similar(new_text: str, existing_texts: List[str], similarity_threshold: float = 0.85) -> bool:
    """
    检查新文本是否与已有文本过于相似
    
    Args:
        new_text: 新生成的文本
        existing_texts: 已有的文本列表
        similarity_threshold: 相似度阈值（默认0.85）
    
    Returns:
        如果过于相似返回True，否则返回False
    """
    if not new_text or not existing_texts:
        return False
    
    new_normalized = new_text.strip().lower()
    if not new_normalized:
        return True  # 空文本视为重复
    
    for existing in existing_texts:
        existing_normalized = existing.strip().lower()
        
        # 完全相同
        if new_normalized == existing_normalized:
            return True
        
        # 计算相似度（基于字符级别的Jaccard相似度）
        set_new = set(new_normalized)
        set_existing = set(existing_normalized)
        intersection = len(set_new & set_existing)
        union = len(set_new | set_existing)
        
        if union > 0:
            similarity = intersection / union
            # 相似度超过阈值认为重复
            if similarity > similarity_threshold:
                return True
        
        # 检查是否一个是另一个的前缀（长度差不超过5个字符）
        if len(new_normalized) > 5 and len(existing_normalized) > 5:
            if new_normalized.startswith(existing_normalized[:10]) or existing_normalized.startswith(new_normalized[:10]):
                if abs(len(new_normalized) - len(existing_normalized)) < 5:
                    return True
    
    return False


def is_garbled_text(text: str) -> bool:
    """
    检测文本是否为乱码
    
    乱码特征：
    1. 大量数字和特殊字符（如 "031516-726626492020..."）
    2. 数字占比过高（>50%）
    3. 连续数字过长（>20个连续数字）
    4. 几乎没有字母或常见标点
    
    Args:
        text: 待检测的文本
    
    Returns:
        如果是乱码返回True，否则返回False
    """
    if not text or len(text.strip()) < 3:
        return False
    
    text_clean = text.strip()
    
    # 统计字符类型
    digit_count = sum(1 for c in text_clean if c.isdigit())
    letter_count = sum(1 for c in text_clean if c.isalpha())
    total_chars = len(text_clean)
    
    if total_chars == 0:
        return False
    
    # 1. 数字占比过高（>60%）
    digit_ratio = digit_count / total_chars
    if digit_ratio > 0.6:
        return True
    
    # 2. 连续数字过长（>20个连续数字）
    import re
    long_digit_sequences = re.findall(r'\d{20,}', text_clean)
    if long_digit_sequences:
        return True
    
    # 3. 几乎没有字母（字母占比<10%且总长度>10）
    if total_chars > 10:
        letter_ratio = letter_count / total_chars
        if letter_ratio < 0.1 and digit_ratio > 0.3:
            return True
    
    # 4. 检查是否主要是特殊字符和数字的组合（如 "057837b47. 0"）
    special_char_count = sum(1 for c in text_clean if not c.isalnum() and c not in ' .,!?;:\'"-')
    if total_chars > 5 and (digit_count + special_char_count) / total_chars > 0.8:
        return True
    
    return False


def clean_generated_text(text: str, max_length: int = 512) -> str:
    """
    清洗生成的文本：
    1. 移除重复的标点符号（如 "OUT!!! OUT!!!" -> "OUT!")
    2. 移除重复的短语和句子
    3. 截断过长的文本
    4. 提取第一段有效对话
    
    Args:
        text: 原始生成文本
        max_length: 最大输出长度（字符数）
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 0. 检测并移除乱码
    if is_garbled_text(text):
        return ""  # 如果是乱码，直接返回空字符串
    
    # 1. 移除元数据和角色标识
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 1.1. 截断到第一个角色标识或标签之前（只保留第一个用户消息）
    # 生成的文本应该是用户的下一条消息，不应该包含 Assistant 回复或其他标签
    # 检测模式：Assistant:, User:, AI的回复是:, [ANSWER], [USER_MESSAGE] 等
    truncation_patterns = [
        r'\s*Assistant:\s*',
        r'\s*User:\s*',
        r'\s*AI的回复是:\s*',
        r'\s*AI的回复:\s*',
        r'\s*\[ANSWER\]\s*',
        r'\s*\[USER_MESSAGE\]\s*',
        r'\s*\[SYSTEM_PROMPT',
        r'\s*\[AI\]\s*',
        r'\s*\[AI_PROFILE\]',
        r'\s*\[USER\]\s*',
        r'\s*\[ASSISTANT\]\s*',
    ]
    
    # 找到第一个匹配的截断模式，截断到该位置之前
    min_pos = len(text)
    for pattern in truncation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pos = match.start()
            if pos < min_pos:
                min_pos = pos
    
    # 如果找到了截断点，截断文本
    if min_pos < len(text):
        text = text[:min_pos].strip()
    
    # 1.2. 移除所有方括号内的辅助信息（如 [角色], [需求], [FINISH], [MASK] 等）
    # 这些方括号内容都是训练数据中的辅助标记，不应该出现在生成的文本中
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # 1.3. 移除训练数据格式标签（如 [ANSWER], [USER_MESSAGE], [SYSTEM_PROMPT] 等）
    # 这些标签不应该出现在生成的文本中（在截断后可能仍有残留）
    # 注意：由于上面已经移除了所有方括号内容，这一步主要是为了确保清理干净
    text = re.sub(r'\[ANSWER\]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[USER_MESSAGE\]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SYSTEM_PROMPT\]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SYSTEM_PROMPT\d+\]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[AI\]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[AI_PROFILE\]\s*\d*', '', text, flags=re.IGNORECASE)
    
    # 1.4. 规范化特殊Unicode字符（em dash, en dash等）
    # 将各种破折号统一转换为普通连字符或空格
    text = text.replace('\u2014', '-')  # em dash (—) -> -
    text = text.replace('\u2013', '-')  # en dash (–) -> -
    text = text.replace('\u2015', '-')  # horizontal bar (―) -> -
    text = text.replace('\u2010', '-')  # hyphen (‐) -> -
    text = text.replace('\u2011', '-')  # non-breaking hyphen (‑) -> -
    # 将其他特殊引号转换为普通引号
    text = text.replace('\u2018', "'")  # left single quotation mark (') -> '
    text = text.replace('\u2019', "'")  # right single quotation mark (') -> '
    text = text.replace('\u201C', '"')  # left double quotation mark (") -> "
    text = text.replace('\u201D', '"')  # right double quotation mark (") -> "
    text = text.replace('\u201E', '"')  # double low-9 quotation mark („) -> "
    text = text.replace('\u201F', '"')  # double high-reversed-9 quotation mark (‟) -> "
    # 将其他特殊空格转换为普通空格
    text = text.replace('\u00A0', ' ')  # non-breaking space -> space
    text = text.replace('\u2000', ' ')  # en quad -> space
    text = text.replace('\u2001', ' ')  # em quad -> space
    text = text.replace('\u2002', ' ')  # en space -> space
    text = text.replace('\u2003', ' ')  # em space -> space
    text = text.replace('\u2004', ' ')  # three-per-em space -> space
    text = text.replace('\u2005', ' ')  # four-per-em space -> space
    text = text.replace('\u2006', ' ')  # six-per-em space -> space
    text = text.replace('\u2007', ' ')  # figure space -> space
    text = text.replace('\u2008', ' ')  # punctuation space -> space
    text = text.replace('\u2009', ' ')  # thin space -> space
    text = text.replace('\u200A', ' ')  # hair space -> space
    text = text.replace('\u202F', ' ')  # narrow no-break space -> space
    text = text.replace('\u205F', ' ')  # medium mathematical space -> space
    text = text.replace('\u3000', ' ')  # ideographic space -> space
    
    # 1.5. 移除开头的标点符号和空白字符
    # 例如：". he. what about..." -> "he. what about..."
    # 注意：此时特殊Unicode字符已经规范化为普通字符，所以只需要处理普通字符
    text = text.lstrip(r'.!?,\s\-')
    # 如果开头是单个标点后跟空格，也移除（包括连字符）
    text = re.sub(r'^[.!?,:;\-]\s+', '', text)
    # 如果开头是连字符（可能来自em dash的转换），也移除
    if text.startswith('-'):
        text = text.lstrip('-').lstrip()
    
    # 2. 清理过度重复的标点符号（保留最多1个）
    # 例如："OUT!!! OUT!!!" -> "OUT!", "....." -> "."
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)  # 重复3次或更多 -> 1次
    
    # 3. 清理连续重复的相同字符（保留最多2个，但标点符号只保留1个）
    # 例如："aaaaa" -> "aa", "!!!!!" -> "!"
    # 先处理标点符号
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # 再处理其他字符
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 4. 清理重复的短语和句子（更智能的去重）
    # 先按句子分割
    sentence_pattern = r'([.!?]\s*|$)'
    parts = re.split(sentence_pattern, text)
    
    # 重组句子并去重
    seen_sentences = set()
    unique_parts = []
    for i in range(0, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sentence = (parts[i] + parts[i + 1]).strip()
        else:
            sentence = parts[i].strip()
        
        if not sentence:
            continue
        
        # 标准化句子用于比较（转小写，移除多余空格）
        sentence_key = re.sub(r'\s+', ' ', sentence.lower().strip())
        # 如果句子太短（少于3个字符），跳过去重检查
        if len(sentence_key) < 3:
            unique_parts.append(sentence)
            continue
        
        # 检查是否是重复句子（允许部分匹配，如果相似度>80%）
        is_duplicate = False
        for seen in seen_sentences:
            # 简单的相似度检查：如果一个是另一个的子串，或者长度相近且内容相似
            if sentence_key in seen or seen in sentence_key:
                is_duplicate = True
                break
            # 如果两个句子长度相近，检查字符重叠度
            if abs(len(sentence_key) - len(seen)) < max(len(sentence_key), len(seen)) * 0.3:
                common_chars = sum(1 for c in sentence_key if c in seen)
                if common_chars / max(len(sentence_key), len(seen)) > 0.8:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_sentences.add(sentence_key)
            unique_parts.append(sentence)
    
    text = ' '.join(unique_parts)
    
    # 5. 按句子分割（英文句号、问号、感叹号）
    sentences = re.split(r'([.!?]\s+)', text)
    
    # 6. 去重句子（保留第一次出现）
    seen = set()
    unique_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        sentence_clean = sentence.strip().lower()
        if sentence_clean and sentence_clean not in seen:
            seen.add(sentence_clean)
            unique_sentences.append(sentence)
    
    result = ''.join(unique_sentences)
    
    # 如果去重后为空，保留原始文本
    if not result.strip():
        result = text
    
    # 7. 提取第一段有效内容（按段落分割）
    paragraphs = [p.strip() for p in result.split('\n') if p.strip()]
    if paragraphs:
        result = paragraphs[0]
    
    # 8. 截断过长文本（保留完整的句子）
    if len(result) > max_length:
        # 尝试在句子边界截断
        truncated = result[:max_length]
        # 找到最后一个句号、问号或感叹号
        last_punct = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        if last_punct > max_length * 0.5:  # 如果找到的标点在文本的后半部分
            result = truncated[:last_punct + 1]
        else:
            result = truncated
    
    # 9. 移除开头的标点符号（再次确保，因为可能在前面步骤中又出现了）
    # 例如：". he. what about..." -> "he. what about..."
    # 注意：此时特殊Unicode字符已经规范化，所以只需要处理普通字符
    # 注意：不要移除单引号，因为它可能是单词的一部分（如 "don't", "'tis"）
    result = result.lstrip(r'.!?,\s\-')
    result = re.sub(r'^[.!?,:;\-]\s+', '', result)
    # 如果开头是连字符（可能来自em dash的转换），也移除
    if result.startswith('-'):
        result = result.lstrip('-').lstrip()
    
    # 10. 最后清理：移除多余的空白字符
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()
    
    # 11. 确保结果不为空
    if not result:
        return ""
    
    # 12. 如果结果以标点开头，尝试找到第一个单词
    if result and result[0] in '.!?,:;':
        # 找到第一个字母或数字
        match = re.search(r'[a-zA-Z0-9]', result)
        if match:
            result = result[match.start():]
    
    # 13. 处理开头的单引号：检测并修复不完整的开头
    # 例如："'t " 或 "'the" 可能是模型生成的不完整文本
    if result and result.startswith("'"):
        # 检查是否是常见的完整缩写（如 'tis, 'twas, 'til）
        common_contractions = ["'tis", "'twas", "'til", "'tween", "'neath", "'gainst"]
        is_valid_contraction = any(result.lower().startswith(contraction) for contraction in common_contractions)
        
        if not is_valid_contraction:
            # 如果单引号后跟空格，移除单引号和空格
            if len(result) > 1 and result[1:2] == ' ':
                result = result[2:].lstrip()
            # 如果单引号后跟单个字母+空格（如 "'t "），可能是模型生成的不完整文本
            elif len(result) > 2 and result[1:2].isalpha() and result[2:3] == ' ':
                # 检查是否是常见的单字母缩写（如 's, 'm, 'd, 'll, 've, 're）
                single_letter_contractions = ["'s", "'m", "'d", "'ll", "'ve", "'re"]
                if result[1:2].lower() + result[2:3] not in single_letter_contractions:
                    # 如果不是常见的单字母缩写，可能是误生成，移除单引号
                    result = result[1:].lstrip()
            # 如果单引号后直接跟单词（如 "'the"），可能是误加的单引号，移除它
            elif len(result) > 1 and result[1:2].isalpha():
                # 检查是否是完整的单词（至少3个字符）
                match = re.match(r"^'([a-zA-Z]{3,})", result)
                if match:
                    # 如果是完整的单词，移除单引号
                    result = result[1:]
            # 如果单引号后直接跟非字母字符，移除单引号
            elif len(result) > 1 and not result[1:2].isalpha():
                result = result[1:].lstrip()
    
    return result


def build_inference_prompt(
    user_info: Dict[str, Any],
    use_profile: bool = True,
    use_context: bool = True,
    use_history: bool = False,
    dataset_name: str = "Unknown",
    max_context_turns: int = 15  # 限制 context 的最大轮次数（保守估计，确保不超过 max_model_len）
) -> str:
    """
    构建推理 prompt（与训练时的格式一致）
    
    Args:
        user_info: 用户信息（包含 profile, context, history）
        use_profile: 是否使用 profile
        use_context: 是否使用对话上下文
        use_history: 是否使用历史信息
        dataset_name: 数据集名称
    
    Returns:
        完整的 prompt 字符串
    """
    parts = []
    
    # 1. 用户画像
    if use_profile and user_info.get('user_profile'):
        profile = user_info['user_profile']
        
        # 对于 DMSC/MovieReview，使用与训练时一致的简单格式
        if dataset_name in ["DMSC", "MovieReview"]:
            # DMSC/MovieReview: 使用训练时的简单格式 "用户: xxx"
            user_name = profile.get('name', 'Unknown')
            parts.append(f"用户: {user_name}")
        else:
            # 其他数据集：使用标签格式
            profile_tags = []
            
            if 'name' in profile:
                profile_tags.append(f"[USER_NAME={profile['name']}]")
            if 'age' in profile:
                profile_tags.append(f"[USER_AGE={profile['age']}]")
            if 'gender' in profile:
                profile_tags.append(f"[USER_GENDER={profile['gender']}]")
            
            # 人格维度（dimensions）
            if 'dimensions' in profile:
                dims = profile['dimensions']
                if isinstance(dims, dict):
                    for dim_key, dim_score in dims.items():
                        if isinstance(dim_score, (int, float)):
                            # dim_key 格式: "Ocean.Extraversion" 或简单键名
                            tag_name = f"DIM_{dim_key.upper().replace('.', '_')}"
                            profile_tags.append(f"[{tag_name}={dim_score}]")
            
            # 其他以 DIM_ 开头的字段
            for key, value in profile.items():
                if key.startswith('DIM_') or key.startswith('dim_'):
                    profile_tags.append(f"[{key.upper()}={value}]")
            
            if profile_tags:
                parts.append("[USER_PROFILE]")
                parts.extend(profile_tags)
            parts.append("")
    
    # 2. 任务描述（根据数据集选择语言）
    # 对于 DMSC/MovieReview，训练时没有任务描述，所以推理时也不添加
    if dataset_name not in ["DMSC", "MovieReview"]:
        if dataset_name == "Chameleons":
            task_text = "Given the historical dialogue of a character in a movie, model the character's speaking style and behavioral patterns, and predict the next utterance the user would produce."
        elif dataset_name == "RealPersonaChat":
            task_text = "RealPersonaChatデータセットにおけるユーザーの過去の会話データに基づき、当該ユーザーの会話行動パターンをシミュレートする："
        elif dataset_name == "LovinkQuestionnaire":
            # LovinkQuestionnaire: 使用与训练时一致的任务描述
            task_text = "基于用户在 Lovink 问卷中的回答数据，模拟该用户的回答风格和行为模式"
        elif dataset_name == "LovinkDialogue":
            # LovinkDialogue: 使用与训练时一致的任务描述
            task_text = "基于用户在 Lovink 对话中的历史数据，模拟该用户的对话行为模式"
        else:
            task_text = f"基于用户在 {dataset_name} 中的历史数据，模拟该用户的对话行为模式，并预测用户的下一条回复。"
        
        parts.append(f"[TASK]\n{task_text}")
        parts.append("")
    
    # 3. 历史信息
    if use_history and user_info.get('history'):
        history = user_info['history']
        # 对于 DMSC/MovieReview，使用与训练时一致的格式
        if dataset_name in ["DMSC", "MovieReview"]:
            # DMSC/MovieReview: 使用训练时的格式 "历史影评记录 (N条):" 和 "电影《xxx》: 评论"
            if history:
                parts.append(f"\n历史影评记录 ({len(history)}条):")
                for h in history:
                    # history 可能是 dict 格式 {'movie': 'xxx', 'review': 'yyy'} 或列表格式
                    if isinstance(h, dict):
                        movie = h.get('movie', '')
                        review = h.get('review', '')
                        if movie and review:
                            parts.append(f"  电影《{movie}》: {review}")
                    else:
                        # 兼容其他格式
                        content = str(h)
                        if len(content) > 200:
                            content = content[:197] + "..."
                        parts.append(f"  {content}")
                parts.append("")
        else:
            # 其他数据集：使用标签格式
            history_parts = ["[HISTORY]"]
            history_to_use = history[-15:]  # 最近15条
            for i, item in enumerate(history_to_use, 1):
                if isinstance(item, str):
                    content = item
                elif isinstance(item, dict):
                    content = item.get('next_question', '') or item.get('content', '') or item.get('continuation', '')
                else:
                    content = str(item)
                
                if content:
                    if len(content) > 200:
                        content = content[:197] + "..."
                    history_parts.append(f"{i}. {content}")
            
            if len(history_parts) > 1:
                parts.extend(history_parts)
            parts.append("")
    
    # 4. 对话上下文（使用 [RECENT_DIALOGUE] 标签，与训练时一致）
    # 对于 DMSC/MovieReview，如果 context 为空但有 continuation_prefix，使用 continuation_prefix
    if use_context:
        context = user_info.get('context', [])
        continuation_prefix = user_info.get('continuation_prefix', '')
        
        if context:
            # 有 context，正常处理
            parts.append("[RECENT_DIALOGUE]")
            # 限制 context 长度：只保留最近的对话轮次
            if len(context) > max_context_turns:
                context = context[-max_context_turns:]
            
            for turn in context:
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                # 如果单条内容太长，也截断（每条最多 300 字符，更保守）
                if len(content) > 300:
                    content = content[:297] + "..."
                label = "User" if role == 'user' else "Assistant" if role == 'assistant' else "Unknown"
                parts.append(f"{label}: {content}")
            parts.append("")
        elif continuation_prefix and dataset_name in ["DMSC", "MovieReview"]:
            # DMSC/MovieReview: context 为空但有 continuation_prefix
            # 训练时格式：直接使用电影名，格式为 "模仿用户风格为电影《xxx》写一条影评："
            movie_name = continuation_prefix.rstrip(': ').strip()
            parts.append(f"\n模仿用户风格为电影《{movie_name}》写一条影评：")
    
    # 5. 生成提示（根据数据集选择语言）
    # 对于 DMSC/MovieReview，提示已经在上面添加了（"模仿用户风格为电影《xxx》写一条影评："）
    if dataset_name not in ["DMSC", "MovieReview"]:
        if dataset_name == "Chameleons":
            parts.append("Predict the user's next message:")
        elif dataset_name == "RealPersonaChat":
            parts.append("ユーザーの次のメッセージを予測してください：")
        elif dataset_name == "LovinkQuestionnaire":
            # LovinkQuestionnaire: 使用与训练时一致的生成提示
            parts.append("预测用户针对该问题的回复：")
        elif dataset_name == "LovinkDialogue":
            # LovinkDialogue: 使用与训练时一致的生成提示
            parts.append("预测用户的下一条消息:")
        else:
            parts.append("根据以上信息，预测用户的下一条回复:")
    
    return "\n".join(parts)


def generate_with_vllm(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams | List[SamplingParams],
    show_progress: bool = True
) -> List[str]:
    """
    使用 vLLM 批量生成
    
    Args:
        llm: vLLM LLM 实例
        prompts: 待生成的 prompts 列表
        sampling_params: 采样参数（单个或列表，如果是列表则每个prompt使用对应的参数）
        show_progress: 是否显示进度条
    
    Returns:
        生成的文本列表
    """
    if show_progress:
        print(f"使用 vLLM 生成 {len(prompts)} 个样本...")
        if isinstance(sampling_params, list):
            print(f"  使用不同的采样参数以增加多样性...")
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    # 提取生成的文本
    generated_texts = []
    for output in outputs:
        # vLLM 输出格式: output.outputs[0].text
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    
    return generated_texts


def run_inference_vllm(
    checkpoint_dir: str,
    scenario_path: str,
    ablation_config: str,
    use_profile: bool,
    use_context: bool,
    use_history: bool,
    num_samples: int,
    output_dir: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    temperature: float = 1.2,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 512,
    seed: int = 42,
    max_context_turns: int = 15,  # 新增：context 最大轮次数
    max_chars_per_turn: int = 300  # 新增：每轮对话最大字符数
):
    """
    使用 vLLM 运行推理
    """
    print("=" * 80)
    print("vLLM 推理配置")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: {os.path.basename(scenario_path)}")
    print(f"数据路径: {scenario_path}")
    print(f"Ablation: {ablation_config}")
    print(f"  use_profile: {use_profile}")
    print(f"  use_context: {use_context}")
    print(f"  use_history: {use_history}")
    print(f"Samples per user: {num_samples}")
    print(f"Output: {output_dir}")
    print(f"Tensor Parallel: {tensor_parallel_size} GPU(s)")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Model Length: {max_model_len}")
    print(f"Max Context Turns: {max_context_turns}")
    print(f"Max Chars Per Turn: {max_chars_per_turn}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    train_data_path = os.path.join(scenario_path, 'train.json')
    
    if not os.path.exists(test_leaderboard_path):
        print(f"错误: 测试数据不存在: {test_leaderboard_path}")
        print(f"请检查数据集路径: {scenario_path}")
        return
    
    test_leaderboard = load_test_leaderboard(test_leaderboard_path)
    train_data = load_train_data(train_data_path)
    
    print(f"测试集用户数: {len(test_leaderboard)}")
    
    # 初始化 vLLM
    print(f"\n初始化 vLLM (Tensor Parallel={tensor_parallel_size})...")
    start_time = time.time()
    
    llm = LLM(
        model=checkpoint_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",  # 使用 bfloat16 以节省内存
        seed=seed,
        enforce_eager=False,  # 使用 CUDA graph 加速
    )
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 (耗时: {load_time:.2f}s)")
    
    # 获取数据集名称（需要在采样参数设置前确定）
    dataset_name = os.path.basename(scenario_path)
    
    # 采样参数（根据数据集类型调整）
    # 对于 Questionnaire 数据集：使用相同的 temperature，但只生成1个答案，然后复制5次
    # 对于其他数据集：使用多样性采样（提高temperature）
    is_questionnaire = 'Questionnaire' in dataset_name or 'questionnaire' in dataset_name.lower()
    
    # 所有数据集使用相同的 temperature 设置
    enhanced_temperature = max(temperature, 1.2)  # 至少1.2以增加多样性
    
    if is_questionnaire:
        # Questionnaire: 使用相同的 temperature，但只生成1个答案，然后复制5次
        actual_num_samples = 1  # 只生成1个，后续复制5次
        print(f"\n采样参数 (Questionnaire 模式):")
        print(f"  temperature: {enhanced_temperature} (与其他数据集保持一致)")
        print(f"  top_p: {top_p}")
        print(f"  top_k: {top_k}")
        print(f"  max_tokens: {max_tokens}")
        print(f"  seed: {seed}")
        print(f"  注意: 将生成1个回答，然后复制{num_samples}次")
    else:
        # 其他数据集：使用多样性采样
        actual_num_samples = num_samples
        print(f"\n采样参数 (多样性采样):")
        print(f"  temperature: {enhanced_temperature} (增强多样性)")
        print(f"  top_p: {top_p}")
        print(f"  top_k: {top_k}")
        print(f"  max_tokens: {max_tokens}")
        print(f"  base_seed: {seed} (每个样本会使用不同的seed)")
    
    # 准备所有推理请求
    print("\n准备推理请求...")
    all_prompts = []
    all_metadata = []  # 存储每个 prompt 的元信息，用于填充回 test_leaderboard
    all_sampling_params = []  # 为每个请求存储不同的采样参数
    
    # 遍历 test_leaderboard，记录每个 data_item 的位置
    for test_sample_idx, test_sample in enumerate(tqdm(test_leaderboard, desc="构建 prompts")):
        # 从测试样本中获取用户信息
        user_info = get_user_info_from_leaderboard(
            sample=test_sample,
            train_data=train_data
        )
        
        if not user_info:
            continue
        
        # 获取 user_hash
        user_hash = test_sample.get('user_hash', test_sample.get('user', {}).get('hash', 'unknown'))
        
        # ✅ 修复：正确提取 context 从 test_leaderboard.json 的结构
        # test_leaderboard.json的结构：context在task.task_behavior_collections[0].data[0].context
        task = test_sample.get('task', {})
        collections = task.get('task_behavior_collections', [])
        
        if not collections:
            print(f"警告: 用户 {user_hash} 缺少 task_behavior_collections，跳过")
            continue
        
        # 处理每个collection中的data
        for collection_idx, collection in enumerate(collections):
            data_items = collection.get('data', [])
            for data_item_idx, data_item in enumerate(data_items):
                # 提取 context
                raw_context = data_item.get('context', [])
                
                # 对于 DMSC/MovieReview 数据集，即使 context 为空，也可能有 continuation_prefix
                # 这种情况下应该继续处理（使用 continuation_prefix 作为提示）
                if not raw_context:
                    if dataset_name in ["DMSC", "MovieReview"]:
                        # DMSC/MovieReview: 即使 context 为空，也继续处理（使用 continuation_prefix）
                        continuation_prefix = data_item.get('continuation_prefix', '')
                        if not continuation_prefix:
                            continue  # 如果连 continuation_prefix 都没有，跳过
                        # 创建一个空的 context，后续会在 prompt 中使用 continuation_prefix
                        raw_context = []
                    else:
                        # 其他数据集：context 为空则跳过
                        continue
                
                # 转换 context 格式：从 {source, content, timestamp} 转换为 {role, content}
                # 在对话生成任务中：
                # - 'assistant' 表示目标用户（我们要预测的用户，如 DANTE）
                # - 'user' 表示对话者（其他人，如 CAITLIN）
                user_name = user_info.get('user_profile', {}).get('name', '')
                converted_context = []
                
                # 限制 context 长度：只保留最近的对话轮次（在转换时就截断，避免构建过长的 prompt）
                # 使用传入的参数
                max_turns = max_context_turns
                max_chars = max_chars_per_turn
                
                # 只处理最近的 max_turns 轮
                context_to_process = raw_context[-max_turns:] if len(raw_context) > max_turns else raw_context
                
                for turn in context_to_process:
                    source = turn.get('source', '')
                    content = turn.get('content', '')
                    
                    # 截断单条内容（如果太长）
                    if len(content) > max_chars:
                        content = content[:max_chars-3] + "..."
                    
                    # 判断 role：如果 source 匹配用户名（目标用户），则为 'assistant'，否则为 'user'（对话者）
                    role = 'assistant' if source == user_name else 'user'
                    converted_context.append({
                        'role': role,
                        'content': content
                    })
                
                # 将 context 添加到 user_info
                user_info_with_context = user_info.copy()
                user_info_with_context['context'] = converted_context
                
                # 对于 DMSC/MovieReview，如果有 continuation_prefix，将其添加到 user_info
                if dataset_name in ["DMSC", "MovieReview"]:
                    continuation_prefix = data_item.get('continuation_prefix', '')
                    if continuation_prefix:
                        user_info_with_context['continuation_prefix'] = continuation_prefix
                
                # 获取历史证据（如果需要）
                if use_history and user_info.get('user_train_samples'):
                    # 对于 DMSC/MovieReview，需要从训练样本中提取历史影评，格式化为训练时的格式
                    if dataset_name in ["DMSC", "MovieReview"]:
                        # DMSC/MovieReview: 从训练样本中提取历史影评
                        # 训练样本格式: {'movie_name': 'xxx', 'next_question': 'yyy', ...}
                        # 需要转换为: [{'movie': 'xxx', 'review': 'yyy'}, ...]
                        history_list = []
                        train_samples = user_info['user_train_samples']
                        # 使用所有训练样本作为历史（或最近N个）
                        for sample in train_samples:
                            movie_name = sample.get('movie_name', '')
                            review = sample.get('next_question', '')
                            if movie_name and review:
                                history_list.append({
                                    'movie': movie_name,
                                    'review': review,
                                    'timestamp': sample.get('timestamp', '')
                                })
                        user_info_with_context['history'] = history_list
                    else:
                        # 其他数据集：直接使用训练样本
                        history_evidence = user_info['user_train_samples'][-3:]  # 使用最近3个样本
                        user_info_with_context['history'] = history_evidence
                else:
                    user_info_with_context['history'] = []
                
                # 为每个 data_item 生成样本（根据数据集类型决定生成数量）
                for sample_idx in range(actual_num_samples):
                    prompt = build_inference_prompt(
                        user_info=user_info_with_context,
                        use_profile=use_profile,
                        use_context=use_context,
                        use_history=use_history,
                        dataset_name=dataset_name,
                        max_context_turns=max_context_turns
                    )
                    
                    all_prompts.append(prompt)
                    # 记录位置信息，用于后续填充回 test_leaderboard
                    all_metadata.append({
                        'test_sample_idx': test_sample_idx,
                        'collection_idx': collection_idx,
                        'data_item_idx': data_item_idx,
                        'sample_idx': sample_idx
                    })
                    # 采样参数：所有数据集使用相同的 temperature，但 Questionnaire 只生成1个样本
                    if is_questionnaire:
                        # Questionnaire: 使用相同的 temperature，固定seed（因为只生成1个）
                        all_sampling_params.append(SamplingParams(
                            temperature=enhanced_temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            seed=seed,  # 固定seed
                            skip_special_tokens=True,
                        ))
                    else:
                        # 其他数据集：为每个样本使用不同的seed以增加多样性
                        sample_seed = seed + sample_idx * 1000  # 确保每个样本的seed不同
                        all_sampling_params.append(SamplingParams(
                            temperature=enhanced_temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            seed=sample_seed,
                            skip_special_tokens=True,
                        ))
    
    print(f"总推理请求数: {len(all_prompts)}")
    
    # 批量推理
    print("\n开始批量推理...")
    inference_start = time.time()
    
    # 使用每个请求不同的采样参数以增加多样性
    generated_texts = generate_with_vllm(
        llm=llm,
        prompts=all_prompts,
        sampling_params=all_sampling_params,  # 传递列表，每个请求使用不同的参数
        show_progress=True
    )
    
    inference_time = time.time() - inference_start
    throughput = len(all_prompts) / inference_time
    
    print(f"\n✓ 推理完成")
    print(f"  总样本数: {len(all_prompts)}")
    print(f"  推理时间: {inference_time:.2f}s")
    print(f"  吞吐量: {throughput:.2f} samples/sec ({throughput * 60:.0f} samples/min)")
    
    # 将生成的文本填充回 test_leaderboard
    print(f"\n填充结果到 test_leaderboard...")
    
    # 按 data_item 组织生成的文本，并进行清洗
    print(f"\n清洗生成的文本...")
    data_item_continuations = {}  # {(test_sample_idx, collection_idx, data_item_idx): [continuations]}
    
    cleaned_count = 0
    empty_count = 0
    sample_texts = []  # 用于调试：保存前几个生成的文本
    for idx, (metadata, generated_text) in enumerate(zip(all_metadata, generated_texts)):
        key = (metadata['test_sample_idx'], metadata['collection_idx'], metadata['data_item_idx'])
        if key not in data_item_continuations:
            data_item_continuations[key] = []
        
        # 保存前几个原始文本用于调试
        if idx < 5:
            sample_texts.append(f"样本 {idx}: {repr(generated_text[:200])}")
        
        # 对于 Questionnaire，使用更宽松的清洗（保留更多内容）
        if is_questionnaire:
            # Questionnaire: 只做基本清理，保留更多原始内容
            cleaned_text = generated_text.strip()
            # 只移除明显的元数据标签，保留其他内容
            cleaned_text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', cleaned_text)
            cleaned_text = re.sub(r'\[ANSWER\]\s*|\[USER_MESSAGE\]\s*|\[SYSTEM_PROMPT', '', cleaned_text, flags=re.IGNORECASE)
            # 移除方括号内容（但保留文本本身）
            cleaned_text = re.sub(r'\[[^\]]*\]', '', cleaned_text)
            # 截断到合理长度
            if len(cleaned_text) > 300:
                # 尝试在句子边界截断
                truncated = cleaned_text[:300]
                last_punct = max(truncated.rfind('。'), truncated.rfind('！'), truncated.rfind('？'), 
                               truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                if last_punct > 200:
                    cleaned_text = truncated[:last_punct + 1]
                else:
                    cleaned_text = truncated
            cleaned_text = cleaned_text.strip()
        else:
            # 其他数据集：使用完整的清洗函数
            cleaned_text = clean_generated_text(generated_text, max_length=300)
        
        if cleaned_text != generated_text.strip():
            cleaned_count += 1
        
        # 直接添加清洗后的文本（不去重）
        if cleaned_text:
            data_item_continuations[key].append(cleaned_text)
        else:
            empty_count += 1
            # 如果清洗后为空，保留原始文本的前300个字符
            if generated_text.strip():
                fallback_text = generated_text.strip()[:300]
                data_item_continuations[key].append(fallback_text)
    
    # 打印前几个样本用于调试
    if sample_texts and is_questionnaire:
        print(f"\n前5个生成的原始文本样本（用于调试）:")
        for text in sample_texts:
            print(f"  {text}")
    
    if cleaned_count > 0:
        print(f"✓ 已清洗 {cleaned_count} 个生成样本")
    if empty_count > 0:
        print(f"  警告: {empty_count} 个生成样本清洗后为空，已使用原始文本")
    
    # 确保每个 data_item 都有 num_samples 个 continuations
    print(f"\n确保每个 data_item 都有 {num_samples} 个 continuations...")
    insufficient_count = 0
    for key, continuations in data_item_continuations.items():
        if len(continuations) < num_samples:
            insufficient_count += 1
            # 如果不足，从已有的样本中填充
            original_count = len(continuations)
            while len(continuations) < num_samples:
                if original_count > 0:
                    if is_questionnaire:
                        # Questionnaire: 直接复制第一个（最可能的）回答
                        continuations.append(continuations[0])
                    else:
                        # 其他数据集: 循环使用已有的样本
                        base_idx = (len(continuations) - original_count) % original_count
                        continuations.append(continuations[base_idx])
                else:
                    # 如果完全没有样本，使用占位符
                    continuations.append("[生成失败]")
            data_item_continuations[key] = continuations
    
    if insufficient_count > 0:
        print(f"  警告: {insufficient_count} 个 data_item 使用了备用策略填充")
    
    
    # 填充到 test_leaderboard
    filled_count = 0
    insufficient_count = 0
    for (test_sample_idx, collection_idx, data_item_idx), continuations in data_item_continuations.items():
        test_sample = test_leaderboard[test_sample_idx]
        collection = test_sample['task']['task_behavior_collections'][collection_idx]
        data_item = collection['data'][data_item_idx]
        
        # 确保有 num_samples 个（前面已经保证，这里再次确认）
        final_continuations = continuations[:num_samples]
        if len(final_continuations) < num_samples:
            # 如果还是不够，填充到 num_samples 个
            while len(final_continuations) < num_samples:
                if len(final_continuations) > 0:
                    final_continuations.append(final_continuations[-1])  # 复制最后一个
                else:
                    final_continuations.append("[生成失败]")  # 占位符
            insufficient_count += 1
        
        data_item['continuations'] = final_continuations
        filled_count += 1
    
    print(f"✓ 已填充 {filled_count} 个 data_item 的 continuations")
    if insufficient_count > 0:
        print(f"  警告: {insufficient_count} 个 data_item 使用了备用策略填充")
    
    # 保存填充后的完整 test_leaderboard.json
    print(f"\n保存填充后的 test_leaderboard 到: {output_dir}")
    output_test_leaderboard_path = os.path.join(output_dir, 'test_leaderboard.json')
    with open(output_test_leaderboard_path, 'w', encoding='utf-8') as f:
        json.dump(test_leaderboard, f, indent=2, ensure_ascii=False)
    
    print(f"✓ test_leaderboard.json 已保存: {output_test_leaderboard_path}")
    
    # 保存汇总信息
    summary = {
        'checkpoint': checkpoint_dir,
        'scenario': scenario_path,
        'ablation_config': ablation_config,
        'num_data_items_filled': filled_count,
        'num_samples_per_item': num_samples,
        'total_samples': len(all_prompts),
        'inference_time_seconds': inference_time,
        'throughput_samples_per_sec': throughput,
        'tensor_parallel_size': tensor_parallel_size,
        'sampling_params': {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_tokens': max_tokens
        },
        'output_file': output_test_leaderboard_path,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(output_dir, 'inference_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 汇总信息已保存: {summary_file}")
    
    # 清理模型
    try:
        destroy_model_parallel()
    except:
        pass
    
    print("\n" + "=" * 80)
    print("vLLM 推理完成！")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='vLLM 高性能推理')
    
    # 模型和数据配置
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型 checkpoint 目录')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称 (Chameleons, DMSC, etc.)')
    parser.add_argument('--scenario_path', type=str, default=None,
                       help='场景数据路径（默认从 dataset 推断）')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 
                               'profile_and_context', 'history_and_context', 
                               'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    
    # 推理参数
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个用户生成的样本数（默认: 5）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    
    # vLLM 配置
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor Parallel 大小（使用多少张 GPU，默认: 1）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU 内存利用率（0.0-1.0，默认: 0.9）')
    parser.add_argument('--max_model_len', type=int, default=8192,
                       help='最大模型序列长度（默认: 8192，可根据需要设置为更大值，如 16384）')
    
    # 采样参数
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='采样温度（默认: 1.0）')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p 采样（默认: 0.9）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k 采样（默认: 50）')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='最大生成 token 数（默认: 512）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    # Context 长度控制参数
    parser.add_argument('--max_context_turns', type=int, default=15,
                       help='Context 最大对话轮次数（默认: 15，可根据需要调整）')
    parser.add_argument('--max_chars_per_turn', type=int, default=300,
                       help='每轮对话最大字符数（默认: 300，可根据需要调整）')
    
    args = parser.parse_args()
    
    # 推断 scenario_path
    if args.scenario_path is None:
        # 数据集路径映射（不同数据集在不同基础路径下）
        dataset_path_mapping = {
            'Chameleons': '/mnt/parallel/GIDigitalTwinBench/RealSelf/Chameleons',
            'DMSC': '/mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC',
            'MovieReview': '/mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC',  # MovieReview 使用 DMSC 数据
            'LovinkDialogue': '/mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue',
            'LovinkQuestionnaire': '/mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkQuestionnaire',
            'RealPersonaChat': '/mnt/parallel/GIDigitalTwinBench/IdealSelf/RealPersonaChat',
        }
        
        if args.dataset in dataset_path_mapping:
            args.scenario_path = dataset_path_mapping[args.dataset]
        else:
            # 默认尝试 IdealSelf
            args.scenario_path = f"/mnt/parallel/GIDigitalTwinBench/IdealSelf/{args.dataset}"
        
        print(f"自动推断数据路径: {args.scenario_path}")
    
    # 从 ablation_config 推断配置
    config_mapping = {
        'profile_and_history_and_context': (True, True, True),
        'profile_and_history': (True, True, False),
        'profile_and_context': (True, False, True),
        'history_and_context': (False, True, True),
        'profile_only': (True, False, False),
        'history_only': (False, True, False),
        'context_only': (False, False, True),
    }
    use_profile, use_history, use_context = config_mapping[args.ablation_config]
    
    # 运行推理
    run_inference_vllm(
        checkpoint_dir=args.checkpoint_dir,
        scenario_path=args.scenario_path,
        ablation_config=args.ablation_config,
        use_profile=use_profile,
        use_context=use_context,
        use_history=use_history,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
