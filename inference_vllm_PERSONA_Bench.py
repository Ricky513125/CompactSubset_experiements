"""
使用 vLLM 进行高性能推理 - PERSONA-Bench 专用版本

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
python inference_vllm_PERSONA_Bench.py \
    --checkpoint_dir outputs/PERSONA_Bench_history_context_0221_1 \
    --ablation_config history_and_context \
    --num_samples 5 \
    --output_dir outputs/leaderboards/PERSONA_Bench_vllm_8B \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 2048
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


def is_garbled_text(text: str) -> bool:
    """
    检测文本是否为乱码
    """
    if not text or len(text.strip()) < 3:
        return False
    
    text_clean = text.strip()
    digit_count = sum(1 for c in text_clean if c.isdigit())
    letter_count = sum(1 for c in text_clean if c.isalpha())
    total_chars = len(text_clean)
    
    if total_chars == 0:
        return False
    
    # 数字占比过高（>60%）
    digit_ratio = digit_count / total_chars
    if digit_ratio > 0.6:
        return True
    
    # 连续数字过长（>20个连续数字）
    long_digit_sequences = re.findall(r'\d{20,}', text_clean)
    if long_digit_sequences:
        return True
    
    # 几乎没有字母（字母占比<10%且总长度>10）
    if total_chars > 10:
        letter_ratio = letter_count / total_chars
        if letter_ratio < 0.1 and digit_ratio > 0.3:
            return True
    
    # 检测重复模式（如 "30$ USD 30$ USD..." 或 "ANSWER] ANSWER]..."）
    # 检测短字符串（3-50字符）重复3次以上
    if total_chars > 20:
        # 尝试匹配重复的短语（至少3次重复）
        # 匹配模式：短字符串（3-50字符）+ 空格 + 相同字符串，至少重复2次（总共至少3次）
        repeat_pattern = r'(.{3,50}?)(?:\s+\1){2,}'
        if re.search(repeat_pattern, text_clean):
            return True
    
    return False


def _clean_extracted_content(content: str, max_length: int = 300) -> str:
    """
    对提取的内容进行完整清洗（步骤1-6）
    
    Args:
        content: 从模板中提取的原始内容
        max_length: 最大输出长度
    
    Returns:
        清洗后的内容，如果清洗后为空则返回空字符串
    """
    if not content:
        return ""
    
    text = content.strip()
    
    # 步骤1: 删除整行污染
    # 移除残留的 [ANSWER] 和 [/ANSWER] 标签（包括不完整的标签如 [/ANS）
    text = re.sub(r'\[/?ANSWER[^\]]*\]?', '', text).strip()  # 添加 ? 以匹配不完整的标签
    text = re.sub(r'\[/?ANS[^\]]*\]?', '', text).strip()  # 匹配不完整的 [/ANS 等
    
    # 移除元数据标签
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 移除对话格式标记
    text = re.sub(r'User:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    
    # 移除指令性文本模式（英文）
    instruction_patterns = [
        r'Note:.*?',
        r'Do not include.*?',
        r'Only output.*?',
        r'Example format.*?',
        r'Predict.*?next.*?message.*?',
        r'The word.*?should not.*?',
        r'If you are using.*?',
        r'If there is no answer.*?',
        r'ensure you.*?',
        r'correct syntax.*?',
        r'\(without markdown\)',  # 移除 "(without markdown)"
        r'\(with markdown\)',  # 移除 "(with markdown)"
        r'without markdown',  # 移除 "without markdown"
        r'with markdown',  # 移除 "with markdown"
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除所有标签（方括号）
    text = re.sub(r'\[[^\]]*\]', '', text)
    # 移除不完整的标签（如 [/ANS 后面没有 ]）
    text = re.sub(r'\[/?[A-Z]+[^\]]*$', '', text)  # 移除行尾的不完整标签
    
    # 移除分隔线和重复符号
    text = re.sub(r'#+', '', text)  # 移除 # 符号
    text = re.sub(r'-{10,}', '', text)
    text = re.sub(r'[-\s]{20,}', '', text)
    
    # 步骤2-3: 清理重复内容（改进版，PERSONA-Bench 主要是英文）
    # 移除明显的重复模式（多次迭代以确保彻底清理）
    max_iterations = 5
    for _ in range(max_iterations):
        old_text = text
        # 匹配短字符串（3-100字符）重复3次以上（非贪婪匹配）
        text = re.sub(r'(.{3,100}?)(?:\s+\1){2,}', r'\1', text)  # 移除3次以上的重复（带空格分隔）
        text = re.sub(r'(.{3,100}?)\1{2,}', r'\1', text)  # 移除3次以上的重复（无空格分隔）
        if text == old_text:
            break  # 如果没有变化，停止迭代
    
    # 步骤4: Unicode 规范化
    text = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2015', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u00A0', ' ').replace('\u3000', ' ')
    
    # 移除开头的标点符号
    text = re.sub(r'^[.!?,:;\-\s]+', '', text)
    text = text.lstrip()
    
    # 清理重复的标点符号
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)
    
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 移除只有标点符号的文本
    if text and len(text) <= 3 and all(c in '.!?,:;\-\s#' for c in text):
        return ""
    
    # 步骤5: 最终长度限制
    if len(text) > max_length:
        truncated = text[:max_length]
        last_punct = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        if last_punct > max_length * 0.5:
            text = truncated[:last_punct + 1]
        else:
            text = truncated
    
    # 最终清理
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_generated_text(text: str, max_length: int = 300) -> str:
    """
    清洗 PERSONA-Bench 生成的文本（专门针对英文对话）
    
    改进的清洗逻辑：
    1. 先找到所有可能的 [ANSWER]...[/ANSWER] 匹配
    2. 对每个匹配进行完整清洗
    3. 如果清洗后还有有效内容（>= 3字符），就使用它
    4. 如果清洗后为空，尝试下一个匹配
    5. 如果所有匹配都清洗后为空，再考虑其他策略
    
    Args:
        text: 原始生成文本
        max_length: 最大输出长度（字符数，默认300）
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 检测并移除乱码
    if is_garbled_text(text):
        return ""
    
    original_text = text
    
    # ============================================
    # 步骤 0: 优先提取 [ANSWER]...[/ANSWER] 模板
    # 先提取中间的内容，然后再进行清洗和判断
    # ============================================
    
    # 0.1. 尝试匹配完整的 [ANSWER]...[/ANSWER] 对
    if '[ANSWER]' in text and '[/ANSWER]' in text:
        answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
        matches = re.findall(answer_pattern, text, re.DOTALL)
        
        if matches:
            # 对每个匹配进行完整清洗，找到第一个有效的
            for match in matches:
                # 先移除中间的 [ANSWER_FORMAT=...] 等标签
                content = match.strip()
                content = re.sub(r'\[ANSWER[^\]]*\]', '', content).strip()
                
                # 在清洗前先检测是否为乱码或重复内容
                if is_garbled_text(content):
                    continue  # 跳过乱码内容
                
                # 对提取的内容进行完整清洗
                cleaned = _clean_extracted_content(content, max_length)
                
                # 清洗后再次检测是否为乱码
                if cleaned and len(cleaned) >= 3 and not is_garbled_text(cleaned):
                    return cleaned
    
    # 0.2. 如果没有找到 [/ANSWER]，但找到了 [ANSWER]，提取第一个 [ANSWER] 到下一个 [ANSWER] 或文本末尾
    if '[ANSWER]' in text:
        answer_pattern = r'\[ANSWER\](.*?)(?=\[ANSWER\]|$)'
        matches = re.findall(answer_pattern, text, re.DOTALL)
        
        if matches:
            for match in matches:
                content = match.strip()
                content = re.sub(r'\[/?ANSWER[^\]]*\]', '', content).strip()
                
                # 在清洗前先检测是否为乱码或重复内容
                if is_garbled_text(content):
                    continue  # 跳过乱码内容
                
                # 对提取的内容进行完整清洗
                cleaned = _clean_extracted_content(content, max_length)
                
                # 清洗后再次检测是否为乱码
                if cleaned and len(cleaned) >= 3 and not is_garbled_text(cleaned):
                    return cleaned
    
    # 0.3. 如果没有找到 [ANSWER] 标签，对原始文本进行完整清洗
    # 在清洗前先检测是否为乱码
    if is_garbled_text(original_text):
        return ""
    
    # 对原始文本进行完整清洗
    cleaned = _clean_extracted_content(original_text, max_length)
    
    # 清洗后再次检测是否为乱码
    if cleaned and len(cleaned) >= 3 and not is_garbled_text(cleaned):
        return cleaned
    
    # 如果所有清洗后都为空，返回空字符串
    return ""


def build_inference_prompt(
    user_info: Dict[str, Any],
    use_profile: bool = True,
    use_context: bool = True,
    use_history: bool = False,
    max_context_turns: int = 15
) -> str:
    """
    构建 PERSONA-Bench 推理 prompt（与训练时的格式一致）
    
    格式与训练时的 build_simple_training_prompt 保持一致：
    [USER_HASH=...]
    [USER_PROFILE]
    [USER_NAME=...]
    [USER_AGE=...]
    [DIM_OCEAN_EXTRAVERSION=90]
    ...
    [TASK]
    Given the historical dialogue of a user on Reddit, model the user's speaking style and behavioral patterns, and predict the next utterance the user would produce.
    [HISTORY]
    1. ...
    2. ...
    [RECENT_DIALOGUE]
    User: ...
    Assistant: ...
    Predict the assistant's next message:
    """
    parts = []
    
    # 0. USER_HASH 部分 - 始终包含（无论 use_profile 是否启用）
    user_hash = user_info.get('user_hash') or user_info.get('user', {}).get('hash')
    if user_hash:
        parts.append(f"[USER_HASH={user_hash}]")
    
    # 1. USER_PROFILE 部分 - 使用方括号标签格式（由 use_profile 控制）
    if use_profile and user_info.get('user_profile'):
        profile = user_info['user_profile']
        profile_tags = []
        
        # 基础信息标签
        if 'name' in profile and profile['name']:
            profile_tags.append(f"[USER_NAME={profile['name']}]")
        if 'age' in profile and profile['age']:
            profile_tags.append(f"[USER_AGE={profile['age']}]")
        if 'gender' in profile and profile['gender']:
            profile_tags.append(f"[USER_GENDER={profile['gender']}]")
        
        # 心理维度标签（dimensions）
        # 支持两种格式：
        # 1. 扁平化格式: {"Ocean.Extraversion": 90, ...}
        # 2. 嵌套格式: {"BIRI": {"dimensions": {"PerspectiveTaking": {"score": 65}}}}
        if 'dimensions' in profile and isinstance(profile['dimensions'], dict):
            dims = profile['dimensions']
            
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
        for key, value in profile.items():
            if key not in excluded_keys and value:
                # 将其他字段也转为标签格式
                tag_name = f"USER_{key.upper()}"
                profile_tags.append(f"[{tag_name}={value}]")
        
        if profile_tags:
            parts.append("[USER_PROFILE]")
            parts.extend(profile_tags)
    
    # 2. TASK 部分 - 任务描述（PERSONA-Bench 使用英文）
    task_text = "Given the historical dialogue of a user on Reddit, model the user's speaking style and behavioral patterns, and predict the next utterance the user would produce."
    parts.append(f"[TASK]\n{task_text}")
    
    # 2.5. HISTORY 部分 - 历史信息（在 TASK 和 RECENT_DIALOGUE 之间）
    if use_history and user_info.get('history'):
        history = user_info['history']
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
            parts.append("\n".join(history_parts))
    
    # 3. RECENT_DIALOGUE 部分 - 对话上下文
    if use_context:
        context = user_info.get('context', [])
        if context:
            parts.append("[RECENT_DIALOGUE]")
            # 限制 context 长度
            if len(context) > max_context_turns:
                context = context[-max_context_turns:]
            
            for turn in context:
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                # 如果单条内容太长，也截断
                if len(content) > 300:
                    content = content[:297] + "..."
                label = "User" if role == 'user' else "Assistant" if role == 'assistant' else "Unknown"
                parts.append(f"{label}: {content}")
    
    # 4. 生成提示（预测 assistant 的回复，因为目标用户映射为 assistant）
    parts.append("Predict the assistant's next message:")
    
    # 5. 添加输出要求说明（与训练时保持一致，使用 [ANSWER] 标签）
    parts.append("Note: Do not include any thinking process, explanation, or instructions. Only output the assistant's actual next message between [ANSWER] and [/ANSWER] tags.")
    parts.append("Example format: [ANSWER]The assistant's actual response here[/ANSWER]")
    
    return "\n\n".join(parts)


def generate_with_vllm(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams | List[SamplingParams],
    show_progress: bool = True
) -> List[str]:
    """
    使用 vLLM 批量生成
    """
    if show_progress:
        print(f"使用 vLLM 生成 {len(prompts)} 个样本...")
        if isinstance(sampling_params, list):
            print(f"  使用不同的采样参数以增加多样性...")
        # vLLM 0.14.1+ 支持通过 generate 方法的 extra_body 参数传递 chat_template_kwargs
        # 但 Python API 可能不支持，需要通过其他方式控制
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    # 提取生成的文本
    generated_texts = []
    for output in outputs:
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
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_tokens: int = 1024,  # 增加到 1024 以支持更长的回复（训练数据最大长度 2676 字符）
    seed: int = 42,
    max_context_turns: int = 15,
    max_chars_per_turn: int = 300
):
    """
    使用 vLLM 运行推理
    """
    print("=" * 80)
    print("vLLM 推理配置 - PERSONA-Bench")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: PERSONA-Bench")
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
        return
    
    test_leaderboard = load_test_leaderboard(test_leaderboard_path)
    train_data = load_train_data(train_data_path)
    
    print(f"测试集用户数: {len(test_leaderboard)}")
    
    # 初始化 vLLM
    print(f"\n初始化 vLLM (Tensor Parallel={tensor_parallel_size})...")
    start_time = time.time()
    
    # 规范化 checkpoint_dir 路径
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
    if checkpoint_dir.startswith('/outputs/'):
        checkpoint_dir = checkpoint_dir.replace('/outputs/', '/mnt/parallel/CompactSubset_experiement/outputs/')
    
    llm = LLM(
        model=checkpoint_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=seed,
        enforce_eager=False,
        # 注意：chat_template_kwargs 只在 API server 模式下可用
        # 在 Python API 中，思考模式需要通过 prompt 和模型配置控制
    )
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 (耗时: {load_time:.2f}s)")
    
    # 采样参数
    enhanced_temperature = temperature
    enhanced_top_p = top_p
    # 使用传入的 repetition_penalty，如果没有传入则使用默认值 1.1
    actual_repetition_penalty = repetition_penalty if repetition_penalty > 0 else 1.1
    actual_max_tokens = max_tokens
    
    print(f"\n采样参数 (PERSONA-Bench 优化):")
    print(f"  temperature: {enhanced_temperature}")
    print(f"  top_p: {enhanced_top_p}")
    print(f"  top_k: {top_k}")
    print(f"  repetition_penalty: {actual_repetition_penalty}")
    print(f"  max_tokens: {actual_max_tokens}")
    print(f"  base_seed: {seed} (每个样本会使用不同的seed)")
    
    # 准备所有推理请求
    print("\n准备推理请求...")
    all_prompts = []
    all_metadata = []
    all_sampling_params = []
    
    # 遍历 test_leaderboard
    for test_sample_idx, test_sample in enumerate(tqdm(test_leaderboard, desc="构建 prompts")):
        user_info = get_user_info_from_leaderboard(
            sample=test_sample,
            train_data=train_data
        )
        
        if not user_info:
            continue
        
        user_hash = test_sample.get('user_hash', test_sample.get('user', {}).get('hash', 'unknown'))
        task = test_sample.get('task', {})
        collections = task.get('task_behavior_collections', [])
        
        if not collections:
            continue
        
        # 处理每个collection中的data
        for collection_idx, collection in enumerate(collections):
            data_items = collection.get('data', [])
            for data_item_idx, data_item in enumerate(data_items):
                raw_context = data_item.get('context', [])
                
                if not raw_context:
                    continue
                
                # 转换 context 格式
                user_name = user_info.get('user_profile', {}).get('name', '')
                converted_context = []
                
                context_to_process = raw_context[-max_context_turns:] if len(raw_context) > max_context_turns else raw_context
                
                for turn in context_to_process:
                    source = turn.get('source', '')
                    content = turn.get('content', '')
                    
                    if len(content) > max_chars_per_turn:
                        content = content[:max_chars_per_turn-3] + "..."
                    
                    role = 'assistant' if source == user_name else 'user'
                    converted_context.append({
                        'role': role,
                        'content': content
                    })
                
                user_info_with_context = user_info.copy()
                user_info_with_context['context'] = converted_context
                
                # 获取历史证据
                if use_history and user_info.get('user_train_samples'):
                    history_evidence = user_info['user_train_samples'][-3:]
                    user_info_with_context['history'] = history_evidence
                else:
                    user_info_with_context['history'] = []
                
                # 为每个 data_item 生成样本
                for sample_idx in range(num_samples):
                    prompt = build_inference_prompt(
                        user_info=user_info_with_context,
                        use_profile=use_profile,
                        use_context=use_context,
                        use_history=use_history,
                        max_context_turns=max_context_turns
                    )
                    
                    all_prompts.append(prompt)
                    all_metadata.append({
                        'test_sample_idx': test_sample_idx,
                        'collection_idx': collection_idx,
                        'data_item_idx': data_item_idx,
                        'sample_idx': sample_idx
                    })
                    
                    # 为每个样本使用不同的seed
                    sample_seed = seed + sample_idx 
                    all_sampling_params.append(SamplingParams(
                        temperature=enhanced_temperature,
                        top_p=enhanced_top_p,
                        top_k=top_k,
                        repetition_penalty=actual_repetition_penalty,  # 添加 repetition_penalty 以减少重复和乱码
                        max_tokens=actual_max_tokens,
                        seed=sample_seed,
                        skip_special_tokens=True,
                        # 注意：当前版本的 vLLM 不支持 enable_thinking 参数
                        # 思考模式已通过 prompt 中的指令控制（"注意：思考過程や説明は不要です"）
                    ))
    
    print(f"总推理请求数: {len(all_prompts)}")
    
    # 批量推理
    print("\n开始批量推理...")
    inference_start = time.time()
    
    generated_texts = generate_with_vllm(
        llm=llm,
        prompts=all_prompts,
        sampling_params=all_sampling_params,
        show_progress=True
    )
    
    inference_time = time.time() - inference_start
    throughput = len(all_prompts) / inference_time
    
    print(f"\n✓ 推理完成")
    print(f"  总样本数: {len(all_prompts)}")
    print(f"  推理时间: {inference_time:.2f}s")
    print(f"  吞吐量: {throughput:.2f} samples/sec ({throughput * 60:.0f} samples/min)")
    
    # 打印前5个示例
    print("\n" + "=" * 80)
    print("示例输入和输出（前5个）")
    print("=" * 80)
    
    examples_content = []
    examples_content.append("=" * 80)
    examples_content.append("推理示例：输入 Prompt 和模型原始输出")
    examples_content.append("=" * 80)
    examples_content.append(f"数据集: PERSONA-Bench")
    examples_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    examples_content.append(f"总样本数: {len(all_prompts)}")
    examples_content.append("=" * 80)
    examples_content.append("")
    
    num_examples = min(5, len(all_prompts))
    for i in range(num_examples):
        example_header = f"\n【示例 {i+1}/{num_examples}】"
        print(example_header)
        examples_content.append(example_header)
        
        separator = "-" * 80
        print(separator)
        examples_content.append(separator)
        
        prompt_label = "【输入 Prompt】"
        print(prompt_label)
        examples_content.append(prompt_label)
        
        prompt = all_prompts[i]
        if len(prompt) > 800:
            prompt_display = prompt[:800] + "\n... (已截断，总长度: {} 字符)".format(len(prompt))
            print(prompt_display)
            examples_content.append(prompt)
            examples_content.append(f"\n(总长度: {len(prompt)} 字符)")
        else:
            print(prompt)
            examples_content.append(prompt)
        
        output_label = "\n【模型原始输出】"
        print(output_label)
        examples_content.append(output_label)
        
        raw_output = generated_texts[i]
        if len(raw_output) > 500:
            output_display = raw_output[:500] + "\n... (已截断，总长度: {} 字符)".format(len(raw_output))
            print(output_display)
            examples_content.append(raw_output)
            examples_content.append(f"\n(总长度: {len(raw_output)} 字符)")
        else:
            print(raw_output)
            examples_content.append(raw_output)
        
        print(separator)
        examples_content.append(separator)
    
    print("=" * 80)
    examples_content.append("\n" + "=" * 80)
    
    # 保存到文件
    examples_file = os.path.join(output_dir, 'inference_examples.txt')
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(examples_content))
    
    print(f"\n✓ 示例已保存到: {examples_file}")
    
    # 清洗生成的文本并处理空结果（包括重试机制）
    print(f"\n清洗生成的文本...")
    # 按 (test_sample_idx, collection_idx, data_item_idx, sample_idx) 分组
    data_item_samples = {}  # key: (test_sample_idx, collection_idx, data_item_idx), value: dict of {sample_idx: cleaned_text}
    data_item_prompts = {}  # key: (test_sample_idx, collection_idx, data_item_idx), value: prompt
    data_item_metadata = {}  # key: (test_sample_idx, collection_idx, data_item_idx), value: metadata dict
    
    cleaned_count = 0
    empty_count = 0
    retry_count = 0
    
    # 保存原始输出和清洗后的结果
    processing_log_content = []
    processing_log_content.append("=" * 80)
    processing_log_content.append("推理阶段原始输出和处理结果")
    processing_log_content.append("=" * 80)
    processing_log_content.append("")
    
    # 第一轮：按 data_item 和 sample_idx 分组处理
    for idx, (metadata, generated_text) in enumerate(zip(all_metadata, generated_texts)):
        key = (metadata['test_sample_idx'], metadata['collection_idx'], metadata['data_item_idx'])
        sample_idx = metadata['sample_idx']
        
        if key not in data_item_samples:
            data_item_samples[key] = {}
            # 保存 prompt 和 metadata 以便重试（所有 sample_idx 共享同一个 prompt）
            # 找到第一个 sample_idx=0 的 prompt 作为该 data_item 的 prompt
            prompt_idx = idx - sample_idx  # 回退到 sample_idx=0 的位置
            data_item_prompts[key] = all_prompts[prompt_idx]
            data_item_metadata[key] = metadata
        
        cleaned_text = clean_generated_text(generated_text, max_length=300)
        
        if cleaned_text != generated_text.strip():
            cleaned_count += 1
        
        # 保存原始输出和清洗后的结果
        test_sample = test_leaderboard[metadata['test_sample_idx']]
        collection = test_sample['task']['task_behavior_collections'][metadata['collection_idx']]
        data_item = collection['data'][metadata['data_item_idx']]
        
        processing_log_content.append(f"样本 #{idx + 1}")
        processing_log_content.append(f"Data Item: test_sample_idx={metadata['test_sample_idx']}, collection_idx={metadata['collection_idx']}, data_item_idx={metadata['data_item_idx']}, sample_idx={sample_idx}")
        processing_log_content.append("-" * 80)
        processing_log_content.append("【原始输出】")
        processing_log_content.append(generated_text)
        processing_log_content.append("")
        processing_log_content.append("【清洗后结果】")
        if cleaned_text:
            processing_log_content.append(cleaned_text)
        else:
            processing_log_content.append("[清洗后为空]")
        processing_log_content.append("")
        processing_log_content.append("=" * 80)
        processing_log_content.append("")
        
        if cleaned_text:
            # 对于每个 sample_idx，只保存第一个有效回答
            if sample_idx not in data_item_samples[key]:
                data_item_samples[key][sample_idx] = cleaned_text
        else:
            empty_count += 1
            # 如果这个 sample_idx 还没有有效回答，标记为 None（稍后重试）
            if sample_idx not in data_item_samples[key]:
                data_item_samples[key][sample_idx] = None
    
    # 第二轮：对清洗后为空的样本进行重试（调高 temperature，批量处理）
    print(f"\n处理空结果：对 {empty_count} 个空样本进行重试...")
    max_retries = 3
    retry_temperature_increment = 0.3
    
    # 按轮次重试：每轮只对仍然失败的样本进行重试
    for retry_attempt in range(max_retries):
        # 收集当前轮次需要重试的样本
        retry_items = []  # [(key, sample_idx, prompt, retry_params), ...]
        
        for key, samples_dict in data_item_samples.items():
            prompt = data_item_prompts[key]
            
            # 对每个仍然为空的 sample_idx 进行重试
            for sample_idx in range(num_samples):
                if sample_idx in samples_dict and samples_dict[sample_idx] is None:
                    retry_temperature = enhanced_temperature + (retry_attempt + 1) * retry_temperature_increment
                    retry_seed = seed + sample_idx + (retry_attempt + 1) * 1000
                    
                    retry_params = SamplingParams(
                        temperature=retry_temperature,
                        top_p=enhanced_top_p,
                        top_k=top_k,
                        repetition_penalty=actual_repetition_penalty,
                        max_tokens=actual_max_tokens,
                        seed=retry_seed,
                        skip_special_tokens=True,
                    )
                    
                    retry_items.append((key, sample_idx, prompt, retry_params))
        
        if not retry_items:
            print(f"  所有样本已成功，停止重试")
            break
        
        print(f"  第 {retry_attempt + 1} 轮重试: {len(retry_items)} 个样本 (temperature={enhanced_temperature + (retry_attempt + 1) * retry_temperature_increment:.2f})...")
        
        # 分批处理，每批最多1000个（避免内存问题）
        batch_size = 1000
        total_batches = (len(retry_items) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(retry_items))
            batch_items = retry_items[start_idx:end_idx]
            
            if total_batches > 1:
                print(f"    处理批次 {batch_idx + 1}/{total_batches} ({len(batch_items)} 个样本)...")
            
            # 提取prompts和params
            batch_prompts = [item[2] for item in batch_items]
            batch_params = [item[3] for item in batch_items]
            
            # 批量生成
            batch_outputs = llm.generate(batch_prompts, batch_params, use_tqdm=(total_batches == 1))
            
            # 处理结果
            for item, output in zip(batch_items, batch_outputs):
                key, sample_idx, prompt, retry_params = item
                retry_text = output.outputs[0].text
                
                # 清洗重试生成的文本
                retry_cleaned = clean_generated_text(retry_text, max_length=300)
                
                # 检查是否已经成功（可能在其他批次中已经成功）
                samples_dict = data_item_samples[key]
                if sample_idx in samples_dict and samples_dict[sample_idx] is not None:
                    continue  # 已经成功，跳过
                
                if retry_cleaned and len(retry_cleaned) >= 3:
                    # 重试成功，保存结果
                    samples_dict[sample_idx] = retry_cleaned
                    retry_count += 1
                    processing_log_content.append(f"重试成功: Data Item key={key}, sample_idx={sample_idx}, retry_attempt={retry_attempt+1}")
                    processing_log_content.append(f"  temperature={retry_params.temperature}, seed={retry_params.seed}")
                    processing_log_content.append(f"  【重试原始输出】{retry_text[:200]}...")
                    processing_log_content.append(f"  【重试清洗后】{retry_cleaned}")
                    processing_log_content.append("")
                else:
                    # 记录重试失败（只在最后一次尝试时记录，避免日志过多）
                    if retry_attempt == max_retries - 1:
                        processing_log_content.append(f"重试失败: Data Item key={key}, sample_idx={sample_idx}, retry_attempt={retry_attempt+1}")
                        processing_log_content.append(f"  temperature={retry_params.temperature}, seed={retry_params.seed}")
                        processing_log_content.append(f"  【重试原始输出】{retry_text[:200]}...")
                        processing_log_content.append(f"  【重试清洗后】[仍为空]")
                        processing_log_content.append("")
        
        # 统计当前轮次后的成功数
        current_success = sum(1 for samples_dict in data_item_samples.values() 
                            for sample_idx in range(num_samples) 
                            if sample_idx in samples_dict and samples_dict[sample_idx] is not None)
        print(f"  第 {retry_attempt + 1} 轮完成，当前成功样本数: {current_success}")
    
    if retry_count > 0:
        print(f"✓ 重试成功: {retry_count} 个样本通过重试获得有效回答")
    
    if cleaned_count > 0:
        print(f"✓ 已清洗 {cleaned_count} 个生成样本")
    
    # 保存原始输出和清洗后的结果到文件
    processing_log_file = os.path.join(output_dir, 'inference_processing_log.txt')
    with open(processing_log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processing_log_content))
    print(f"✓ 原始输出和处理结果已保存到: {processing_log_file}")
    
    # 填充到 test_leaderboard
    # 对于每个 data_item，按 sample_idx 顺序填充 continuations
    print(f"\n填充 continuations 到 test_leaderboard...")
    filled_count = 0
    copied_count = 0
    
    for key, samples_dict in data_item_samples.items():
        test_sample_idx, collection_idx, data_item_idx = key
        test_sample = test_leaderboard[test_sample_idx]
        collection = test_sample['task']['task_behavior_collections'][collection_idx]
        data_item = collection['data'][data_item_idx]
        
        # 先收集所有有效的回答
        valid_answers = []
        for sample_idx in range(num_samples):
            if sample_idx in samples_dict and samples_dict[sample_idx]:
                valid_answers.append(samples_dict[sample_idx])
        
        # 按 sample_idx 顺序（0, 1, 2, 3, 4）填充 continuations
        final_continuations = []
        for sample_idx in range(num_samples):
            if sample_idx in samples_dict and samples_dict[sample_idx]:
                # 使用对应 sample_idx 的有效回答
                final_continuations.append(samples_dict[sample_idx])
            else:
                # 如果该 sample_idx 没有有效回答，从其他有效的回答中复制一个
                if valid_answers:
                    # 循环使用有效回答
                    copy_idx = len(final_continuations) % len(valid_answers)
                    final_continuations.append(valid_answers[copy_idx])
                    copied_count += 1
                else:
                    # 如果所有 sample_idx 都没有有效回答，从其他 data_item 复制（不应该发生，但作为最后手段）
                    # 实际上，如果所有重试都失败，我们应该至少有一个有效回答（从其他样本复制）
                    # 这里不应该出现，但为了安全起见，我们使用一个默认值
                    final_continuations.append("I don't know what to say.")
                    copied_count += 1
        
        data_item['continuations'] = final_continuations
        filled_count += 1
    
    if copied_count > 0:
        print(f"  信息: {copied_count} 个空样本已从其他有效回答中复制（确保没有 [生成失败] 标记）")
    
    print(f"✓ 已填充 {filled_count} 个 data_item 的 continuations")
    
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
            'temperature': enhanced_temperature,
            'top_p': enhanced_top_p,
            'top_k': top_k,
            'repetition_penalty': actual_repetition_penalty,
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
    parser = argparse.ArgumentParser(description='vLLM 高性能推理 - PERSONA-Bench 专用')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型 checkpoint 目录')
    parser.add_argument('--scenario_path', type=str, default=None,
                       help='场景数据路径（默认从 dataset 推断）')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 
                               'profile_and_context', 'history_and_context', 
                               'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个用户生成的样本数（默认: 5）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor Parallel 大小（使用多少张 GPU，默认: 1）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU 内存利用率（0.0-1.0，默认: 0.9）')
    parser.add_argument('--max_model_len', type=int, default=8192,
                       help='最大模型序列长度（默认: 8192）')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='采样温度（默认: 0.7）')
    parser.add_argument('--top_p', type=float, default=0.85,
                       help='Top-p 采样（默认: 0.85）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k 采样（默认: 50）')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='重复惩罚（默认: 1.1）')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='最大生成 token 数（默认: 1024，PERSONA-Bench 推荐 512-1024 以生成更长回复）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    parser.add_argument('--max_context_turns', type=int, default=15,
                       help='Context 最大对话轮次数（默认: 15）')
    parser.add_argument('--max_chars_per_turn', type=int, default=300,
                       help='每轮对话最大字符数（默认: 300）')
    
    args = parser.parse_args()
    
    # 推断 scenario_path
    if args.scenario_path is None:
        args.scenario_path = '/mnt/parallel/GIDigitalTwinBench/RealSelf/PERSONA-Bench'
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
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_context_turns=args.max_context_turns,
        max_chars_per_turn=args.max_chars_per_turn
    )


if __name__ == '__main__':
    main()
