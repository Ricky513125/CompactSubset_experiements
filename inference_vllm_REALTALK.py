"""
使用 vLLM 进行高性能推理 - REALTALK 专用版本

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
python inference_vllm_REALTALK.py \
    --checkpoint_dir outputs/REALTALK_8B_context_sampled_seed42 \
    --ablation_config profile_and_context \
    --num_samples 5 \
    --output_dir outputs/leaderboards/REALTALK_vllm_8B \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384
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
    检测文本是否为乱码（针对英文对话）
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
    
    return False


def clean_generated_text(text: str, max_length: int = 512) -> str:
    """
    清洗 REALTALK 生成的文本（专门针对英文对话）
    
    清洗逻辑：
    1. 先提取有效信息（从特殊格式中提取实际内容）
    2. 再逐步清洗（移除标签、指令性文本等）
    3. 最后规范化（去重、格式处理等）
    
    Args:
        text: 原始生成文本
        max_length: 最大输出长度（字符数，默认512）
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 0. 检测并移除乱码
    if is_garbled_text(text):
        return ""
    
    # ============================================
    # 第一步：提取有效信息（从特殊格式中提取实际内容）
    # ============================================
    
    # 1.1. 处理 [OUTPUT] 和 [OUTPUT_SCORE] 格式
    if '[OUTPUT]' in text and '[OUTPUT_SCORE=' in text:
        output_pattern = r'\[OUTPUT\](.*?)\[OUTPUT_SCORE=([\d.]+)\]'
        matches = re.findall(output_pattern, text, re.DOTALL)
        if matches:
            best_output = None
            best_score = -1
            for output_text, score_str in matches:
                try:
                    score = float(score_str)
                    if score > best_score:
                        best_score = score
                        best_output = output_text.strip()
                except:
                    pass
            if best_output:
                text = best_output
            elif matches:
                text = matches[0][0].strip()
    
    # 1.2. 处理 [RESPONSE_1], [RESPONSE_2] 等格式
    if re.search(r'\[RESPONSE_\d+\]', text):
        response_pattern = r'\[RESPONSE_\d+\]\s*(.*?)(?=\[RESPONSE_\d+\]|$)'
        matches = re.findall(response_pattern, text, re.DOTALL)
        if matches:
            text = matches[0].strip()
    
    # 1.3. 处理 JSON 数组格式
    json_array_match = re.search(r'\[\s*\[(.*?)\]\s*\]', text, re.DOTALL)
    if json_array_match:
        inner_array = json_array_match.group(1)
        string_match = re.search(r'"([^"]+)"', inner_array)
        if string_match:
            text = string_match.group(1)
        else:
            parts = [p.strip() for p in inner_array.split(',') if p.strip()]
            if parts:
                text = parts[0].strip('"\'')
    
    # 1.4. 处理 [RESULT] 格式
    if '[RESULT]' in text:
        result_match = re.search(r'\[RESULT\](.*?)(?:\[SYSTEM_LOGIC\]|$)', text, re.DOTALL)
        if result_match:
            result_text = result_match.group(1).strip()
            result_text = re.sub(r'\([^)]*\)', '', result_text)
            text = result_text.strip()
    
    # ============================================
    # 第二步：移除元数据和格式标记
    # ============================================
    
    # 2.1. 移除元数据标签
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 2.2. 移除对话格式标记（User:, Assistant:）
    text = re.sub(r'User:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    
    # 2.3. 移除所有标签（方括号）
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # ============================================
    # 第三步：移除指令性文本
    # ============================================
    
    # 3.1. 移除所有指令性文本模式
    instruction_patterns = [
        r'Predict the user\'s next message[：:]?\s*',
        r'Predict the user\'s response[：:]?\s*',
        r'Note:.*?directly[。.]?\s*',
        r'Note:.*?output[。.]?\s*',
        r'No thinking process.*?needed[。.]?\s*',
        r'Output only.*?reply[。.]?\s*',
        r'Direct output.*?message[。.]?\s*',
        r'思考过程.*?不需要[。.]?\s*',
        r'不需要思考过程[，,]?.*?直接输出[。.]?\s*',
        r'只需直接输出.*?消息[。.]?\s*',
        r'直接输出用户.*?消息[。.]?\s*',
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3.2. 检测并移除思考过程（在截断之前先移除明显的思考模式）
    thinking_patterns = [
        r'Okay,?\s+so\s+',
        r'Now,?\s+the\s+',
        r'Wait,?\s+the\s+',
        r'Looking\s+at\s+',
        r'Alternatively,?\s+',
        r'However,?\s+',
        r'Since\s+the\s+',
        r'Given\s+the\s+',
        r'The\s+user\s+(?:has|had|might|could|would|should)',
        r'user\s+(?:has|had|might|could|would|should)',
        r'assistant\'s\s+(?:last|response|message)',
        r'user\'s\s+(?:next|last|response|message)',
        r'user\s+could\s+say',
        r'user\s+might\s+',
        r'user\s+would\s+',
        r'Maybe\s+they\'ll',
        r'But\s+the\s+user',
    ]
    
    # 移除明显的思考过程句子
    sentences = re.split(r'([.!?]\s+)', text)
    filtered_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sent = (sentences[i] + sentences[i + 1]).strip()
        else:
            sent = sentences[i].strip()
        
        if not sent:
            continue
        
        # 检查是否是思考过程
        is_thinking = False
        sent_lower = sent.lower()
        for pattern in thinking_patterns:
            if re.search(pattern, sent_lower, re.IGNORECASE):
                is_thinking = True
                break
        
        # 如果句子以 "Keep it concise" 或类似指令开头，也移除
        if re.match(r'^(Keep\s+it\s+concise|Be\s+concise|Make\s+it\s+short)', sent, re.IGNORECASE):
            is_thinking = True
        
        if not is_thinking:
            filtered_sentences.append(sent)
    
    if filtered_sentences:
        text = ''.join(filtered_sentences)
    else:
        # 如果所有句子都被过滤掉了，尝试保留第一个非思考句子
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sent = (sentences[i] + sentences[i + 1]).strip()
            else:
                sent = sentences[i].strip()
            if sent and len(sent) > 10:
                # 检查是否包含明显的思考关键词
                has_thinking_keywords = any(
                    keyword in sent.lower() 
                    for keyword in ['okay, so', 'now, the', 'wait,', 'looking at', 'the user', 'user might', 'user could', 'user would', 'assistant\'s']
                )
                if not has_thinking_keywords:
                    text = sent
                    break
    
    # 3.3. 截断到第一个角色标识或提示信息之前
    truncation_patterns = [
        r'\s*Assistant:\s*',
        r'\s*User:\s*',
        r'\s*AI\'s reply[：:]?',
        r'\s*AI\'s response[：:]?',
        r'\s*Predict the user\'s next message[：:]?\s*',
    ]
    
    min_pos = len(text)
    for pattern in truncation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pos = match.start()
            if pos < min_pos:
                min_pos = pos
    
    if min_pos < len(text):
        text = text[:min_pos].strip()
    
    # 3.4. 移除括号注释
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 3.5. 移除开头的 "Keep it concise" 等指令性文本
    text = re.sub(r'^(Keep\s+it\s+concise|Be\s+concise|Make\s+it\s+short)[.!?\s]*', '', text, flags=re.IGNORECASE)
    
    # ============================================
    # 第四步：去重
    # ============================================
    
    # 4.1. 按句子分割并去重
    sentences = re.split(r'([.!?]\s+)', text)
    if len(sentences) > 2:
        seen = set()
        unique_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sent = (sentences[i] + sentences[i + 1]).strip()
            else:
                sent = sentences[i].strip()
            
            if not sent or len(sent) < 3:
                continue
            
            sent_clean = sent.lower().strip()
            
            is_duplicate = False
            for seen_key in seen:
                if sent_clean == seen_key:
                    is_duplicate = True
                    break
                if len(sent_clean) > 10 and len(seen_key) > 10:
                    if sent_clean in seen_key or seen_key in sent_clean:
                        if abs(len(sent_clean) - len(seen_key)) / max(len(sent_clean), len(seen_key)) < 0.2:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                seen.add(sent_clean)
                unique_sentences.append(sent)
        
        if unique_sentences:
            text = ''.join(unique_sentences)
        elif len(sentences) > 2:
            text = (sentences[0] + sentences[1]).strip() if len(sentences) > 1 else sentences[0].strip()
    
    # ============================================
    # 第五步：格式规范化
    # ============================================
    
    # 5.1. 规范化特殊Unicode字符
    text = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2015', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u00A0', ' ').replace('\u3000', ' ')
    
    # 5.2. 移除开头的标点符号
    text = re.sub(r'^\\?"', '', text)
    text = re.sub(r'^\\?\'', '', text)
    text = text.lstrip(r'.!?,\s\-')
    text = re.sub(r'^[.!?,:;\-]\s+', '', text)
    if text.startswith('-'):
        text = text.lstrip('-').lstrip()
    
    # 5.3. 清理重复的标点符号
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    
    # 5.4. 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # ============================================
    # 第六步：最终处理
    # ============================================
    
    # 6.1. 截断过长文本（保留完整的句子）
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
    
    # 6.2. 最终清理空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 6.3. 确保结果不为空
    if not text:
        return ""
    
    return text


def build_inference_prompt(
    user_info: Dict[str, Any],
    use_profile: bool = True,
    use_context: bool = True,
    use_history: bool = False,
    max_context_turns: int = 15
) -> str:
    """
    构建 REALTALK 推理 prompt（与训练时的格式一致）
    """
    parts = []
    
    # 1. 用户画像
    if use_profile and user_info.get('user_profile'):
        profile = user_info['user_profile']
        profile_tags = []
        
        if 'name' in profile:
            profile_tags.append(f"[USER_NAME={profile['name']}]")
        if 'age' in profile:
            profile_tags.append(f"[USER_AGE={profile['age']}]")
        if 'gender' in profile:
            profile_tags.append(f"[USER_GENDER={profile['gender']}]")
        
        # 人格维度
        if 'dimensions' in profile:
            dims = profile['dimensions']
            if isinstance(dims, dict):
                for dim_key, dim_score in dims.items():
                    if isinstance(dim_score, (int, float)):
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
    
    # 2. 任务描述
    parts.append("[TASK]")
    parts.append("Given the historical dialogue of a user on REALTALK, model the user's speaking style and behavioral patterns, and predict the next utterance the user would produce.")
    parts.append("")
    
    # 3. 历史信息
    if use_history and user_info.get('history'):
        history = user_info['history']
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
    
    # 4. 对话上下文
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
            parts.append("")
    
    # 5. 生成提示
    parts.append("Predict the user's next message:")
    
    # 6. 添加输出要求说明（加强版，明确禁止思考过程）
    parts.append("")
    parts.append("IMPORTANT: Output ONLY the user's next message. Do NOT include:")
    parts.append("- Any thinking process, reasoning, or explanation")
    parts.append("- Analysis of the conversation")
    parts.append("- Phrases like 'Okay, so...', 'Now, the user...', 'Wait, the user...', 'Looking at...'")
    parts.append("- Any text starting with 'Assistant:' or 'User:'")
    parts.append("- Any meta-commentary about what the user might say")
    parts.append("")
    parts.append("Output format: Directly write what the user would say next, as if you are the user responding in the conversation.")
    
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
    max_tokens: int = 256,
    repetition_penalty: float = 1.1,
    seed: int = 42,
    max_context_turns: int = 15,  # 新增：context 最大轮次数
    max_chars_per_turn: int = 300  # 新增：每轮对话最大字符数
):
    """
    使用 vLLM 运行推理
    """
    print("=" * 80)
    print("vLLM 推理配置 - REALTALK")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: REALTALK")
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
    
    # 规范化 checkpoint_dir 路径
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
    if checkpoint_dir.startswith('/outputs/'):
        checkpoint_dir = checkpoint_dir.replace('/outputs/', '/mnt/parallel/CompactSubset_experiement/outputs/')
    
    # 检查路径是否存在
    if not os.path.exists(checkpoint_dir):
        print(f"错误: 模型路径不存在: {checkpoint_dir}")
        print(f"请检查 checkpoint_dir 参数是否正确")
        return
    
    print(f"使用模型路径: {checkpoint_dir}")
    
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
    
    # 采样参数（REALTALK 是英文对话，使用中等 temperature 以平衡多样性和稳定性）
    enhanced_temperature = max(temperature, 1.0)  # REALTALK 推荐 1.0-1.2
    actual_repetition_penalty = repetition_penalty if repetition_penalty > 0 else 1.1
    actual_max_tokens = min(max_tokens, 256)  # 限制最大长度以减少思考过程
    
    actual_num_samples = num_samples
    print(f"\n采样参数 (REALTALK 优化):")
    print(f"  temperature: {enhanced_temperature} (平衡多样性和稳定性)")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  repetition_penalty: {actual_repetition_penalty} (增加可减少重复和思考过程)")
    print(f"  max_tokens: {actual_max_tokens} (限制长度以减少思考过程)")
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
                
                # REALTALK: context 为空则跳过
                if not raw_context:
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
                
                # 获取历史证据
                if use_history and user_info.get('user_train_samples'):
                    history_evidence = user_info['user_train_samples'][-3:]  # 使用最近3个样本
                    user_info_with_context['history'] = history_evidence
                else:
                    user_info_with_context['history'] = []
                
                # 为每个 data_item 生成样本
                for sample_idx in range(actual_num_samples):
                    prompt = build_inference_prompt(
                        user_info=user_info_with_context,
                        use_profile=use_profile,
                        use_context=use_context,
                        use_history=use_history,
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
                    
                    # 为每个样本使用不同的seed以增加多样性
                    sample_seed = seed + sample_idx * 1000  # 确保每个样本的seed不同
                    all_sampling_params.append(SamplingParams(
                        temperature=enhanced_temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=actual_repetition_penalty,  # 添加 repetition_penalty 以减少重复和思考过程
                        max_tokens=actual_max_tokens,
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
    
    # 打印前5个示例的输入prompt和原始输出，并保存到文件
    print("\n" + "=" * 80)
    print("示例输入和输出（前5个）")
    print("=" * 80)
    
    # 准备保存到文件的内容
    examples_content = []
    examples_content.append("=" * 80)
    examples_content.append("推理示例：输入 Prompt 和模型原始输出")
    examples_content.append("=" * 80)
    examples_content.append(f"数据集: REALTALK")
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
        
        # 输入 Prompt
        prompt_label = "【输入 Prompt】"
        print(prompt_label)
        examples_content.append(prompt_label)
        
        prompt = all_prompts[i]
        # 打印时如果prompt太长，只显示前800个字符
        if len(prompt) > 800:
            prompt_display = prompt[:800] + "\n... (已截断，总长度: {} 字符)".format(len(prompt))
            print(prompt_display)
            # 保存到文件时保存完整内容
            examples_content.append(prompt)
            examples_content.append(f"\n(总长度: {len(prompt)} 字符)")
        else:
            print(prompt)
            examples_content.append(prompt)
        
        # 模型原始输出
        output_label = "\n【模型原始输出】"
        print(output_label)
        examples_content.append(output_label)
        
        raw_output = generated_texts[i]
        # 打印时如果输出太长，只显示前500个字符
        if len(raw_output) > 500:
            output_display = raw_output[:500] + "\n... (已截断，总长度: {} 字符)".format(len(raw_output))
            print(output_display)
            # 保存到文件时保存完整内容
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
        
        # 使用完整的清洗函数
        cleaned_text = clean_generated_text(generated_text, max_length=512)
        
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
                    # 循环使用已有的样本
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
            'temperature': enhanced_temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': actual_repetition_penalty,
            'max_tokens': actual_max_tokens
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
    parser = argparse.ArgumentParser(description='vLLM 高性能推理 - REALTALK 专用')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型 checkpoint 目录')
    parser.add_argument('--scenario_path', type=str, default=None,
                       help='场景数据路径（默认从 REALTALK 推断）')
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
    
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='采样温度（默认: 1.0，REALTALK 推荐 1.0-1.2）')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p 采样（默认: 0.9）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k 采样（默认: 50）')
    parser.add_argument('--max_tokens', type=int, default=256,
                       help='最大生成 token 数（默认: 256，降低以减少思考过程）')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='重复惩罚（默认: 1.1，增加可减少重复和思考过程）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    parser.add_argument('--max_context_turns', type=int, default=15,
                       help='Context 最大对话轮次数（默认: 15）')
    parser.add_argument('--max_chars_per_turn', type=int, default=300,
                       help='每轮对话最大字符数（默认: 300）')
    
    args = parser.parse_args()
    
    # 推断 scenario_path
    if args.scenario_path is None:
        args.scenario_path = '/mnt/parallel/GIDigitalTwinBench/RealSelf/REALTALK'
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
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_context_turns=args.max_context_turns,
        max_chars_per_turn=args.max_chars_per_turn
    )


if __name__ == '__main__':
    main()
