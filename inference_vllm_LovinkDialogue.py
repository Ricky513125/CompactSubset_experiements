"""
使用 vLLM 进行高性能推理 - LovinkDialogue 专用版本

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
python inference_vllm_LovinkDialogue.py \
    --checkpoint_dir outputs/LovinkDialogue_8B_context_sampled_seed42 \
    --ablation_config context_only \
    --num_samples 5 \
    --output_dir outputs/leaderboards/LovinkDialogue_vllm_8B \
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
    
    return False


def extract_and_clean_answer_blocks(text: str, max_length: int = 512) -> str:
    """
    从文本中提取所有 [ANSWER]...[/ANSWER] 块或 [/ANSWER]...[/ANSWER] 之间的内容，
    逐个清洗，返回第一个有效的（非指令性文本）
    
    Args:
        text: 原始生成文本
        max_length: 最大输出长度
    
    Returns:
        第一个有效的清洗后文本，如果所有块都无效则返回空字符串
    """
    if not text:
        return ""
    
    # 找到所有 [/ANSWER] 标签的位置
    answer_end_positions = [m.end() for m in re.finditer(r'\[/ANSWER\]', text, re.IGNORECASE)]
    
    matches = []
    
    if len(answer_end_positions) >= 2:
        # 有至少两个 [/ANSWER] 标签，提取每两个 [/ANSWER] 标签之间的内容
        # 直接跳过第一个块（第一个 [/ANSWER] 之前的内容），从第二个块开始
        for i in range(len(answer_end_positions) - 1):
            start_pos = answer_end_positions[i]
            end_pos = answer_end_positions[i + 1]
            block = text[start_pos:end_pos].strip()
            # 移除可能的残留标签
            block = re.sub(r'\[/?ANSWER[^\]]*\]', '', block).strip()
            if block and len(block) >= 3:
                matches.append(block)
    elif len(answer_end_positions) == 1:
        # 只有一个 [/ANSWER] 标签，尝试提取完整的 [ANSWER]...[/ANSWER] 块
        # 但如果有多个块，跳过第一个
        answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
        full_matches = re.findall(answer_pattern, text, re.DOTALL)
        if len(full_matches) > 1:
            # 跳过第一个，从第二个开始
            matches = full_matches[1:]
        # 如果只有一个块，直接返回空（跳过第一个块）
    
    if not matches:
        return ""
    
    # 对每个块进行清洗，找到第一个有效的（非指令性文本）
    for match in matches:
        candidate = match.strip()
        if not candidate:
            continue
        
        # 检查是否包含换行符或空格分隔的指令性文本和实际内容
        # 如果包含空格分割，优先保留空格后面的内容
        if ' ' in candidate and len(candidate.split()) > 1:
            parts = candidate.split(' ', 1)  # 只分割第一个空格
            if len(parts) == 2:
                first_part = parts[0].strip()
                second_part = parts[1].strip()
                
                # 检查第一部分是否是指令性文本
                first_part_lower = first_part.lower()
                is_first_part_instruction = any(
                    first_part_lower.startswith(phrase) or phrase in first_part_lower
                    for phrase in [
                        '当没有更多内容时', '请回复空字符串',
                        '你的输出应该是不带任何', '不带任何markdown格式', '不要使用markdown格式',
                        '没有特殊要求时', '不要输出其他任何内容',
                        '当用户的问题需要进一步澄清时', '请提出问题', '当用户分享信息时', '请进行简短回应',
                        '并且不要输出任何其他多余的内容',
                        '你的回答要求', '并且不要使用任何markdown格式',
                        '记得保持口语化', '不要用Markdown格式', '不要使用任何markdown格式',
                        '并且避免使用任何格式', '并且保证输出格式正确', '避免使用任何格式',
                        '没有内容时输出空行', '如果需要随机输出',
                        '下面是示例', '你的输出应该严格符合这个格式', '你的回答将被用于训练AI模型',
                        '需要严格遵循格式要求', '没有特别说明的都默认是中文语境'
                    ]
                ) or any(
                    first_part.endswith(punct) for punct in ['。', '.', '：', ':']
                )
                
                # 如果第一部分是指令性文本，且第二部分有实际内容，使用第二部分
                if is_first_part_instruction and second_part and len(second_part.strip()) >= 1:
                    # 检查第二部分是否包含实际对话内容（不是纯指令）
                    has_dialogue_features = any(
                        char in second_part for char in ['?', '？', '!', '！']
                    ) or any(
                        second_part.startswith(starter) for starter in 
                        ['如果', '你', '我', '他', '她', '它', '这', '那', '什么', '怎么', '为什么', '如何', '要是', '假如', '你会', '你会同时', '您', '喜欢', '觉得', '认为', '哦', 'OK', 'OK', '拜拜', 'OK']
                    ) or len(second_part.strip()) <= 20  # 短文本更可能是实际内容
                    
                    if has_dialogue_features:
                        candidate = second_part
                    # 如果第二部分也看起来像指令，继续处理整个文本
                # 如果第一部分不是指令性文本，保持原样继续处理
        
        # 先尝试移除开头的指令性文本（即使没有换行符）
        instruction_patterns_at_start = [
            r'^当没有更多内容时.*?请回复空字符串[。.]\s*',
            r'^你的输出应该是不带任何markdown格式的纯文本[。.]\s*',
            r'^没有特殊要求时.*?不要使用markdown格式[。.]\s*',
            r'^不要输出其他任何内容[。.]\s*',
            r'^当用户的问题需要进一步澄清时.*?请进行简短回应[。.]\s*',
            r'^并且不要输出任何其他多余的内容[。.]\s*',
            r'^当没有更多内容时.*?',
            r'^请回复空字符串[。.]\s*',
            r'^你的输出应该是不带任何.*?格式[。.]\s*',
            r'^不带任何markdown格式[。.]\s*',
            r'^不要使用markdown格式[。.]\s*',
            r'^不要输出其他任何内容[。.]\s*',
            r'^当用户的问题需要进一步澄清时.*?',
            r'^请提出问题[；;].*?',
            r'^当用户分享信息时.*?请进行简短回应[。.]\s*',
            r'^并且不要输出任何其他多余的内容[。.]\s*',
            r'^你的回答要求[：:]\s*',
            r'^并且不要使用任何markdown格式[。.]\s*',
            r'^记得保持口语化.*?不要用Markdown格式[。.]\s*',
            r'^记得保持口语化[，,]\s*',
            r'^不要用Markdown格式[。.]\s*',
            r'^不要使用任何markdown格式[。.]\s*',
            r'^并且避免使用任何格式[。.]\s*',
            r'^并且保证输出格式正确[。.]\s*',
            r'^避免使用任何格式[。.]\s*',
            r'^没有内容时输出空行[。.]\s*',
            r'^如果需要随机输出.*?',
            r'^下面是示例[：:]\s*',
            r'^你的输出应该严格符合这个格式[。.]\s*',
            r'^你的回答将被用于训练AI模型.*?需要严格遵循格式要求[。.]\s*',
            r'^没有特别说明的都默认是中文语境[。.]\s*',
        ]
        for pattern in instruction_patterns_at_start:
            candidate = re.sub(pattern, '', candidate, flags=re.IGNORECASE | re.DOTALL)
        candidate = candidate.strip()
        
        # 检查是否包含换行符分隔的指令性文本和实际内容
        # 如果包含换行符，尝试分割并只保留后面的实际内容
        if '\n' in candidate:
            lines = candidate.split('\n')
            # 检查每一行，找到第一个非指令性文本的行
            actual_content_start = None
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                # 检查这一行是否是指令性文本
                line_lower = line_stripped.lower()
                is_instruction = any(
                    line_lower.startswith(phrase) or phrase in line_lower
                    for phrase in [
                        '您需要', '需要以', '以最自然', '口语化', '避免使用', 'markdown', '格式',
                        '当你的回答需要调用外部知识库',
                        '请使用', '标注引用来源',
                        '你的输出应该是用户最有可能输入',
                        '你的回答将直接用于训练模型',
                        '并且符合其语言风格',
                        '当用户的消息与你的知识库无关',
                        '你的回答将直接返回给用户',
                        '你的输出应该仅包含',
                        '不要输出任何多余的内容',
                        '输出应该仅包含', '应该仅包含', '仅包含',
                        '并且不要输出', '不要输出任何',
                        '你的输出必须符合', '必须符合以下格式',
                        '符合以下格式', '避免使用任何格式化',
                        '保持纯文本输出', '包括文本格式说明',
                        '当你的回答是中文时', '请使用和标签包裹', '的一句话', '并且符合他的语言风格',
                        '当你的回答以问句结尾时', '结束语句', '其他情况请使用',
                        '当没有更多内容时', '请回复空字符串',
                        '你的输出应该是不带任何', '不带任何markdown格式', '不要使用markdown格式',
                        '没有特殊要求时', '不要输出其他任何内容',
                        '当用户的问题需要进一步澄清时', '请提出问题', '当用户分享信息时', '请进行简短回应',
                        '并且不要输出任何其他多余的内容',
                        '你的回答要求', '并且不要使用任何markdown格式',
                        '记得保持口语化', '不要用Markdown格式', '不要使用任何markdown格式',
                        '并且避免使用任何格式'
                    ]
                )
                
                # 如果这一行不是指令性文本，且包含实际对话内容，从这里开始
                if not is_instruction:
                    # 检查是否包含对话特征（问号、感叹号、对话开头词等）
                    has_dialogue_features = any(
                        char in line_stripped for char in ['?', '？', '!', '！']
                    ) or any(
                        line_stripped.startswith(starter) for starter in 
                        ['如果', '你', '我', '他', '她', '它', '这', '那', '什么', '怎么', '为什么', '如何', '要是', '假如', '你会', '你会同时', '您', '喜欢', '觉得', '认为']
                    )
                    
                    if has_dialogue_features:
                        actual_content_start = i
                        break
            
            # 如果找到了实际内容的起始位置，只保留从那里开始的内容
            if actual_content_start is not None:
                candidate = '\n'.join(lines[actual_content_start:]).strip()
            else:
                # 如果没有找到，尝试移除所有看起来像指令的行
                filtered_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    line_lower = line_stripped.lower()
                    is_instruction = any(
                        line_lower.startswith(phrase) or phrase in line_lower
                        for phrase in [
                            '您需要', '需要以', '以最自然', '口语化', '避免使用', 'markdown', '格式',
                            '当你的回答需要调用外部知识库', '请使用', '标注引用来源',
                            '你的输出应该是用户最有可能输入', '你的回答将直接用于训练模型',
                            '并且符合其语言风格', '当用户的消息与你的知识库无关',
                            '你的回答将直接返回给用户', '你的输出应该仅包含',
                            '不要输出任何多余的内容', '输出应该仅包含', '应该仅包含', '仅包含',
                            '并且不要输出', '不要输出任何', '你的输出必须符合', '必须符合以下格式',
                            '符合以下格式', '避免使用任何格式化', '保持纯文本输出', '包括文本格式说明',
                            '当你的回答是中文时', '请使用和标签包裹', '的一句话', '并且符合他的语言风格',
                            '当你的回答以问句结尾时', '结束语句', '其他情况请使用',
                            '当没有更多内容时', '请回复空字符串',
                            '你的输出应该是不带任何', '不带任何markdown格式', '不要使用markdown格式',
                            '没有特殊要求时', '不要输出其他任何内容',
                            '当用户的问题需要进一步澄清时', '请提出问题', '当用户分享信息时', '请进行简短回应',
                            '并且不要输出任何其他多余的内容'
                        ]
                    )
                    if not is_instruction:
                        filtered_lines.append(line)
                
                if filtered_lines:
                    candidate = '\n'.join(filtered_lines).strip()
                else:
                    # 如果所有行都是指令，跳过这个块
                    continue
        
        # 先进行初步清洗（移除明显的指令性文本）
        # 移除常见的指令性文本开头（使用更精确的模式）
        instruction_prefixes = [
            r'^当你的回答需要调用外部知识库时[，,].*?请使用.*?标注引用来源.*?',
            r'^当你的回答需要调用外部知识库时.*?',
            r'^请使用.*?标注引用来源.*?',
            r'^你的输出应该是用户最有可能输入.*?',
            r'^你的回答将直接用于训练模型.*?',
            r'^并且符合其语言风格.*?',
            r'^当用户的消息与你的知识库无关时.*?',
            r'^你的回答将直接返回给用户.*?',
            r'^你的输出应该仅包含.*?',
            r'^不要输出任何多余的内容.*?',
            r'^输出应该仅包含.*?',
            r'^应该仅包含.*?',
            r'^仅包含.*?',
            r'^并且不要输出.*?',
            r'^不要输出任何.*?',
            r'^你的输出必须符合.*?',
            r'^必须符合以下格式.*?',
            r'^符合以下格式.*?',
            r'^避免使用任何格式化.*?',
            r'^保持纯文本输出.*?',
            r'^包括文本格式说明.*?',
            r'^当你的回答是中文时.*?',
            r'^请使用和标签包裹.*?',
            r'^的一句话.*?',
            r'^当你的回答以问句结尾时.*?',
            r'^请使用.*?结束语句.*?',
            r'^其他情况请使用.*?',
            r'^当没有更多内容时.*?请回复空字符串[。.]\s*',
            r'^你的输出应该是不带任何markdown格式的纯文本[。.]\s*',
            r'^没有特殊要求时.*?不要使用markdown格式[。.]\s*',
            r'^不要输出其他任何内容[。.]\s*',
            r'^当用户的问题需要进一步澄清时.*?请进行简短回应[。.]\s*',
            r'^并且不要输出任何其他多余的内容[。.]\s*',
            r'^当没有更多内容时.*?',
            r'^请回复空字符串[。.]\s*',
            r'^你的输出应该是不带任何.*?格式[。.]\s*',
            r'^不带任何markdown格式[。.]\s*',
            r'^不要使用markdown格式[。.]\s*',
            r'^当用户的问题需要进一步澄清时.*?',
            r'^请提出问题[；;].*?',
            r'^当用户分享信息时.*?请进行简短回应[。.]\s*',
            r'^你的回答要求[：:]\s*',
            r'^并且不要使用任何markdown格式[。.]\s*',
            r'^记得保持口语化.*?不要用Markdown格式[。.]\s*',
            r'^记得保持口语化[，,]\s*',
            r'^不要用Markdown格式[。.]\s*',
            r'^不要使用任何markdown格式[。.]\s*',
        ]
        for pattern in instruction_prefixes:
            candidate = re.sub(pattern, '', candidate, flags=re.IGNORECASE | re.DOTALL)
        candidate = candidate.strip()
        
        # 如果清洗后为空或很短，跳过这个块
        if not candidate or len(candidate.strip()) < 3:
            continue
        
        # 检查是否整个内容都是指令性文本（在完整清洗之前）
        candidate_lower = candidate.lower().strip()
        instruction_phrases = [
            '当你的回答需要调用外部知识库',
            '请使用', '标注引用来源',
            '你的输出应该是用户最有可能输入',
            '你的回答将直接用于训练模型',
            '并且符合其语言风格',
            '当用户的消息与你的知识库无关',
            '你的回答将直接返回给用户',
            '你的输出应该仅包含',
            '不要输出任何多余的内容',
            '输出应该仅包含', '应该仅包含', '仅包含',
            '并且不要输出', '不要输出任何',
            '你的输出必须符合', '必须符合以下格式',
            '符合以下格式', '避免使用任何格式化',
            '保持纯文本输出', '包括文本格式说明',
            '当你的回答是中文时', '请使用和标签包裹', '的一句话', '并且符合他的语言风格',
            '当你的回答以问句结尾时', '请使用', '结束语句', '其他情况请使用',
            '当没有更多内容时', '请回复空字符串',
            '你的输出应该是不带任何', '不带任何markdown格式', '不要使用markdown格式',
            '没有特殊要求时', '不要输出其他任何内容',
            '当用户的问题需要进一步澄清时', '请提出问题', '当用户分享信息时', '请进行简短回应',
            '并且不要输出任何其他多余的内容',
            '你的回答要求', '并且不要使用任何markdown格式',
            '记得保持口语化', '不要用Markdown格式', '不要使用任何markdown格式',
            '并且避免使用任何格式'
        ]
        
        # 检查是否以指令性文本开头
        starts_with_instruction = any(candidate_lower.startswith(phrase) for phrase in instruction_phrases)
        
        # 检查是否包含指令性文本（但可能混合了实际内容）
        contains_instruction = any(phrase in candidate_lower for phrase in instruction_phrases)
        
        # 如果以指令性文本开头，需要更严格地检查
        if starts_with_instruction:
            # 尝试移除开头的指令性文本，提取实际内容
            temp_candidate = candidate
            for phrase in instruction_phrases:
                if temp_candidate.lower().startswith(phrase):
                    # 移除匹配的指令性文本
                    temp_candidate = temp_candidate[len(phrase):].strip()
                    # 移除可能的分隔符（句号、逗号、空格等）
                    temp_candidate = re.sub(r'^[，,。.\s]+', '', temp_candidate)
                    break
            
            # 检查移除指令性文本后是否还有实际内容
            dialogue_starters = ['如果', '你', '我', '他', '她', '它', '这', '那', '什么', '怎么', '为什么', '如何', '要是', '假如', '你会', '你会同时']
            has_actual_content = False
            
            if temp_candidate and len(temp_candidate.strip()) >= 3:
                # 检查是否以对话开头
                if any(temp_candidate.startswith(s) for s in dialogue_starters):
                    has_actual_content = True
                # 或者包含问号、感叹号，可能是实际对话
                elif any(char in temp_candidate for char in ['?', '？', '!', '！']):
                    # 检查是否包含对话开头
                    for starter in dialogue_starters:
                        if starter in temp_candidate and temp_candidate.find(starter) < len(temp_candidate) * 0.5:
                            has_actual_content = True
                            break
            
            # 如果有实际内容，使用移除指令性文本后的版本
            if has_actual_content:
                candidate = temp_candidate
            else:
                # 如果以指令性文本开头且没有实际对话内容，跳过
                continue  # 跳过这个块，尝试下一个
        
        # 如果包含指令性文本但不以它开头，可能是混合内容，需要进一步处理
        elif contains_instruction and len(candidate_lower) < 50:
            # 短文本且包含指令性文本，很可能是纯指令性文本
            has_actual_content = any(char in candidate for char in ['?', '？', '!', '！'])
            if not has_actual_content:
                continue  # 跳过这个块，尝试下一个
        
        # 对候选文本进行完整清洗
        cleaned = clean_generated_text(candidate, max_length=max_length)
        
        # 如果清洗后还有有效内容，进行最终检查
        if cleaned and len(cleaned.strip()) >= 3:
            cleaned_lower = cleaned.lower().strip()
            
            # 最终检查：如果清洗后仍然是指令性文本，跳过
            is_still_instruction = any(
                cleaned_lower.startswith(phrase) or 
                (phrase in cleaned_lower and len(cleaned_lower) < 100 and not any(char in cleaned for char in ['?', '？', '!', '！']))
                for phrase in instruction_phrases
            )
            
            if is_still_instruction:
                # 检查是否包含实际对话内容
                has_actual_content = any(char in cleaned for char in ['?', '？', '!', '！']) and any(
                    cleaned.startswith(starter) for starter in ['如果', '你', '我', '他', '她', '它', '这', '那', '什么', '怎么', '为什么', '如何', '要是', '假如', '你会']
                )
                if not has_actual_content:
                    continue  # 跳过这个块，尝试下一个
            
            # 如果通过了所有检查，返回这个块
            return cleaned
    
    # 如果所有块都无效，返回空字符串
    return ""


def clean_generated_text(text: str, max_length: int = 512) -> str:
    """
    清洗 Chameleons 生成的文本（专门针对英文电影对话）
    
    清洗逻辑（6个步骤）：
    
    步骤 0: 乱码检测
    - 检测并移除乱码文本（数字占比过高、连续数字过长等）
    
    步骤 1: 删除整行污染
    - 移除元数据标签（<|im_start|>, <think> 等）
    - 移除对话格式标记（User:, Assistant:, Character:）
    - 移除所有指令性文本模式（Note:, Thinking process:, Explanation: 等）
    - 移除特定指令性文本（No extra lines, Use only the words, You may not use 等）
    - 移除所有标签（方括号）
    - 移除 LaTeX 格式标记和分隔线
    
    步骤 2: 提取有效块
    - 移除不完整的标签（[/ANS, [/ANSWER 等）
    - 移除重复的标签模式（[/ANSWER][/ANSWER]...）
    - 提取 [ANSWER]...[/ANSWER] 标签内的内容（优先选择第一个非指令性文本的匹配）
    - 处理其他特殊格式（[AI_RESPONSE], [RESPONSE], [OUTPUT], [RESULT] 等）
    - 处理 JSON 数组格式
    
    步骤 3: 连续重复压缩
    - 按英文句子分隔符分割并去重
    - 检测并移除完全重复的子串（长度>=15字符）
    
    步骤 4: 有效句筛选
    - 移除所有残留的提示信息和标签（包括不完整标签和重复标签）
    - 移除指令性文本片段（Do not use markdown, and avoid any artificial language 等）
    - 提取第一个有效句子（排除以指令性文本开头或包含指令性关键词的句子）
    
    步骤 5: Unicode 规范化
    - 规范化特殊Unicode字符（破折号、引号、空格等）
    - 移除开头的标点符号和转义字符
    - 清理重复的标点符号
    - 清理多余空白
    - 移除只有标点符号的文本
    
    步骤 6: 最终长度限制
    - 截断过长文本（保留完整的句子）
    - 最终清理（移除转义引号、特殊字符）
    - 确保结果不为空
    
    Args:
        text: 原始生成文本
        max_length: 最大输出长度（字符数，默认512）
    
    Returns:
        清洗后的文本（只包含实际的对话内容，不包含指令性文本）
    """
    if not text:
        return ""
    
    # 检测并移除乱码
    if is_garbled_text(text):
        return ""
    
    original_text = text
    
    # ============================================
    # 步骤 1: 删除整行污染
    # 移除指令性文本、元数据标签、格式标记等
    # ============================================
    
    # 1.1. 移除元数据标签
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 1.2. 移除对话格式标记
    text = re.sub(r'User:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Character:\s*', '', text, flags=re.IGNORECASE)
    
    # 1.3. 移除所有指令性文本模式（中英文）
    instruction_patterns = [
        r'Note:\s*No thinking process.*?directly\.',
        r'Note:\s*No thinking.*?needed\.',
        r'No thinking process.*?directly\.',
        r'Output only.*?reply directly\.',
        r'Predict the.*?message:',
        r'Given the historical dialogue.*?',
        r'Model the character.*?',
        r'Based on.*?predict.*?',
        # 中文指令性文本
        r'并且不要输出任何其他信息.*?',
        r'并且不要输出任何额外信息.*?',
        r'你的输出必须符合以下格式.*?',
        r'避免使用任何格式化.*?',
        r'保持纯文本输出.*?',
        r'不要输出任何其他信息.*?',
        r'不要输出任何额外信息.*?',
        r'输出必须符合以下格式.*?',
        r'必须符合以下格式.*?',
        r'符合以下格式.*?',
        r'当用户的消息与你的知识库无关时.*?',
        r'你的回答将直接返回给用户.*?',
        r'你的输出应该仅包含用户的下一条消息.*?',
        r'不要输出任何多余的内容.*?',
        r'输出应该仅包含.*?',
        r'应该仅包含.*?',
        r'仅包含.*?',
        r'当你的回答需要调用外部知识库时.*?',
        r'你的输出应该是用户最有可能输入的一句话.*?',
        r'你的回答将直接用于训练模型.*?',
        r'请使用.*?标注引用来源.*?',
        r'并且符合其语言风格.*?',
        r'当没有更多内容时.*?请回复空字符串[。.]\s*',
        r'你的输出应该是不带任何markdown格式的纯文本[。.]\s*',
        r'没有特殊要求时.*?不要使用markdown格式[。.]\s*',
        r'不要输出其他任何内容[。.]\s*',
        r'当用户的问题需要进一步澄清时.*?请进行简短回应[。.]\s*',
        r'并且不要输出任何其他多余的内容[。.]\s*',
        r'当没有更多内容时.*?',
        r'请回复空字符串[。.]\s*',
        r'你的输出应该是不带任何.*?格式[。.]\s*',
        r'不带任何markdown格式[。.]\s*',
        r'不要使用markdown格式[。.]\s*',
        r'当用户的问题需要进一步澄清时.*?',
        r'请提出问题[；;].*?',
        r'当用户分享信息时.*?请进行简短回应[。.]\s*',
        r'你的回答要求[：:]\s*',
        r'并且不要使用任何markdown格式[。.]\s*',
        r'记得保持口语化.*?不要用Markdown格式[。.]\s*',
        r'记得保持口语化[，,]\s*',
        r'不要用Markdown格式[。.]\s*',
        r'不要使用任何markdown格式[。.]\s*',
        r'并且避免使用任何格式[。.]\s*',
        r'\[ANSWER\]',
        r'\[/ANSWER\]',
        r'\[USER_MESSAGE\]',
        r'\[SYSTEM_PROMPT\]',
        r'\[OUTPUT\]',
        r'\[RESPONSE\]',
        r'\[RESULT\]',
        r'Thinking process:',
        r'Explanation:',
        r'Analysis:',
        r'Reasoning:',
        r'Let me think',
        r'Let me analyze',
        r'Based on the context',
        r'According to the character',
        r'The character would say',
        r'This character typically',
        # 新增：过滤常见的指令性文本
        r'No extra lines or characters\.',
        r'Use\s+<u>only</u>\s+the words.*?don\'t invent new ones\.',
        r'Use\s+only\s+the words.*?don\'t invent new ones\.',
        r'You may not use the word.*?',
        r'The character\'s name is\s+\w+\.',
        r'The answer should be the natural continuation.*?',
        r'Use natural language\.',
        r'Use a natural tone.*?',
        r'Keep your tone.*?',
        r'Don\'t use.*?',
        r'Don\'t ever break character\.',
        r'Don\'t use any markdown formatting\.',
        r'Keep your.*?',
        r'Be concise\.',
        r'Don\'t ramble\.',
        r'Don\'t repeat yourself\.',
        r'Don\'t use contractions\.',
        r'Only use full words\.',
        r'Keep your sentences.*?',
        r'Avoid using.*?',
        r'Keep your paragraphs.*?',
        r'Keep your language.*?',
        r'Don\'t use any slang.*?',
        r'Don\'t use any idioms.*?',
        r'Keep your speech.*?',
        r'Don\'t use any emotional language\.',
        r'Keep your tone neutral.*?',
        r'Don\'t use any rhetorical questions\.',
        r'Keep your tone consistent.*?',
        r'Don\'t use any metaphors.*?',
        r'Keep your tone professional.*?',
        r'Don\'t use any hyperbole.*?',
        r'Keep your tone factual.*?',
        r'Don\'t use any sarcasm.*?',
        r'Keep your tone sincere.*?',
        r'Don\'t use any colloquialisms.*?',
        r'Keep your tone respectful.*?',
        r'Don\'t use any profanity.*?',
        r'The user\'s message should continue.*?',
        r'If the user is talking about something else.*?',
        r'Just keep talking as if.*?',
        r'Continue the dialogue.*?',
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 1.3.1. 移除特定的指令性文本（精确匹配 - 中英文）
    specific_instructions = [
        r'No extra lines or characters\.',
        r'Use\s+<u>only</u>\s+the words that are in the previous dialogue,?\s+don\'t invent new ones\.',
        r'Use\s+only\s+the words that are in the previous dialogue,?\s+don\'t invent new ones\.',
        r'You may not use the word\s+"?fucking"?\.',
        r'You may not use the word\s+"?you"?\s+unless.*?',
        r'You may not use the word\s+"?I"?\s+or\s+"?you"?\.',
        r'The character\'s name is\s+\w+\.',
        r'The answer should be the natural continuation of the previous dialogue\.',
        r'Use natural language\.',
        r'Use a natural tone and avoid any artificial language\.',
        r'Do not use markdown\.',
        r'and avoid any artificial language\.',
        r'exactly one line long\.',
        r'End with a period\.',
        r'If the answer requires multiple lines.*?',
        r'use line breaks.*?',
        r'Just write the lines of the scene\.',
        r'The language used must be in English\.',
        r'Keep all responses within the character\'s point of view\.',
        r'Keep the lines short\.',
        r'No extra lines\.',
        r'Use\s+<u>only</u>\s+the words that are in the previous example\.',
        r'Use\s+only\s+the words that are in the previous example\.',
        r'Use English\.',
        r'Keep answers to one line\.',
        r'The answer must not have more than twenty words\.',
        r'The assistant must never use any language outside these tags\.',
        r'No markdown please\.',
        r'Keep the same tense, person, mood, voice, and perspective\.',
        r'Keep it simple\.',
        r'Keep it simple and direct\.',
        r'Use\s+<u>your</u>\s+voice\.',
        r'Keep the lines as short as possible, in the most natural language\.',
        r'The following is the next line spoken by the character, followed by a blank line\.',
        r'This is a direct answer\.',
        # 中文指令性文本
        r'并且不要输出任何其他信息.*?',
        r'并且不要输出任何额外信息.*?',
        r'你的输出必须符合以下格式.*?',
        r'避免使用任何格式化.*?',
        r'保持纯文本输出.*?',
        r'不要输出任何其他信息.*?',
        r'不要输出任何额外信息.*?',
        r'输出必须符合以下格式.*?',
        r'必须符合以下格式.*?',
        r'符合以下格式.*?',
        r'包括文本格式说明等.*?',
        r'文本格式说明等.*?',
        r'当用户的消息与你的知识库无关时.*?',
        r'你的回答将直接返回给用户.*?',
        r'你的输出应该仅包含用户的下一条消息.*?',
        r'不要输出任何多余的内容.*?',
        r'输出应该仅包含.*?',
        r'应该仅包含.*?',
        r'仅包含.*?',
        r'当你的回答需要调用外部知识库时.*?',
        r'你的输出应该是用户最有可能输入的一句话.*?',
        r'你的回答将直接用于训练模型.*?',
        r'请使用.*?标注引用来源.*?',
        r'并且符合其语言风格.*?',
    ]
    
    for pattern in specific_instructions:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 1.4. 移除所有标签（方括号）
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'【[^】]*】', '', text)
    
    # 1.5. 移除 LaTeX 格式标记
    text = re.sub(r'\\boxed\{[^}]*\}', '', text)
    
    # 1.6. 移除分隔线
    text = re.sub(r'-{10,}', '', text)
    text = re.sub(r'[-\s]{20,}', '', text)
    
    # ============================================
    # 步骤 2: 提取有效块
    # 从特殊格式中提取实际内容
    # ============================================
    
    # 2.0. 先移除所有不完整的标签（如 [/ANS, [/ANSWER, [ANSWER 等）
    text = re.sub(r'\[/?ANS[^\]]*$', '', text)  # 移除不完整的标签
    text = re.sub(r'\[/?ANSWER[^\]]*$', '', text)  # 移除不完整的标签
    
    # 2.0.1. 移除重复的标签模式（如 [/ANSWER][/ANSWER]... 或 [/ANSWER] [/ANSWER] [ANSWER]）
    text = re.sub(r'\[/?ANSWER\][\s]*\[/?ANSWER\]+', '', text)  # 移除连续的重复标签
    text = re.sub(r'(\[/ANSWER\][\s]*)+', '', text)  # 移除多个连续的 [/ANSWER]
    text = re.sub(r'(\[ANSWER\][\s]*)+', '', text)  # 移除多个连续的 [ANSWER]
    
    # 2.1. 处理 [ANSWER] 和 [/ANSWER] 格式
    # 注意：如果已经在 extract_and_clean_answer_blocks 中处理过，这里不应该再次提取
    # 因为 clean_generated_text 是在 extract_and_clean_answer_blocks 内部调用的
    # 所以这里只处理没有在 extract_and_clean_answer_blocks 中处理的情况
    if '[ANSWER]' in text:
        # 先找到所有 [ANSWER]...[/ANSWER] 对
        answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
        matches = re.findall(answer_pattern, text, re.DOTALL)
        
        if matches:
            # 对每个匹配进行清洗，找到第一个有效的
            for match in matches:
                candidate = match.strip()
                if not candidate:
                    continue
                
                # 对候选文本进行完整清洗（调用清洗函数，但只清洗内容部分）
                # 先移除明显的指令性文本
                instruction_cleanup = [
                    r'No extra lines or characters\.',
                    r'No extra lines\.',
                    r'Use\s+<u>only</u>\s+the words.*?don\'t invent new ones\.',
                    r'Use\s+only\s+the words.*?don\'t invent new ones\.',
                    r'Use\s+<u>only</u>\s+the words that are in the previous example\.',
                    r'Use\s+only\s+the words that are in the previous example\.',
                    r'You may not use the word.*?',
                    r'unless it is part of the phrase.*?',
                    r'or\s+"?you"?\.',
                    r'or\s+"?I"?\s+or\s+"?you"?\.',
                    r'you"\s+or\s+"your"\.',
                    r'The character\'s name is\s+\w+\.',
                    r'The answer should be the natural continuation.*?',
                    r'Use natural language\.',
                    r'Use a natural tone.*?',
                    r'Do not use markdown\.',
                    r'and avoid any artificial language\.',
                    r'exactly one line long\.',
                    r'End with a period\.',
                    r'If the answer requires multiple lines.*?',
                    r'Just write the lines of the scene\.',
                    r'The language used must be in English\.',
                    r'Use English\.',
                    r'Keep answers to one line\.',
                    r'Keep all responses within the character\'s point of view\.',
                    r'Keep the lines short\.',
                    r'No extra lines\.',
                    r'Use\s+<u>only</u>\s+the words that are in the previous example\.',
                    r'Use\s+only\s+the words that are in the previous example\.',
                    r'The answer must not have more than twenty words\.',
                    r'The assistant must never use any language outside these tags\.',
                    r'No markdown please\.',
                    r'Keep the same tense, person, mood, voice, and perspective\.',
                    r'Keep it simple\.',
                    r'Keep it simple and direct\.',
                    r'Use\s+<u>your</u>\s+voice\.',
                    r'Keep the lines as short as possible, in the most natural language\.',
                    r'The following is the next line spoken by the character, followed by a blank line\.',
                    r'This is a direct answer\.',
                    # 中文指令性文本
                    r'并且不要输出任何其他信息.*?',
                    r'并且不要输出任何额外信息.*?',
                    r'你的输出必须符合以下格式.*?',
                    r'避免使用任何格式化.*?',
                    r'保持纯文本输出.*?',
                    r'不要输出任何其他信息.*?',
                    r'不要输出任何额外信息.*?',
                    r'输出必须符合以下格式.*?',
                    r'必须符合以下格式.*?',
                    r'符合以下格式.*?',
                    r'包括文本格式说明等.*?',
                    r'文本格式说明等.*?',
                    r'当用户的消息与你的知识库无关时.*?',
                    r'你的回答将直接返回给用户.*?',
                    r'你的输出应该仅包含用户的下一条消息.*?',
                    r'不要输出任何多余的内容.*?',
                    r'输出应该仅包含.*?',
                    r'应该仅包含.*?',
                    r'仅包含.*?',
                    r'当你的回答需要调用外部知识库时.*?',
                    r'你的输出应该是用户最有可能输入的一句话.*?',
                    r'你的回答将直接用于训练模型.*?',
                    r'请使用.*?标注引用来源.*?',
                    r'并且符合其语言风格.*?',
                    r'当你的回答是中文时.*?',
                    r'请使用和标签包裹.*?',
                    r'的一句话.*?',
                    r'并且符合他的语言风格.*?',
                ]
                for pattern in instruction_cleanup:
                    candidate = re.sub(pattern, '', candidate, flags=re.IGNORECASE | re.DOTALL)
                candidate = candidate.strip()
                
                # 如果清理后还有内容，且不是纯指令性文本，使用它
                if candidate and len(candidate) > 3:
                    # 再次检查是否包含明显的指令性关键词
                    invalid_keywords = [
                        'use markdown', 'avoid any artificial', 'exactly one line',
                        'end with a period', 'if the answer requires', 'do not use',
                        'you may not use', 'the character\'s name', 'the answer should',
                        'use natural', 'use a natural', 'no extra lines',
                        'just write the lines', 'the language used must be',
                        'keep all responses within', 'keep the lines short',
                        'use english', 'keep answers to one line',
                        'use only the words', 'don\'t invent new ones',
                        'no extra lines', 'use only the words that are in the previous example',
                        'you" or "your', 'keep answers to one line',
                        'the answer must not have more than', 'the assistant must never use',
                        'no markdown please', 'keep the same tense',
                        'keep it simple', 'use your voice',
                        'keep the lines as short as possible', 'the following is the next line',
                        'this is a direct answer', 'must not have more than twenty words',
                        'must never use any language outside', 'followed by a blank line',
                    # 中文指令性关键词
                    '并且不要输出', '不要输出任何', '输出必须符合', '必须符合以下格式',
                    '符合以下格式', '避免使用任何格式化', '保持纯文本输出',
                    '包括文本格式说明', '文本格式说明', '当用户的消息与你的知识库无关',
                    '你的回答将直接返回给用户', '你的输出应该仅包含', '不要输出任何多余的内容',
                    '输出应该仅包含', '应该仅包含', '仅包含', '当你的回答需要调用外部知识库',
                    '你的输出应该是用户最有可能输入', '你的回答将直接用于训练模型',
                    '请使用', '标注引用来源', '并且符合其语言风格',
                    '当你的回答是中文时', '请使用和标签包裹', '的一句话', '并且符合他的语言风格'
                    ]
                    is_instruction = any(keyword in candidate.lower() for keyword in invalid_keywords)
                    if not is_instruction:
                        text = candidate
                        break
            else:
                # 如果所有匹配都无效，使用第一个匹配（至少提取一些内容）
                if matches:
                    text = matches[0].strip()
        
        # 移除所有残留的标签
        text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
    
    # 2.2. 处理 [AI_RESPONSE] 格式
    if '[AI_RESPONSE]' in text:
        ai_response_pattern = r'\[AI_RESPONSE\](.*?)(?=\[AI_RESPONSE\]|\[RECENT_DIALOGUE\]|$)'
        matches = re.findall(ai_response_pattern, text, re.DOTALL)
        if matches:
            text = matches[0].strip()
    
    # 2.3. 处理 [RESPONSE] 格式
    if '[RESPONSE]' in text:
        response_pattern = r'\[RESPONSE\]\s*(.*?)(?=\[RESPONSE\]|$)'
        matches = re.findall(response_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                content = match.strip()
                temp_content = re.sub(r'\[[^\]]*\]', '', content)
                temp_content = re.sub(r'【[^】]*】', '', temp_content)
                temp_content = temp_content.strip()
                if len(temp_content) >= 5:
                    text = content
                    break
            else:
                text = matches[0].strip() if matches else text
    
    # 2.4. 处理 [OUTPUT] 和 [OUTPUT_SCORE] 格式
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
    
    # 2.5. 处理 [RESPONSE_1], [RESPONSE_2] 等格式
    if re.search(r'\[RESPONSE_\d+\]', text):
        response_pattern = r'\[RESPONSE_\d+\]\s*(.*?)(?=\[RESPONSE_\d+\]|$)'
        matches = re.findall(response_pattern, text, re.DOTALL)
        if matches:
            text = matches[0].strip()
    
    # 2.6. 处理 JSON 数组格式
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
    
    # 2.7. 处理 [RESULT] 格式
    if '[RESULT]' in text:
        result_match = re.search(r'\[RESULT\](.*?)(?:\[SYSTEM_LOGIC\]|$)', text, re.DOTALL)
        if result_match:
            result_text = result_match.group(1).strip()
            result_text = re.sub(r'\([^)]*\)', '', result_text)
            text = result_text.strip()
    
    # ============================================
    # 步骤 3: 连续重复压缩
    # 去重，移除重复内容
    # ============================================
    
    # 3.1. 按英文句子分隔符分割并去重
    sentences = re.split(r'([.!?])', text)
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
            
            sent_clean = re.sub(r'[\s\.,!?]', '', sent)
            sent_key = sent_clean.lower().strip()
            
            is_duplicate = False
            for seen_key in seen:
                if sent_key == seen_key:
                    is_duplicate = True
                    break
                if len(sent_key) > 10 and len(seen_key) > 10:
                    if sent_key in seen_key or seen_key in sent_key:
                        if abs(len(sent_key) - len(seen_key)) / max(len(sent_key), len(seen_key)) < 0.2:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                seen.add(sent_key)
                unique_sentences.append(sent)
        
        if unique_sentences:
            text = ' '.join(unique_sentences)
        elif len(sentences) > 2:
            text = (sentences[0] + sentences[1]).strip() if len(sentences) > 1 else sentences[0].strip()
    
    # 3.2. 检测完全重复的子串（长度>=15字符）
    if len(text) > 30:
        for substr_len in range(min(100, len(text) // 2), 14, -1):
            found_repeat = False
            for start_pos in range(min(50, len(text) - substr_len)):
                substr = text[start_pos:start_pos + substr_len]
                substr_normalized = re.sub(r'[\s\.,!?]', '', substr)
                if len(substr_normalized) < 10:
                    continue
                
                occurrences = []
                search_start = 0
                while True:
                    pos = text.find(substr, search_start)
                    if pos == -1:
                        break
                    occurrences.append(pos)
                    search_start = pos + 1
                
                if len(occurrences) >= 2:
                    new_text_parts = []
                    last_end = 0
                    for i, pos in enumerate(occurrences):
                        if i == 0:
                            new_text_parts.append(text[last_end:pos + len(substr)])
                            last_end = pos + len(substr)
                        else:
                            if pos > last_end:
                                new_text_parts.append(text[last_end:pos])
                            last_end = pos + len(substr)
                    if last_end < len(text):
                        new_text_parts.append(text[last_end:])
                    text = ''.join(new_text_parts)
                    found_repeat = True
                    break
            
            if found_repeat:
                break
    
    # ============================================
    # 步骤 4: 有效句筛选
    # 提取第一个有效句子
    # ============================================
    
    # 4.1. 移除所有残留的提示信息和标签
    # 先移除所有不完整的标签
    text = re.sub(r'\[/?ANS[^\]]*', '', text)
    text = re.sub(r'\[/?ANSWER[^\]]*', '', text)
    
    # 移除重复的标签模式
    text = re.sub(r'(\[/ANSWER\][\s]*)+', '', text)
    text = re.sub(r'(\[ANSWER\][\s]*)+', '', text)
    
    # 移除其他残留标签和提示
    text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
    text = re.sub(r'Predict the.*?message:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Note:.*?directly\.', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Thinking process:.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Explanation:.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Analysis:.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Reasoning:.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^→\s*', '', text)
    text = re.sub(r'\\boxed\{[^}]*\}', '', text)
    text = re.sub(r'^[:\s]*', '', text)
    
    # 4.1.1. 移除所有指令性文本片段（更彻底的清理）
    instruction_fragments = [
        r'Do not use markdown\.',
        r'and avoid any artificial language\.',
        r'exactly one line long\.',
        r'End with a period\.',
        r'If the answer requires multiple lines.*?',
        r'use line breaks.*?',
        r'You may not use the word.*?',
        r'unless it is part of the phrase.*?',
        r'or\s+"?you"?\.',
        r'or\s+"?I"?\s+or\s+"?you"?\.',
        r'Just write the lines of the scene\.',
        r'The language used must be in English\.',
        r'Keep all responses within the character\'s point of view\.',
        r'Keep the lines short\.',
        r'No extra lines\.',
        r'Use\s+<u>only</u>\s+the words that are in the previous example\.',
        r'Use\s+only\s+the words that are in the previous example\.',
        r'Use English\.',
        r'Keep answers to one line\.',
        r'The answer must not have more than twenty words\.',
        r'The assistant must never use any language outside these tags\.',
        r'No markdown please\.',
        r'Keep the same tense, person, mood, voice, and perspective\.',
        r'Keep it simple\.',
        r'Keep it simple and direct\.',
        r'Use\s+<u>your</u>\s+voice\.',
        r'Keep the lines as short as possible, in the most natural language\.',
        r'The following is the next line spoken by the character, followed by a blank line\.',
        r'This is a direct answer\.',
        # 中文指令性文本片段
        r'并且不要输出任何其他信息.*?',
        r'并且不要输出任何额外信息.*?',
        r'你的输出必须符合以下格式.*?',
        r'避免使用任何格式化.*?',
        r'保持纯文本输出.*?',
        r'不要输出任何其他信息.*?',
        r'不要输出任何额外信息.*?',
        r'输出必须符合以下格式.*?',
        r'必须符合以下格式.*?',
        r'符合以下格式.*?',
        r'包括文本格式说明等.*?',
        r'文本格式说明等.*?',
        r'当用户的消息与你的知识库无关时.*?',
        r'你的回答将直接返回给用户.*?',
        r'你的输出应该仅包含用户的下一条消息.*?',
        r'不要输出任何多余的内容.*?',
        r'输出应该仅包含.*?',
        r'应该仅包含.*?',
        r'仅包含.*?',
        r'当你的回答需要调用外部知识库时.*?',
        r'你的输出应该是用户最有可能输入的一句话.*?',
        r'你的回答将直接用于训练模型.*?',
        r'请使用.*?标注引用来源.*?',
        r'并且符合其语言风格.*?',
        r'当你的回答是中文时.*?',
        r'请使用和标签包裹.*?',
        r'的一句话.*?',
        r'并且符合他的语言风格.*?',
    ]
    for pattern in instruction_fragments:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 4.2. 提取第一个有效句子
    english_sentences = re.split(r'([.!?])', text)
    if len(english_sentences) > 2:
        first_valid_sentence = None
        for i in range(0, len(english_sentences) - 1, 2):
            if i + 1 < len(english_sentences):
                sent = (english_sentences[i] + english_sentences[i + 1]).strip()
            else:
                sent = english_sentences[i].strip()
            
            if sent and len(sent) > 3:
                # 检查是否是有效句子（不是指令性文本）
                # 排除指令性文本的开头
                invalid_starts = [
                    'Note:', 'Thinking', 'Explanation', 'Analysis', 'Reasoning', 
                    'Predict', 'Given', 'Based', 'According', 'The character', 
                    'This character', 'User:', 'Assistant:', 'Character:', 
                    '→', '[', 'No extra', 'Use only', 'Use <u>only</u>', 
                    'You may not', 'The character\'s name', 'The answer should',
                    'Use natural', 'Use a natural', 'Keep your', 'Don\'t use',
                    'Don\'t ever', 'Be concise', 'Don\'t ramble', 'Don\'t repeat',
                    'Only use', 'Avoid using', 'The user\'s message', 'If the user',
                    'Just keep', 'Continue the dialogue', 'Do not use',
                    'End with', 'If the answer', 'exactly one line',
                    'Just write', 'The language used', 'Keep all responses',
                    'Keep the lines', 'The answer must', 'The assistant must',
                    'No markdown', 'Keep the same', 'Keep it simple',
                    'Use your voice', 'The following is', 'This is a direct',
                    # 中文指令性开头
                    '并且不要', '不要输出', '你的输出', '输出必须', '必须符合',
                    '符合以下', '避免使用', '保持纯文本', '包括文本', '当用户的消息',
                    '你的回答', '你的输出应该', '不要输出任何多余', '输出应该仅包含',
                    '应该仅包含', '仅包含', '当你的回答需要', '你的输出应该是用户',
                    '你的回答将直接用于', '请使用', '当你的回答是中文时', '请使用和标签包裹',
                    '的一句话', '并且符合他的语言风格', '并且符合其语言风格',
                    '并且避免使用任何格式'
                ]
                
                # 检查是否包含指令性关键词
                invalid_keywords = [
                    'would say', 'typically', 'should be', 'may not use',
                    'don\'t invent', 'extra lines', 'natural continuation',
                    'natural language', 'natural tone', 'break character',
                    'markdown formatting', 'use contractions', 'full words',
                    'complex sentence', 'slang or jargon', 'formal and polite',
                    'idioms or expressions', 'emotional language',
                    'neutral and objective', 'rhetorical questions',
                    'metaphors or similes', 'professional and appropriate',
                    'hyperbole or understatement', 'factual and accurate',
                    'sarcasm or irony', 'sincere and genuine',
                    'colloquialisms or vulgarities', 'respectful and considerate',
                    'profanity or crude language', 'do not use markdown',
                    'avoid any artificial language', 'exactly one line long',
                    'end with a period', 'if the answer requires',
                    'use line breaks', 'unless it is part of the phrase',
                    'or "you"', 'or "I" or "you"', 'just write the lines',
                    'the language used must be', 'keep all responses within',
                    'keep the lines short', 'write the lines of the scene',
                    'language used must be in english', 'must be in english',
                    # 中文指令性关键词
                    '并且不要输出', '不要输出任何', '输出必须符合', '必须符合以下格式',
                    '符合以下格式', '避免使用任何格式化', '保持纯文本输出',
                    '包括文本格式说明', '文本格式说明', '当用户的消息与你的知识库无关',
                    '你的回答将直接返回给用户', '你的输出应该仅包含', '不要输出任何多余的内容',
                    '输出应该仅包含', '应该仅包含', '仅包含', '当你的回答需要调用外部知识库',
                    '你的输出应该是用户最有可能输入', '你的回答将直接用于训练模型',
                    '请使用', '标注引用来源', '并且符合其语言风格',
                    '当你的回答是中文时', '请使用和标签包裹', '的一句话', '并且符合他的语言风格',
                    '并且避免使用任何格式'
                ]
                
                is_invalid = False
                for invalid_start in invalid_starts:
                    if sent.startswith(invalid_start):
                        is_invalid = True
                        break
                
                if not is_invalid:
                    for keyword in invalid_keywords:
                        if keyword in sent.lower():
                            is_invalid = True
                            break
                
                if not is_invalid:
                    first_valid_sentence = sent
                    break
        
        if first_valid_sentence:
            text = first_valid_sentence
        elif len(english_sentences) > 2:
            for i in range(0, len(english_sentences) - 1, 2):
                if i + 1 < len(english_sentences):
                    sent = (english_sentences[i] + english_sentences[i + 1]).strip()
                else:
                    sent = english_sentences[i].strip()
                if sent and len(sent) > 3:
                    text = sent
                    break
    
    # ============================================
    # 步骤 5: Unicode 规范化
    # 规范化字符和格式
    # ============================================
    
    # 5.1. 规范化特殊Unicode字符
    text = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2015', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u00A0', ' ').replace('\u3000', ' ')
    
    # 5.2. 移除开头的标点符号和转义字符
    text = re.sub(r'^\\?"', '', text)
    text = re.sub(r'^\\?\'', '', text)
    text = text.lstrip(r'.!?,\s\-')
    text = re.sub(r'^[.!?,:;\-]\s+', '', text)
    text = re.sub(r'^[:\s]*', '', text)
    if text.startswith('-'):
        text = text.lstrip('-').lstrip()
    
    # 5.3. 清理重复的标点符号（包括中文和英文标点）
    # 处理英文标点符号
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # 处理中文标点符号
    text = re.sub(r'([！？。])\1{2,}', r'\1', text)
    text = re.sub(r'([！？。])\1+', r'\1', text)
    # 处理混合的重复标点（如多个问号或感叹号）
    text = re.sub(r'[？?]{3,}', '？', text)
    text = re.sub(r'[！!]{3,}', '！', text)
    
    # 5.4. 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 5.5. 移除只有标点符号的文本
    if text and len(text) <= 3 and all(c in '.!?,:;' for c in text):
        text = ""
    
    # 5.6. 移除分隔线（如果还有残留）
    if re.match(r'^[-_\s]{5,}$', text):
        text = ""
    
    # ============================================
    # 步骤 6: 最终长度限制
    # 截断到最大长度
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
    
    # 6.2. 最终清理
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 6.3. 移除转义引号和开头的特殊字符
    text = re.sub(r'^\\?"', '', text)
    text = re.sub(r'^\\?\'', '', text)
    text = re.sub(r'^[:\s]*', '', text)
    
    # 6.4. 再次清理空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 6.5. 确保结果不为空
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
    构建 LovinkDialogue 推理 prompt（与训练时的格式一致）
    
    注意：在 LovinkDialogue 中，目标用户（要预测的人）的话被映射为 assistant，
    对话者的话映射为 user。这与训练时的逻辑一致。
    """
    parts = []
    
    # 0. USER_HASH 部分 - 始终包含（与训练时一致）
    if user_info.get('user_hash'):
        parts.append(f"[USER_HASH={user_info['user_hash']}]")
        parts.append("")
    
    # 1. 用户画像
    if use_profile and user_info.get('user_profile'):
        profile = user_info['user_profile']
        profile_tags = []
        
        # 基础信息
        if 'name' in profile and profile['name']:
            profile_tags.append(f"[USER_NAME={profile['name']}]")
        if 'age' in profile and profile['age']:
            profile_tags.append(f"[USER_AGE={profile['age']}]")
        if 'gender' in profile and profile['gender']:
            profile_tags.append(f"[USER_GENDER={profile['gender']}]")
        
        # 人格维度（支持扁平化和嵌套格式）
        if 'dimensions' in profile and isinstance(profile['dimensions'], dict):
            dims = profile['dimensions']
            
            # 检查是否是扁平化格式（包含 "." 的键）
            is_flat = any('.' in str(k) for k in dims.keys())
            
            if is_flat:
                # 扁平化格式：直接遍历
                for dim_key, dim_score in dims.items():
                    if dim_score is not None:
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
                                tag_name = f"DIM_{scale_name.upper()}_{subdim_name.upper()}"
                                profile_tags.append(f"[{tag_name}={score}]")
        
        # 其他以 DIM_ 开头的字段
        for key, value in profile.items():
            if (key.startswith('DIM_') or key.startswith('dim_')) and value is not None:
                profile_tags.append(f"[{key.upper()}={value}]")
        
        if profile_tags:
            parts.append("[USER_PROFILE]")
            parts.extend(profile_tags)
            parts.append("")
    
    # 2. 任务描述（与训练时保持一致 - 中文）
    parts.append("[TASK]")
    parts.append("基于用户在 Lovink 对话中的历史数据，模拟该用户的对话行为模式")
    parts.append("")
    
    # 3. 历史信息
    if use_history and user_info.get('history'):
        history = user_info['history']
        history_parts = ["[HISTORY]"]
        # 限制历史条目数量，避免过长
        max_history_items = 15
        history_to_use = history[:max_history_items] if len(history) > max_history_items else history
        
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
    
    # 5. 预测指令（与训练时保持一致 - 中文）
    parts.append("预测下一条回复：")
    
    # 6. 添加输出要求说明（使用 [ANSWER] 标签，中文提示词）
    parts.append("注意：请直接给出下一条回复，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
    
    return "\n".join(parts)


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
    max_model_len: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_tokens: int = 128,
    seed: int = 42,
    max_context_turns: int = 15,
    max_chars_per_turn: int = 300,
    max_retries: int = 3,
    retry_temperature_increment: float = 0.2
):
    """
    使用 vLLM 运行推理
    """
    print("=" * 80)
    print("vLLM 推理配置 - LovinkDialogue")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: LovinkDialogue")
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
    )
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 (耗时: {load_time:.2f}s)")
    
    # 采样参数
    # LovinkDialogue 需要稳定的输出，使用适中的 temperature 和 top_p
    # 降低 temperature 以减少重复和指令性文本生成
    enhanced_temperature = min(temperature, 0.8)  # 降低到 0.8 以减少重复
    enhanced_top_p = min(top_p, 0.95)  # 默认 0.85-0.95
    # 提高 repetition_penalty 以减少重复内容
    actual_repetition_penalty = max(repetition_penalty, 1.15) if repetition_penalty > 0 else 1.15
    actual_max_tokens = min(max_tokens, 1024)  # LovinkDialogue 对话可能较长，限制到 1024 tokens
    
    print(f"\n采样参数 (LovinkDialogue 优化):")
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
                # 在 LovinkDialogue 中，目标用户（要预测的人）的话被映射为 assistant，对话者的话映射为 user
                user_name = user_info.get('user_profile', {}).get('name', '')
                converted_context = []
                
                context_to_process = raw_context[-max_context_turns:] if len(raw_context) > max_context_turns else raw_context
                
                for turn in context_to_process:
                    source = str(turn.get('source', '')).strip()
                    content = turn.get('content', '')
                    
                    if len(content) > max_chars_per_turn:
                        content = content[:max_chars_per_turn-3] + "..."
                    
                    # 判断是否是目标用户（要预测的人）说的话
                    # 如果 source == user_name 或 source.lower() == 'user'，则映射为 assistant
                    is_target_user = False
                    if source.lower() == 'user':
                        is_target_user = True
                    elif user_name and source == user_name:
                        is_target_user = True
                    
                    # 目标用户的话映射为 assistant（模型要学习生成的）
                    # 对话者的话映射为 user（输入/上下文）
                    role = 'assistant' if is_target_user else 'user'
                    converted_context.append({
                        'role': role,
                        'content': content
                    })
                
                user_info_with_context = user_info.copy()
                user_info_with_context['context'] = converted_context
                
                # 确保 user_hash 被传递
                if 'user_hash' not in user_info_with_context and user_info.get('user_hash'):
                    user_info_with_context['user_hash'] = user_info['user_hash']
                
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
                        repetition_penalty=actual_repetition_penalty,
                        max_tokens=actual_max_tokens,
                        seed=sample_seed,
                        skip_special_tokens=True,
                    ))
    
    print(f"总推理请求数: {len(all_prompts)}")
    
    # 批量推理（带重试机制）
    print("\n开始批量推理（带空结果重试机制）...")
    inference_start = time.time()
    
    # 第一轮生成
    generated_texts = generate_with_vllm(
        llm=llm,
        prompts=all_prompts,
        sampling_params=all_sampling_params,
        show_progress=True
    )
    
    # 检查并重试空结果
    # max_retries 和 retry_temperature_increment 已从参数传入
    
    print(f"\n检查空结果并重试（最多 {max_retries} 次）...")
    retry_count = 0
    retry_indices = []  # 需要重试的索引
    
    # 第一轮检查：找出清洗后为空的样本（先提取 [ANSWER] 块再清洗）
    for idx, (prompt, generated_text) in enumerate(zip(all_prompts, generated_texts)):
        # 先尝试从 [ANSWER] 块中提取和清洗
        cleaned_text = extract_and_clean_answer_blocks(generated_text, max_length=1024)
        # 如果从 [ANSWER] 块中没有提取到有效内容，尝试直接清洗整个文本
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            cleaned_text = clean_generated_text(generated_text, max_length=1024)
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            retry_indices.append(idx)
    
    # 重试循环
    while retry_indices and retry_count < max_retries:
        retry_count += 1
        current_temperature = enhanced_temperature + retry_temperature_increment * retry_count
        current_temperature = min(current_temperature, 1.5)  # 限制最大 temperature
        
        print(f"  重试第 {retry_count} 轮: {len(retry_indices)} 个空结果, temperature={current_temperature:.2f}")
        
        # 准备重试的 prompts 和 sampling_params
        retry_prompts = [all_prompts[idx] for idx in retry_indices]
        retry_sampling_params = []
        for idx in retry_indices:
            metadata = all_metadata[idx]
            sample_seed = seed + metadata['sample_idx'] + retry_count * 1000  # 使用不同的seed
            retry_sampling_params.append(SamplingParams(
                temperature=current_temperature,
                top_p=enhanced_top_p,
                top_k=top_k,
                repetition_penalty=actual_repetition_penalty,
                max_tokens=actual_max_tokens,
                seed=sample_seed,
                skip_special_tokens=True,
            ))
        
        # 重试生成
        retry_generated_texts = generate_with_vllm(
            llm=llm,
            prompts=retry_prompts,
            sampling_params=retry_sampling_params,
            show_progress=False
        )
        
        # 更新结果并检查（先提取 [ANSWER] 块再清洗）
        new_retry_indices = []
        for i, idx in enumerate(retry_indices):
            retry_text = retry_generated_texts[i]
            # 先尝试从 [ANSWER] 块中提取和清洗
            cleaned_retry_text = extract_and_clean_answer_blocks(retry_text, max_length=1024)
            # 如果从 [ANSWER] 块中没有提取到有效内容，尝试直接清洗整个文本
            if not cleaned_retry_text or len(cleaned_retry_text.strip()) < 3:
                cleaned_retry_text = clean_generated_text(retry_text, max_length=1024)
            
            if cleaned_retry_text and len(cleaned_retry_text.strip()) >= 3:
                # 重试成功，更新结果
                generated_texts[idx] = retry_text
            else:
                # 仍然为空，需要再次重试
                new_retry_indices.append(idx)
        
        retry_indices = new_retry_indices
        
        if retry_indices:
            print(f"    仍有 {len(retry_indices)} 个样本清洗后为空")
        else:
            print(f"    ✓ 所有重试样本都生成了有效结果")
    
    if retry_indices:
        print(f"  警告: 仍有 {len(retry_indices)} 个样本在 {max_retries} 次重试后仍为空")
    
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
    examples_content.append(f"数据集: LovinkDialogue")
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
    
    # 清洗生成的文本
    print(f"\n清洗生成的文本...")
    # 按 (test_sample_idx, collection_idx, data_item_idx, sample_idx) 分组
    data_item_samples = {}  # key: (test_sample_idx, collection_idx, data_item_idx), value: dict of {sample_idx: cleaned_text}
    
    cleaned_count = 0
    empty_count = 0
    
    # 保存原始输出和清洗后的结果
    processing_log_content = []
    processing_log_content.append("=" * 80)
    processing_log_content.append("推理阶段原始输出和处理结果")
    processing_log_content.append("=" * 80)
    processing_log_content.append("")
    
    # 按 data_item 和 sample_idx 分组处理
    for idx, (metadata, generated_text) in enumerate(zip(all_metadata, generated_texts)):
        key = (metadata['test_sample_idx'], metadata['collection_idx'], metadata['data_item_idx'])
        sample_idx = metadata['sample_idx']
        
        if key not in data_item_samples:
            data_item_samples[key] = {}
        
        # 先尝试从 [ANSWER] 块中提取和清洗
        cleaned_text = extract_and_clean_answer_blocks(generated_text, max_length=1024)
        
        # 如果从 [ANSWER] 块中没有提取到有效内容，尝试直接清洗整个文本
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            cleaned_text = clean_generated_text(generated_text, max_length=1024)
        
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
            # 如果这个 sample_idx 还没有有效回答，标记为 None
            if sample_idx not in data_item_samples[key]:
                data_item_samples[key][sample_idx] = None
    
    if cleaned_count > 0:
        print(f"✓ 已清洗 {cleaned_count} 个生成样本")
    if empty_count > 0:
        print(f"  警告: {empty_count} 个生成样本清洗后为空，将标记为[生成失败]")
    
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
    failed_count = 0
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
        # 如果某个 sample_idx 的结果为空，从同一个 data_item 的其他有效回答中复制
        final_continuations = []
        for sample_idx in range(num_samples):
            if sample_idx in samples_dict and samples_dict[sample_idx]:
                # 使用对应 sample_idx 的有效回答
                final_continuations.append(samples_dict[sample_idx])
            else:
                # 如果该 sample_idx 没有有效回答，从同一个 data_item 的其他有效回答中复制
                if valid_answers:
                    # 循环使用有效回答（确保每个 sample_idx 都有不同的回答，如果可能的话）
                    copy_idx = len(final_continuations) % len(valid_answers)
                    final_continuations.append(valid_answers[copy_idx])
                    copied_count += 1
                else:
                    # 如果所有 sample_idx 都没有有效回答，标记为生成失败
                    final_continuations.append("[生成失败]")
                    failed_count += 1
        
        data_item['continuations'] = final_continuations
        filled_count += 1
    
    if copied_count > 0:
        print(f"  信息: {copied_count} 个空样本已从其他有效回答中复制")
    if failed_count > 0:
        print(f"  警告: {failed_count} 个 data_item 的所有样本清洗后为空，已标记为[生成失败]")
    
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
    parser = argparse.ArgumentParser(description='vLLM 高性能推理 - LovinkDialogue 专用')
    
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
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='最大生成 token 数（默认: 512）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    parser.add_argument('--max_context_turns', type=int, default=15,
                       help='Context 最大对话轮次数（默认: 15）')
    parser.add_argument('--max_chars_per_turn', type=int, default=300,
                       help='每轮对话最大字符数（默认: 300）')
    
    parser.add_argument('--max_retries', type=int, default=3,
                       help='空结果最大重试次数（默认: 3）')
    parser.add_argument('--retry_temperature_increment', type=float, default=0.2,
                       help='每次重试时增加的 temperature（默认: 0.2）')
    
    args = parser.parse_args()
    
    # 推断 scenario_path
    if args.scenario_path is None:
        args.scenario_path = '/mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue'
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
        max_chars_per_turn=args.max_chars_per_turn,
        max_retries=args.max_retries,
        retry_temperature_increment=args.retry_temperature_increment
    )


if __name__ == '__main__':
    main()
