"""
使用 vLLM 进行高性能推理 - LovinkQuestionnaire 专用版本

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
python inference_vllm_LovinkQuestionnaire.py \
    --checkpoint_dir outputs/LovinkQuestionnaire_8B_profile_and_history \
    --ablation_config profile_and_history \
    --num_samples 5 \
    --output_dir outputs/leaderboards/LovinkQuestionnaire_vllm_8B \
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
    检测文本是否为乱码（针对中文问卷回答）
    """
    if not text or len(text.strip()) < 3:
        return False
    
    text_clean = text.strip()
    digit_count = sum(1 for c in text_clean if c.isdigit())
    # 中文字符检测
    chinese_count = sum(1 for c in text_clean if '\u4e00' <= c <= '\u9fff')
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
    
    # 几乎没有中文字符（中文占比<10%且总长度>10）
    if total_chars > 10:
        chinese_ratio = chinese_count / total_chars
        if chinese_ratio < 0.1 and digit_ratio > 0.3:
            return True
    
    return False


def clean_generated_text(text: str, max_length: int = 512) -> str:
    """
    清洗 LovinkQuestionnaire 生成的文本（专门针对中文问卷回答）
    
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
    # 步骤 0: 模板识别和提取（最高优先级）
    # 先识别 [ANSWER]...[/ANSWER] 模板，然后针对不同模板采用不同策略
    # ============================================
    
    original_text = text  # 保存原始文本
    template_type = None
    extracted_text = None
    answer_extracted = False
    
    # 0.1. 识别 [ANSWER]...[/ANSWER] 模板（最高优先级）
    # 对于 LovinkQuestionnaire，只保留第一个 [ANSWER] 中的内容
    if '[ANSWER]' in text:
        # 先尝试匹配完整的 [ANSWER]...[/ANSWER]
        if '[/ANSWER]' in text:
            # 匹配所有 [ANSWER]...[/ANSWER] 对，只取第一个
            answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                # 只使用第一个匹配（第一个 [ANSWER] 中的内容）
                first_match = matches[0].strip()
                # 移除中间的 [ANSWER_FORMAT=...] 等标签
                content = re.sub(r'\[ANSWER[^\]]*\]', '', first_match).strip()
                if content and len(content) >= 1:  # 问卷回答可能很短，降低最小长度要求
                    template_type = 'ANSWER'
                    extracted_text = content
                    answer_extracted = True
        else:
            # 如果没有找到 [/ANSWER]，但找到了 [ANSWER]，提取第一个 [ANSWER] 到下一个 [ANSWER] 或文本末尾
            answer_pattern = r'\[ANSWER\](.*?)(?=\[ANSWER\]|$)'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                # 只使用第一个匹配
                first_match = matches[0].strip()
                content = re.sub(r'\[/?ANSWER[^\]]*\]', '', first_match).strip()
                if content and len(content) >= 1:
                    template_type = 'ANSWER'
                    extracted_text = content
                    answer_extracted = True
    
    # 0.2. 如果识别到模板并提取成功，使用提取的内容
    if extracted_text:
        text = extracted_text
    else:
        # 如果没有识别到 [ANSWER] 标签，尝试提取第一句话或第一个有效回答
        # 对于问卷回答，常见的格式是：同意、有点同意、完全同意等
        # 问卷回答关键词列表（按长度从长到短排序，优先匹配长的）
        questionnaire_keywords = [
            '完全同意', '完全不同意', '有点同意', '有点不同意', 
            '同意', '不同意', '不确定'
        ]
        # 按长度从长到短排序，优先匹配长的关键词
        sorted_keywords = sorted(questionnaire_keywords, key=len, reverse=True)
        
        # 先移除开头的空白和换行
        text_clean = original_text.strip()
        
        # 尝试匹配第一句话中的问卷回答关键词
        # 匹配模式：可能包含一些说明文字，但最后是问卷回答
        # 例如："在提交时仔细阅读，同意。" -> 提取 "同意"
        first_sentence_pattern = r'^([^。！？\n]+[。！？]?)'
        first_sentence_match = re.match(first_sentence_pattern, text_clean)
        
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1)
            # 在第一个句子中查找问卷回答关键词（优先匹配长的）
            for keyword in sorted_keywords:
                if keyword in first_sentence:
                    # 找到关键词，直接使用完整的关键词
                    # 不要提取从关键词开始到句子结束的部分，因为可能会包含其他内容
                    text = keyword
                    answer_extracted = True
                    break
        
        # 如果没有找到关键词，尝试提取第一行或第一个非空行
        if not answer_extracted:
            lines = text_clean.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) <= 50:  # 问卷回答通常很短
                    # 检查是否包含问卷回答关键词（优先匹配长的）
                    for keyword in sorted_keywords:
                        if keyword in line:
                            # 直接使用完整的关键词
                            text = keyword
                            answer_extracted = True
                            break
                    if answer_extracted:
                        break
            
            # 如果还是没有找到，使用第一行（去除空白和标点）
            if not answer_extracted and lines:
                first_line = lines[0].strip()
                # 移除开头的标点和空白
                first_line = re.sub(r'^[，。！？\s]+', '', first_line)
                # 如果第一行很短（可能是回答），使用它
                if first_line and len(first_line) <= 50:
                    text = first_line
                else:
                    text = original_text
        else:
            # 如果已经提取到，使用提取的内容
            pass
    
    # ============================================
    # 第一步：提取有效信息（从特殊格式中提取实际内容）
    # ============================================
    
    # 如果已经在步骤0中识别到模板并提取，跳过此步骤
    if not template_type:
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
        
        # 1.1.1. 处理 [ANSWER] 和 [/ANSWER] 格式（备用逻辑）
        if '[ANSWER]' in text and '[/ANSWER]' in text:
            answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                text = matches[0].strip()
                text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
                answer_extracted = True
    
    # 1.2. 处理 [RESPONSE_1], [RESPONSE_2] 等格式（仅在未识别到模板时）
    if not template_type:
        if re.search(r'\[RESPONSE_\d+\]', text):
            response_pattern = r'\[RESPONSE_\d+\]\s*(.*?)(?=\[RESPONSE_\d+\]|$)'
            matches = re.findall(response_pattern, text, re.DOTALL)
            if matches:
                text = matches[0].strip()
    
    # 1.3. 处理 JSON 数组格式（仅在未识别到模板时）
    if not template_type:
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
    
    # 1.4. 处理 [RESULT] 格式（仅在未识别到模板时）
    if not template_type:
        if '[RESULT]' in text:
            result_match = re.search(r'\[RESULT\](.*?)(?:\[SYSTEM_LOGIC\]|$)', text, re.DOTALL)
            if result_match:
                result_text = result_match.group(1).strip()
                result_text = re.sub(r'\([^)]*\)', '', result_text)
                text = result_text.strip()
    
    # ============================================
    # 第二步：移除元数据和格式标记
    # ============================================
    
    # 2.0. 如果已从模板提取，截断到第一个角色标识或提示信息之前
    if template_type:
        # 先移除残留的 [ANSWER] 和 [/ANSWER] 标签
        text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
        
        truncation_patterns = [
            r'\s*Assistant:\s*',
            r'\s*User:\s*',
            r'\s*AI\'s reply[：:]?',
            r'\s*AI\'s response[：:]?',
            r'\s*Predict the user\'s next message[：:]?\s*',
            r'\[RECENT_DIALOGUE\]',
            r'\[SYSTEM_LOGIC\]',
            r'Predict the user\'s next message[：:]?',
            r'Note:.*?',
            r'IMPORTANT:.*?',
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
    
    # 2.1. 移除元数据标签
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 2.2. 移除对话格式标记（User:, Assistant:）
    text = re.sub(r'User:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    
    # 2.3. 移除所有标签（方括号），但保留 [ANSWER] 和 [/ANSWER] 标签（如果还有残留）
    # 先移除其他标签，保留 [ANSWER] 和 [/ANSWER] 以便后续提取
    # 匹配 [XXX] 但不匹配 [ANSWER] 或 [/ANSWER]
    text = re.sub(r'\[(?!/?ANSWER\])[^\]]*\]', '', text)
    
    # ============================================
    # 第三步：移除指令性文本
    # ============================================
    
    # 3.1. 移除所有指令性文本模式（中英文）
    instruction_patterns = [
        r'预测用户.*?回复[：:]?\s*',
        r'预测用户.*?回答[：:]?\s*',
        r'Predict the user\'s next message[：:]?\s*',
        r'Predict the user\'s response[：:]?\s*',
        r'注意[：:].*?直接[。.]?\s*',
        r'注意[：:].*?输出[。.]?\s*',
        r'Note:.*?directly[。.]?\s*',
        r'Note:.*?output[。.]?\s*',
        r'不需要思考过程[，,]?.*?直接输出[。.]?\s*',
        r'只需直接输出.*?回复[。.]?\s*',
        r'直接输出.*?回答[。.]?\s*',
        r'No thinking process.*?needed[。.]?\s*',
        r'Output only.*?reply[。.]?\s*',
        r'Direct output.*?message[。.]?\s*',
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3.2. 检测并移除思考过程（在截断之前先移除明显的思考模式）
    # 针对中文问卷回答的思考模式
    thinking_patterns = [
        r'好的[，,]?\s*所以\s+',
        r'现在[，,]?\s*',
        r'等等[，,]?\s*',
        r'看起来\s+',
        r'另外[，,]?\s*',
        r'然而[，,]?\s*',
        r'由于\s+',
        r'根据\s+',
        r'用户\s+(?:可能|会|应该|可以|需要)',
        r'这个问题\s+(?:是|关于|涉及)',
        r'我认为\s+',
        r'我觉得\s+',
        r'我的看法是\s+',
        r'从\s+.*?\s+来看',
        r'Okay,?\s+so\s+',
        r'Now,?\s+the\s+',
        r'Wait,?\s+the\s+',
        r'Looking\s+at\s+',
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
                # 检查是否包含明显的思考关键词（中英文）
                has_thinking_keywords = any(
                    keyword in sent.lower() 
                    for keyword in ['好的，所以', '现在，', '等等，', '看起来', '用户可能', '用户会', '我认为', '我觉得', '这个问题', 
                                   'okay, so', 'now, the', 'wait,', 'looking at', 'the user', 'user might', 'user could', 'user would', 'assistant\'s']
                )
                if not has_thinking_keywords:
                    text = sent
                    break
    
    # 3.3. 截断到第一个角色标识或提示信息之前（如果未在步骤2.0中处理）
    if not template_type:
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
    # 第四步：去重和提取第一个有效回答
    # ============================================
    
    # 4.1. 对于问卷回答，只保留第一个有效回答
    # 问卷回答关键词列表（按长度从长到短排序，优先匹配长的）
    questionnaire_keywords = [
        '完全同意', '完全不同意', '有点同意', '有点不同意', 
        '同意', '不同意', '不确定'
    ]
    
    # 如果文本中包含多个问卷回答关键词，只保留第一个
    # 优先匹配长的关键词（避免"有点同意"被匹配为"同意"）
    found_keywords = []
    # 按长度从长到短排序，优先匹配长的关键词
    sorted_keywords = sorted(questionnaire_keywords, key=len, reverse=True)
    
    # 使用正则表达式匹配，确保匹配完整的关键词（不是子串）
    # 先匹配长的关键词，避免短的关键词覆盖长的
    matched_positions = set()  # 记录已经被匹配的位置范围
    
    for keyword in sorted_keywords:
        # 使用正则表达式查找所有匹配位置
        pattern = re.escape(keyword)
        matches = list(re.finditer(pattern, text))
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            # 检查这个位置是否已经被更长的关键词覆盖
            is_covered = False
            for (covered_start, covered_end) in matched_positions:
                if covered_start <= start_pos < covered_end:
                    is_covered = True
                    break
            
            if not is_covered:
                found_keywords.append((start_pos, keyword))
                matched_positions.add((start_pos, end_pos))
                break  # 每个关键词只记录第一个匹配位置
    
    if found_keywords:
        # 按位置排序，找到第一个出现的关键词
        found_keywords.sort(key=lambda x: x[0])
        first_keyword_pos, first_keyword = found_keywords[0]
        
        # 提取从第一个关键词开始的内容
        # 关键：只保留完整的关键词本身，不要截断到下一个关键词
        # 因为"有点同意"包含"同意"，如果截断到"同意"的位置，就会丢失"有点"
        
        # 直接提取完整的关键词
        extracted = first_keyword
        
        # 如果关键词后面还有内容，检查是否需要保留
        # 但通常问卷回答就是关键词本身，所以直接使用关键词即可
        # 如果原始文本在关键词后面有标点或其他内容，可以保留（但通常不需要）
        
        text = extracted
    
    # 4.2. 按句子分割并去重（如果还有多个句子）
    sentences = re.split(r'([。！？\.!?]\s*)', text)
    if len(sentences) > 2:
        seen = set()
        unique_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sent = (sentences[i] + sentences[i + 1]).strip()
            else:
                sent = sentences[i].strip()
            
            if not sent or len(sent) < 1:  # 问卷回答可能很短，降低最小长度
                continue
            
            sent_clean = re.sub(r'[，。！？\s]', '', sent).lower().strip()
            
            is_duplicate = False
            for seen_key in seen:
                if sent_clean == seen_key:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(sent_clean)
                unique_sentences.append(sent)
        
        if unique_sentences:
            text = unique_sentences[0]  # 只保留第一个非重复句子
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
    
    # 6.3. 移除所有残留的 [ANSWER] 和 [/ANSWER] 标签
    text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
    
    # 6.4. 确保结果不为空
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
    构建 LovinkQuestionnaire 推理 prompt（与训练时的格式一致）
    """
    parts = []
    
    # 1. USER_PROFILE 部分 - 与训练时格式一致
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
        
        # 心理维度标签（dimensions）- 支持扁平化和嵌套格式
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
        
        # 其他 profile 字段
        excluded_keys = {'name', 'age', 'gender', 'dimensions', 'unstructured'}
        for key, value in profile.items():
            if key not in excluded_keys and value:
                tag_name = f"USER_{key.upper()}"
                profile_tags.append(f"[{tag_name}={value}]")
        
        if profile_tags:
            parts.append("[USER_PROFILE]")
            parts.extend(profile_tags)
            parts.append("")
    
    # 2. TASK 部分 - 与训练时一致
    parts.append("[TASK]")
    parts.append("基于用户在 Lovink 问卷中的回答数据，模拟该用户的回答风格和行为模式")
    parts.append("")
    
    # 3. HISTORY 部分 - 与训练时格式一致
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
        
        if len(history_parts) > 1:
            parts.extend(history_parts)
            parts.append("")
    
    # 4. 问卷问题部分 - LovinkQuestionnaire 使用问题格式，不是对话格式
    if use_context:
        context = user_info.get('context', [])
        if context:
            # 对于问卷，context 是问题列表，只取最后一个问题（当前要回答的问题）
            # 如果 context 中有多个问题，只使用最后一个
            if len(context) > 0:
                # 获取最后一个问题（通常是当前要回答的问题）
                last_question = context[-1]
                question_content = last_question.get('content', '') if isinstance(last_question, dict) else str(last_question)
                
                # 如果问题太长，截断
                if len(question_content) > 500:
                    question_content = question_content[:497] + "..."
                
                # 直接显示问题，使用 [CURRENT_QUESTION] 格式（与训练时一致）
                parts.append(f"[CURRENT_QUESTION]\n{question_content}")
                parts.append("")
    
    # 5. 预测指令 - 与训练时一致（中文）
    parts.append("预测用户针对该问题的回复：")
    
    # 6. 添加输出要求说明（与训练时保持一致，使用 [ANSWER] 标签，中文）
    parts.append("注意：请直接给出用户针对该问题的回复，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
    
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
    max_model_len: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_tokens: int = 64,
    seed: int = 42,
    max_context_turns: int = 15,
    max_chars_per_turn: int = 300  # 新增：每轮对话最大字符数
):
    """
    使用 vLLM 运行推理
    """
    print("=" * 80)
    print("vLLM 推理配置 - LovinkQuestionnaire")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: LovinkQuestionnaire")
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
    
    # 采样参数（LovinkQuestionnaire 是中文问卷回答）
    enhanced_temperature = max(temperature, 0.8)  # 问卷回答推荐 0.8-1.0
    actual_repetition_penalty = repetition_penalty if repetition_penalty > 0 else 1.1
    actual_max_tokens = min(max_tokens, 256)  # 限制最大长度以减少思考过程
    
    actual_num_samples = num_samples
    print(f"\n采样参数 (LovinkQuestionnaire 优化):")
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
                # 提取 context（问卷问题）
                raw_context = data_item.get('context', [])
                
                # LovinkQuestionnaire: context 为空则跳过
                if not raw_context:
                    continue
                
                # 转换 context 格式：对于问卷，context 是问题列表
                # 在问卷任务中，context 只包含问题（不是对话），所以直接使用原始格式
                # 但为了兼容 build_inference_prompt 的接口，仍然转换为 {role, content} 格式
                # 所有 context 项都是问题，role 设为 'user'
                converted_context = []
                
                # 对于问卷，通常只有一个问题（当前要回答的问题）
                # 只取最后一个问题（当前要回答的问题）
                if len(raw_context) > 0:
                    last_question = raw_context[-1]
                    question_content = last_question.get('content', '') if isinstance(last_question, dict) else str(last_question)
                    
                    # 如果问题太长，截断
                    if len(question_content) > 500:
                        question_content = question_content[:497] + "..."
                    
                    # 问卷问题作为 'user' 角色
                    converted_context.append({
                        'role': 'user',
                        'content': question_content
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
                    sample_seed = seed + sample_idx * 1 # 确保每个样本的seed不同
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
    examples_content.append(f"数据集: LovinkQuestionnaire")
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
        
        # 使用完整的清洗函数（只保留第一个 [ANSWER] 中的内容）
        cleaned_text = clean_generated_text(generated_text, max_length=512)
        
        if cleaned_text != generated_text.strip():
            cleaned_count += 1
        
        # 保存原始输出和清洗后的结果
        test_sample = test_leaderboard[metadata['test_sample_idx']]
        collection = test_sample['task']['task_behavior_collections'][metadata['collection_idx']]
        data_item = collection['data'][metadata['data_item_idx']]
        
        processing_log_content.append(f"样本 #{idx + 1}")
        processing_log_content.append(f"Data Item: test_sample_idx={metadata['test_sample_idx']}, collection_idx={metadata['collection_idx']}, data_item_idx={metadata['data_item_idx']}, sample_idx={sample_idx}")
        processing_log_content.append("-" * 80)
        processing_log_content.append("【输入 Prompt】")
        processing_log_content.append(all_prompts[idx])
        processing_log_content.append("")
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
            # 对于每个 sample_idx，保存清洗后的文本
            data_item_samples[key][sample_idx] = cleaned_text
        else:
            empty_count += 1
            # 如果清洗后为空，标记为 None
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
    parser = argparse.ArgumentParser(description='vLLM 高性能推理 - LovinkQuestionnaire 专用')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型 checkpoint 目录')
    parser.add_argument('--scenario_path', type=str, default=None,
                       help='场景数据路径（默认从 LovinkQuestionnaire 推断）')
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
    
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='采样温度（默认: 0.9，LovinkQuestionnaire 推荐 0.8-1.0）')
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
        args.scenario_path = '/mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkQuestionnaire'
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
