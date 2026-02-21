"""
使用 vLLM 进行高性能推理 - RealPersonaChat 专用版本

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
python inference_vllm_RealPersonaChat.py \
    --checkpoint_dir outputs/RealPersonaChat_8B_profile_context_sampled_seed42 \
    --ablation_config profile_and_context \
    --num_samples 5 \
    --output_dir outputs/leaderboards/RealPersonaChat_vllm_8B \
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


def clean_generated_text(text: str, max_length: int = 300) -> str:
    """
    清洗 RealPersonaChat 生成的文本（专门针对日语对话）
    
    清洗逻辑（新结构）：
    0. 结构截断 - 在结构标记处截断，移除后续内容
    1. 删除整行污染 - 移除指令性文本、元数据标签等
    2. 提取有效块 - 从特殊格式中提取实际内容
    3. 连续重复压缩 - 去重，移除重复内容
    4. 有效句筛选 - 提取第一个有效句子
    5. Unicode 规范化 - 规范化字符和格式
    6. 最终长度限制 - 截断到最大长度
    
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
    
    # ============================================
    # 步骤 0: 模板识别和提取（不进行清洗，直接识别模板并提取）
    # 先识别是哪种模板，然后针对不同模板采用不同策略
    # ============================================
    
    original_text = text  # 保存原始文本
    template_type = None
    extracted_text = None
    answer_extracted = False
    
    # 0.1. 识别 [ANSWER]...[/ANSWER] 模板（最高优先级）
    if '[ANSWER]' in text:
        # 先尝试匹配完整的 [ANSWER]...[/ANSWER]
        if '[/ANSWER]' in text:
            # 匹配所有 [ANSWER]...[/ANSWER] 对，找到第一个非空的
            answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                # 找到第一个非空的匹配（去除空白和标签后仍有内容）
                for match in matches:
                    content = match.strip()
                    # 移除中间的 [ANSWER_FORMAT=...] 等标签
                    content = re.sub(r'\[ANSWER[^\]]*\]', '', content).strip()
                    if content and len(content) >= 3:
                        template_type = 'ANSWER'
                        extracted_text = content
                        answer_extracted = True
                        break
                # 如果所有匹配都为空，使用最后一个匹配（可能包含内容）
                if not extracted_text and matches:
                    last_match = matches[-1].strip()
                    last_match = re.sub(r'\[ANSWER[^\]]*\]', '', last_match).strip()
                    if last_match:
                        template_type = 'ANSWER'
                        extracted_text = last_match
                        answer_extracted = True
        else:
            # 如果没有找到 [/ANSWER]，但找到了 [ANSWER]，提取第一个 [ANSWER] 到下一个 [ANSWER] 或文本末尾
            answer_pattern = r'\[ANSWER\](.*?)(?=\[ANSWER\]|$)'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                # 找到第一个非空的匹配
                for match in matches:
                    content = match.strip()
                    content = re.sub(r'\[/?ANSWER[^\]]*\]', '', content).strip()
                    if content and len(content) >= 3:
                        template_type = 'ANSWER'
                        extracted_text = content
                        answer_extracted = True
                        break
    
    # 0.2. 识别 答え: 模板
    elif '答え:' in text or '答え：' in text:
        template_type = 'KOTAE'
        # 提取第一个 答え: 后的内容
        pattern = r'答え[：:]\s*(.*?)(?=答え[：:]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content = matches[0].strip()
            # 移除指令性文本
            content = re.sub(r'ユーザーの次のメッセージを予測する[：:]?', '', content)
            content = re.sub(r'注意[：:].*?直接出力してください[。.]?', '', content, flags=re.DOTALL)
            content = content.strip()
            
            # 提取第一部分（用空格分隔，优先双空格）
            parts = re.split(r'\s{2,}', content)  # 先尝试双空格或更多空格
            if len(parts) == 1:
                parts = content.split(' ', 1)  # 如果没有双空格，按单空格分割，只分割一次
            
            if parts and parts[0].strip():
                first_part = parts[0].strip()
                # 移除末尾可能残留的指令性文本
                first_part = re.sub(r'\s*ユーザーの次のメッセージを予測する[：:]?.*$', '', first_part, flags=re.DOTALL)
                first_part = first_part.strip()
                if len(first_part) >= 3:
                    extracted_text = first_part
                else:
                    extracted_text = content  # 如果第一部分太短，使用整个内容
            else:
                extracted_text = content
    
    # 0.3. 识别 （例）→「...」模板
    elif '（例）' in text or re.search(r'→[「「]', text):
        template_type = 'EXAMPLE'
        # 提取第一个「...」之间的内容
        quote_pattern = r'「([^」]+)」'
        quote_matches = re.findall(quote_pattern, text)
        if quote_matches:
            extracted_text = quote_matches[0].strip()
        else:
            # 如果没有找到引号，使用箭头后的内容
            example_pattern = r'（例）\s*'
            if re.search(example_pattern, text):
                text_temp = re.sub(example_pattern, '', text)
            else:
                text_temp = text
            
            arrow_matches = re.findall(r'→\s*[「「]?([^」」\n→]+?)[」」]?(?=\s*→|\s*$|\s*\n)', text_temp, re.MULTILINE)
            if not arrow_matches:
                arrow_matches = re.findall(r'→\s*([^\n→]+)', text_temp)
            
            if arrow_matches:
                for match in arrow_matches:
                    cleaned_match = match.strip()
                    cleaned_match = re.sub(r'^[「「]|[」」]$', '', cleaned_match).strip()
                    if cleaned_match and len(cleaned_match) >= 3:
                        extracted_text = cleaned_match
                        break
                else:
                    if arrow_matches:
                        first_match = arrow_matches[0].strip()
                        first_match = re.sub(r'^[「「]|[」」]$', '', first_match).strip()
                        extracted_text = first_match if first_match else None
    
    # 0.4. 如果识别到模板并提取成功，使用提取的内容
    if extracted_text:
        text = extracted_text
    else:
        # 如果没有识别到模板，继续使用原有逻辑
        text = original_text
    
    # ============================================
    # 步骤 1: 删除整行污染（对提取后的内容进行清理）
    # 移除指令性文本、元数据标签、格式标记等
    # ============================================
    
    # 1.0. 如果已从模板提取，截断到第一个角色标识或提示信息之前
    if template_type:
        # 先移除残留的 [ANSWER] 和 [/ANSWER] 标签
        text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
        
        truncation_patterns = [
            r'\s*Assistant:\s*',
            r'\s*User:\s*',
            r'\s*AIの回复[：:]?',
            r'\s*AIの回复は[：:]?',
            r'\s*以下を参考にしてください[：:]?\s*',
            r'\s*以下を参考に[：:]?\s*',
            r'\s*答え[：:]\s*',
            r'\[RECENT_DIALOGUE\]',
            r'\[SYSTEM_LOGIC\]',
            r'ユーザーの次のメッセージを予測する[：:]?',
            r'この回答は.*?',  # 移除解释性文本
            r'解説[：:].*?',  # 移除解释性文本（解説：）
            r'また、出力は日本語で行ってください[。.]?',
            r'答えは.*?短くしてください[。.]?',  # 移除指令性文本
            r'1行に収まるよう.*?[。.]?',  # 移除指令性文本
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
        
        # 移除末尾的解释性文本（如"解説： - ..."）
        text = re.sub(r'\s*解説[：:].*$', '', text, flags=re.DOTALL).strip()
    
    # 1.1. 移除元数据标签
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 1.2. 移除对话格式标记
    text = re.sub(r'User:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    
    # 1.3. 移除所有指令性文本模式
    instruction_patterns = [
        r'それも1行にまとめること[。.]?',
        r'それも1行にまとめ[、,]改行コード無しで出力してください[。.]?',
        r'また、出力は1行でお願いします[。.]?',  # 新增：また、出力は1行でお願いします。
        r'出力は1行で[。.]?',  # 新增：出力は1行で
        r'1行でお願いします[。.]?',  # 新增：1行でお願いします
        r'【出力フォーマット】',
        r'→予測メッセージ[：:]?',
        r'^答え[：:]\s*$',
        r'以下を参考にしてください[：:]?\s*',
        r'以下を参考に[：:]?\s*',
        r'参考例\d+[：:]\s*',
        r'参考例[：:]\s*',
        r'ユーザー[：:]\s*',
        r'ユーザーは.*?[。.]',
        r'出力形式は.*?出力してください[。.]?',
        r'出力形式[：:].*?出力してください[。.]?',
        r'回答[：:]\s*',
        r'また、出力は日本語で行ってください[。.]?\s*',
        r'また、出力は日本語で[。.]?\s*',
        r'出力は日本語で[。.]?\s*',
        r'XXXX',
        r'（\d+字程度で）',
        r'（\d+文字以上）',
        r'長い文章より',
        r'このメッセージの特徴',
        r'選択肢',
        r'ア[：:]',
        r'ユーザーの生成メッセージ',
        r'この説明に改善点',
        r'したがって[、,]',
        r'このメッセージは会話の中で',
        r'このタスクにおける正解',
        r'この思考プロセス',
        r'前の会話の続きとして',
        r'ご指定の寸法',
        r'ご指摘がありますか',
        r'お返事遅くなり',
        r'Also,\s*',
        r'分析して[、,]',
        r'ユーザーの次のメッセージを予測する[：:]?',
        r'注意[：:].*?直接出力してください[。.]?',
        r'また、出力は英文とならないよう注意してください[。.]?',
        r'←同じことを言ってはいけません[。.]?',
        r'同じことを言ってはいけません[。.]?',
        r'不需要思考过程或解释[，,]?只需直接输出[。.]?',
        r'不需要思考过程[，,]?.*?直接输出[。.]?',
        r'思考过程.*?不需要[。.]?',
        r'思考過程.*?不要[。.]?',  # 新增：思考過程や説明は不要です
        r'思考過程や説明は不要[。.]?',  # 新增：思考過程や説明は不要です
        r'もう一度言いますが.*?不要[。.]?',  # 新增：もう一度言いますが、思考過程や説明は不要です
        r'思考過程や説明は不要です[。.]?',  # 新增：思考過程や説明は不要です
        r'シンプルで短いメッセージを心がけてください[。.]?',  # 新增：シンプルで短いメッセージを心がけてください
        r'短いメッセージを心がけて[。.]?',  # 新增：短いメッセージを心がけて
        r'メッセージを心がけてください[。.]?',  # 新增：メッセージを心がけてください
        r'只需直接输出.*?消息[。.]?',
        r'直接输出用户.*?消息[。.]?',
        r'用户接下来的消息[。.]?',  # 新增：用户接下来的消息。
        r'用户.*?消息[。.]?',  # 新增：用户...消息
        r'日本語で答えてください[。.]?',
        r'日本語で.*?答えて[。.]?',
        r'答えてください[。.]?',
        r'この回答だと.*?避けた方がいい[。.]?',
        r'会話が終わってしまう[。.]?',
        r'ただし、タグや記号は含めないでください[。.]?',  # 新增：ただし、タグや記号は含めないでください。
        r'タグや記号は含めない[。.]?',  # 新增：タグや記号は含めない
        r'記号は含めないでください[。.]?',  # 新增：記号は含めないでください
        r'答えは.*?短くしてください[。.]?',  # 新增：答えは1行に収まるよう短くしてください
        r'1行に収まるよう.*?[。.]?',  # 新增：1行に収まるよう短くしてください
        r'短くしてください[。.]?',  # 新增：短くしてください
        r'解説[：:].*?',  # 新增：解説： - ...
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 1.4. 移除箭头标记和箭头后的指令性文本
    # 处理 "「...」←..." 格式：只保留引号内的内容
    if '「' in text and '←' in text:
        quote_arrow_pattern = r'「([^」]+)」\s*←.*'
        match = re.search(quote_arrow_pattern, text)
        if match:
            text = f'「{match.group(1)}」'
    text = re.sub(r'→[^\n]*', '', text)
    text = re.sub(r'^→\s*', '', text)
    text = re.sub(r'←[^\n]*', '', text)  # 移除所有箭头后的内容
    
    # 1.5. 移除所有标签（方括号和日文方括号），但保留 [ANSWER] 和 [/ANSWER] 标签
    # 先移除其他标签，保留 [ANSWER] 和 [/ANSWER] 以便后续提取
    # 匹配 [XXX] 但不匹配 [ANSWER] 或 [/ANSWER]
    text = re.sub(r'\[(?!/?ANSWER\])[^\]]*\]', '', text)
    text = re.sub(r'【[^】]*】', '', text)
    
    # 1.6. 移除 LaTeX 格式标记
    text = re.sub(r'\\boxed\{[^}]*\}', '', text)
    text = re.sub(r'→\\boxed\{[^}]*\}\s*', '', text)
    
    # 1.7. 移除括号注释
    text = re.sub(r'（[^）]*）', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 1.8. 移除分隔线
    text = re.sub(r'-{10,}', '', text)
    text = re.sub(r'[-\s]{20,}', '', text)
    
    # ============================================
    # 步骤 2: 提取有效块（如果未在步骤0中识别到模板，则使用备用提取逻辑）
    # 从特殊格式中提取实际内容
    # ============================================
    
    # 如果已经在步骤0中识别到模板并提取，跳过此步骤
    if not template_type:
        # 2.0. 处理 [ANSWER] 和 [/ANSWER] 格式（备用逻辑）
        if '[ANSWER]' in text and '[/ANSWER]' in text:
            answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                text = matches[0].strip()
                text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
                answer_extracted = True
    
    # 2.1. 处理 [AI_RESPONSE] 格式（仅在未识别到模板时）
    if not template_type:
        if '[AI_RESPONSE]' in text:
            ai_response_pattern = r'\[AI_RESPONSE\](.*?)(?=\[AI_RESPONSE\]|\[RECENT_DIALOGUE\]|$)'
            matches = re.findall(ai_response_pattern, text, re.DOTALL)
            if matches:
                text = matches[0].strip()
        
        # 2.2. 处理 [RESPONSE] 格式
        if '[RESPONSE]' in text:
            response_pattern = r'\[RESPONSE\]\s*(.*?)(?=\[RESPONSE\]|$)'
            matches = re.findall(response_pattern, text, re.DOTALL)
            if matches:
                # 找到第一个包含实际内容的块（至少5个字符）
                for match in matches:
                    content = match.strip()
                    # 移除标签后检查是否还有实际内容
                    temp_content = re.sub(r'\[[^\]]*\]', '', content)
                    temp_content = re.sub(r'【[^】]*】', '', temp_content)
                    temp_content = temp_content.strip()
                    if len(temp_content) >= 5:
                        text = content
                        break
                else:
                    text = matches[0].strip() if matches else text
    
    # 2.3. 处理 答え: 格式（仅在未识别到模板时，备用逻辑）
    if not template_type and ('答え:' in text or '答え：' in text):
        pattern = r'答え[：:]\s*(.*?)(?=答え[：:]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # 提取第一个匹配的内容
            content = matches[0].strip()
            # 移除指令性文本
            content = re.sub(r'ユーザーの次のメッセージを予測する[：:]?', '', content)
            content = re.sub(r'注意[：:].*?直接出力してください[。.]?', '', content, flags=re.DOTALL)
            content = content.strip()
            
            # 提取第一部分（用空格分隔，取第一个非空部分）
            # 先按双空格或更多空格分割，如果没有则按单空格分割
            parts = re.split(r'\s{2,}', content)  # 先尝试双空格或更多空格
            if len(parts) == 1:
                parts = content.split(' ', 1)  # 如果没有双空格，按单空格分割，只分割一次
            
            if parts and parts[0].strip():
                # 提取第一部分
                first_part = parts[0].strip()
                # 移除末尾可能残留的指令性文本
                first_part = re.sub(r'\s*ユーザーの次のメッセージを予測する[：:]?.*$', '', first_part, flags=re.DOTALL)
                first_part = first_part.strip()
                if len(first_part) >= 3:
                    text = first_part
                else:
                    text = content  # 如果第一部分太短，使用整个内容
            else:
                text = content
    
    # 2.4. 处理 →予測メッセージ：格式
    if '→予測メッセージ：' in text:
        pattern = r'→予測メッセージ：([^\n→←]+)'
        matches = re.findall(pattern, text)
        if matches:
            text = matches[-1].strip()
    
    # 2.5. 处理 [OUTPUT] 和 [OUTPUT_SCORE] 格式
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
    
    # 2.6. 处理 [RESPONSE_1], [RESPONSE_2] 等格式
    if re.search(r'\[RESPONSE_\d+\]', text):
        response_pattern = r'\[RESPONSE_\d+\]\s*(.*?)(?=\[RESPONSE_\d+\]|$)'
        matches = re.findall(response_pattern, text, re.DOTALL)
        if matches:
            text = matches[0].strip()
    
    # 2.7. 处理 JSON 数组格式
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
    
    # 2.8. 处理 [RESULT] 格式
    if '[RESULT]' in text:
        result_match = re.search(r'\[RESULT\](.*?)(?:\[SYSTEM_LOGIC\]|$)', text, re.DOTALL)
        if result_match:
            result_text = result_match.group(1).strip()
            result_text = re.sub(r'\([^)]*\)', '', result_text)
            text = result_text.strip()
    
    # 2.9. 处理示例列表模式（仅在未识别到模板时，备用逻辑）
    if not template_type and ('（例）' in text or re.search(r'→[「「]', text)):
        # 先尝试提取第一个「...」之间的内容
        quote_pattern = r'「([^」]+)」'
        quote_matches = re.findall(quote_pattern, text)
        if quote_matches:
            # 提取第一个引号中间的所有内容
            text = quote_matches[0].strip()
        else:
            # 如果没有找到引号，使用原有的逻辑
            example_pattern = r'（例）\s*'
            if re.search(example_pattern, text):
                text = re.sub(example_pattern, '', text)
            
            # 提取所有箭头后的内容
            arrow_matches = re.findall(r'→\s*[「「]?([^」」\n→]+?)[」」]?(?=\s*→|\s*$|\s*\n)', text, re.MULTILINE)
            if not arrow_matches:
                arrow_matches = re.findall(r'→\s*([^\n→]+)', text)
            
            if arrow_matches and len(arrow_matches) > 1:
                # 只保留第一个非空的示例
                for match in arrow_matches:
                    cleaned_match = match.strip()
                    cleaned_match = re.sub(r'^[「「]|[」」]$', '', cleaned_match).strip()
                    if cleaned_match and len(cleaned_match) >= 3 and \
                       not re.match(r'^(また、|出力|注意|参考例|ユーザー|また、出力)', cleaned_match):
                        text = cleaned_match
                        break
                else:
                    if arrow_matches:
                        first_match = arrow_matches[0].strip()
                        first_match = re.sub(r'^[「「]|[」」]$', '', first_match).strip()
                        text = first_match if first_match else text
    
    # ============================================
    # 步骤 3: 连续重复压缩
    # 去重，移除重复内容
    # ============================================
    
    # 3.1. 处理引号内的重复（如 "「...」 「...」"）
    if '「' in text and text.count('「') >= 2:
        all_quotes = re.findall(r'「([^」]+)」', text)
        if len(all_quotes) >= 2:
            seen = set()
            unique_quotes = []
            for quote in all_quotes:
                quote_clean = quote.strip()
                if quote_clean and quote_clean not in seen:
                    seen.add(quote_clean)
                    unique_quotes.append(quote_clean)
            
            if len(unique_quotes) < len(all_quotes):
                if unique_quotes:
                    text = f'「{unique_quotes[0]}」'
                else:
                    text = re.sub(r'[「」]', '', text)
    
    # 3.2. 移除明显的重复模式（如 "「...」 →「...」"）
    text = re.sub(r'「([^」]+)」\s*→\s*「\1」', r'「\1」', text)
    text = re.sub(r'「([^」]+)」\s*→\s*\1', r'\1', text)
    
    # 3.3. 按日语句子分隔符分割并去重
    sentences = re.split(r'([。！？])', text)
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
            
            sent_clean = re.sub(r'[「」→\s]', '', sent)
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
            text = ''.join(unique_sentences)
        elif len(sentences) > 2:
            text = (sentences[0] + sentences[1]).strip() if len(sentences) > 1 else sentences[0].strip()
    
    # 3.4. 检测完全重复的子串（长度>=15字符）
    if len(text) > 30:
        for substr_len in range(min(100, len(text) // 2), 14, -1):
            found_repeat = False
            for start_pos in range(min(50, len(text) - substr_len)):
                substr = text[start_pos:start_pos + substr_len]
                substr_normalized = re.sub(r'[「」\s。！？、，]', '', substr)
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
    
    # 4.1. 智能提取：如果文本包含指令性前缀+冒号，提取冒号后的内容
    colon_extraction_patterns = [
        (r'注意[：:]\s*(.+?)(?=注意[：:]|$)', r'\1'),
        (r'以下のフォーマットに沿って出力してください[：:]\s*(.+?)(?=以下のフォーマット|$)', r'\1'),
        (r'出力形式は.*?出力してください[：:]\s*(.+?)(?=出力形式|$)', r'\1'),
        (r'回答形式例[：:]\s*(.+?)(?=回答形式例|$)', r'\1'),
        (r'参考例\d+[：:]\s*(.+?)(?=参考例\d+[：:]|$)', r'\1'),
        (r'参考例[：:]\s*(.+?)(?=参考例[：:]|$)', r'\1'),
        (r'ユーザー[：:]\s*(.+?)(?=ユーザー[：:]|$)', r'\1'),
        (r'回答[：:]\s*(.+?)(?=回答[：:]|$)', r'\1'),
        (r'答え[：:]\s*(.+?)(?=答え[：:]|$)', r'\1'),
    ]
    
    for pattern, replacement in colon_extraction_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) >= 3:
                text = extracted
                break
    
    # 4.2. 移除所有残留的提示信息和标签
    # 先移除所有残留的 [ANSWER] 和 [/ANSWER] 标签
    text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
    
    text = re.sub(r'回答形式例[：:]\s*', '', text)
    text = re.sub(r'以下を参考にしてください[：:]?\s*', '', text)
    text = re.sub(r'参考例\d+[：:]\s*', '', text)
    text = re.sub(r'参考例[：:]\s*', '', text)
    text = re.sub(r'ユーザー[：:]\s*', '', text)
    text = re.sub(r'ユーザーは.*?[。.]', '', text)
    text = re.sub(r'出力形式は.*?出力してください[。.]?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'回答[：:]\s*', '', text)
    text = re.sub(r'答え[：:]\s*', '', text)
    text = re.sub(r'答えは.*?短くしてください[。.]?', '', text)  # 移除指令性文本
    text = re.sub(r'1行に収まるよう.*?[。.]?', '', text)  # 移除指令性文本
    text = re.sub(r'ユーザーの次のメッセージを予測する[：:]?\s*', '', text)
    text = re.sub(r'注意[：:].*?直接出力してください[。.]?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'また、出力は英文とならないよう注意してください[。.]?\s*', '', text)
    text = re.sub(r'また、出力は日本語で行ってください[。.]?\s*', '', text)
    text = re.sub(r'また、出力は日本語で[。.]?\s*', '', text)
    text = re.sub(r'出力は日本語で[。.]?\s*', '', text)
    text = re.sub(r'^→\s*', '', text)
    text = re.sub(r'→[^\n]*', '', text)
    text = re.sub(r'\\boxed\{[^}]*\}', '', text)
    text = re.sub(r'^[：:]\s*', '', text)
    # 移除解释性文本（解説：）
    text = re.sub(r'\s*解説[：:].*$', '', text, flags=re.DOTALL).strip()
    # 移除残留的 ANSWER 单词（不完整的标签）
    text = re.sub(r'\s*ANSWER\s*$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s*ANSWE\s*$', '', text, flags=re.IGNORECASE).strip()
    
    # 4.3. 处理项目符号列表
    if re.match(r'^[・•]\s*', text):
        items = re.split(r'[・•]\s*', text)
        found_valid = False
        instruction_keywords = [
            '日本語で', 'アルファベット', '改行コード', '他の言葉', '回答は',
            '前後の文脈', '1行で完了', '形式例', '半角英数字', '小文字のみ',
            '記号を許容', '含まれない', '30字以内'
        ]
        
        for item in items:
            item = item.strip()
            if not item or len(item) < 5:
                continue
            
            is_instruction = False
            for keyword in instruction_keywords:
                if keyword in item:
                    is_instruction = True
                    break
            
            if not is_instruction and len(item) >= 5:
                text = item
                found_valid = True
                break
        
        if not found_valid:
            text = ""
    
    # 4.4. 提取第一个有效句子（如果已从 [ANSWER] 标签提取，则跳过此步骤）
    if not answer_extracted:
        japanese_sentences = re.split(r'([。！？])', text)
        if len(japanese_sentences) > 2:
            first_valid_sentence = None
            for i in range(0, len(japanese_sentences) - 1, 2):
                if i + 1 < len(japanese_sentences):
                    sent = (japanese_sentences[i] + japanese_sentences[i + 1]).strip()
                else:
                    sent = japanese_sentences[i].strip()
                
                if sent and len(sent) > 3:
                    if not re.match(r'^（\d+字程度で）', sent) and \
                       not sent.startswith('長い文章') and \
                       not sent.startswith('Also,') and \
                       not sent.startswith('それも') and \
                       not sent.startswith('答え') and \
                       not sent.startswith('注意') and \
                       not sent.startswith('ユーザーの次のメッセージ') and \
                       not sent.startswith('以下を参考に') and \
                       not sent.startswith('参考例') and \
                       not sent.startswith('ユーザー') and \
                       not sent.startswith('出力形式') and \
                       not sent.startswith('回答') and \
                       not sent.startswith('→') and \
                       not sent.startswith('←') and \
                       not sent.startswith('User:') and \
                       not sent.startswith('Assistant:') and \
                       not sent.startswith('日本語で') and \
                       not '答えてください' in sent:
                        first_valid_sentence = sent
                        break
            
            if first_valid_sentence:
                text = first_valid_sentence
            elif len(japanese_sentences) > 2:
                for i in range(0, len(japanese_sentences) - 1, 2):
                    if i + 1 < len(japanese_sentences):
                        sent = (japanese_sentences[i] + japanese_sentences[i + 1]).strip()
                    else:
                        sent = japanese_sentences[i].strip()
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
    # 移除转义引号
    text = re.sub(r'^\\?"', '', text)  # 移除开头的 \" 或 "
    text = re.sub(r'^\\?\'', '', text)  # 移除开头的 \' 或 '
    text = text.lstrip(r'.!?,\s\-')
    text = re.sub(r'^[.!?,:;\-]\s+', '', text)
    # 再次确保移除开头的冒号（全角和半角）
    text = re.sub(r'^[：:]\s*', '', text)
    if text.startswith('-'):
        text = text.lstrip('-').lstrip()
    
    # 5.3. 清理重复的标点符号
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    
    # 5.4. 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 5.5. 移除只有标点符号的文本
    if text and len(text) <= 3 and all(c in '.。！？,，、：:' for c in text):
        text = ""
    
    # 5.6. 移除分隔线（如果还有残留）
    if re.match(r'^[-_\s]{5,}$', text):
        text = ""
    
    # 5.7. 检测英文输出（RealPersonaChat 应该是日文）
    if text and len(text) > 10:
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF')
        total_chars = len([c for c in text if c.isalnum() or c in '。！？、，'])
        if total_chars > 0:
            japanese_ratio = japanese_chars / total_chars
            if japanese_ratio < 0.3:
                english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
                if len(english_words) > len(text.split()) * 0.5:
                    text = ""
    
    # ============================================
    # 步骤 6: 最终长度限制
    # 截断到最大长度
    # ============================================
    
    # 6.1. 截断过长文本（保留完整的句子）
    if len(text) > max_length:
        truncated = text[:max_length]
        last_punct = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
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
    text = re.sub(r'^\\?"', '', text)  # 移除开头的 \" 或 "
    text = re.sub(r'^\\?\'', '', text)  # 移除开头的 \' 或 '
    text = re.sub(r'^[：:]\s*', '', text)  # 再次确保移除开头的冒号
    
    # 6.4. 移除日文引号标记（如果只有开头引号没有闭合）
    if text.startswith('「') and '」' not in text:
        text = text[1:].lstrip()
    if text.endswith('」') and '「' not in text:
        text = text[:-1].rstrip()
    
    # 6.5. 再次清理空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 6.6. 确保结果不为空
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
    构建 RealPersonaChat 推理 prompt（与训练时的格式一致）
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
    parts.append("RealPersonaChatデータセットにおけるユーザーの過去の会話データに基づき、当該ユーザーの会話行動パターンをシミュレートする：")
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
    parts.append("ユーザーの次のメッセージを予測する：")
    
    # 6. 添加输出要求说明（与训练时保持一致，使用 [ANSWER] 标签）
    parts.append("")
    parts.append("注意：思考過程や説明は不要です。ユーザーの次のメッセージのみを [ANSWER] と [/ANSWER] の間で出力してください。")
    
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
    max_tokens: int = 512,
    seed: int = 42,
    max_context_turns: int = 15,
    max_chars_per_turn: int = 300
):
    """
    使用 vLLM 运行推理
    """
    print("=" * 80)
    print("vLLM 推理配置 - RealPersonaChat")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: RealPersonaChat")
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
    # RealPersonaChat 需要更稳定的输出，降低 temperature 和 top_p，增加 repetition_penalty
    # 这样可以减少乱码、重复和思考过程
    enhanced_temperature = min(temperature, 0.8)  # 降低 temperature 以提高稳定性（默认 0.7-0.8）
    enhanced_top_p = min(top_p, 0.9)  # 降低 top_p 以提高稳定性（默认 0.85-0.9）
    # 使用传入的 repetition_penalty，如果没有传入则使用默认值 1.1
    actual_repetition_penalty = repetition_penalty if repetition_penalty > 0 else 1.1
    actual_max_tokens = min(max_tokens, 150)  # RealPersonaChat 限制到 150 tokens
    
    print(f"\n采样参数 (RealPersonaChat 优化):")
    print(f"  temperature: {enhanced_temperature} (降低以提高稳定性，减少乱码)")
    print(f"  top_p: {enhanced_top_p} (降低以提高稳定性，减少乱码)")
    print(f"  top_k: {top_k}")
    print(f"  repetition_penalty: {actual_repetition_penalty} (增加以减少重复和乱码)")
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
    examples_content.append(f"数据集: RealPersonaChat")
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
    data_item_samples = {}  # key: (test_sample_idx, collection_idx, data_item_idx), value: list of (sample_idx, cleaned_text)
    
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
    parser = argparse.ArgumentParser(description='vLLM 高性能推理 - RealPersonaChat 专用')
    
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
                       help='采样温度（默认: 0.7，RealPersonaChat 推荐 0.7-0.8，降低可减少乱码）')
    parser.add_argument('--top_p', type=float, default=0.85,
                       help='Top-p 采样（默认: 0.85，RealPersonaChat 推荐 0.85-0.9，降低可减少乱码）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k 采样（默认: 50）')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='重复惩罚（默认: 1.1，RealPersonaChat 推荐 1.05-1.15，增加可减少重复和乱码）')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='最大生成 token 数（默认: 512）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    parser.add_argument('--max_context_turns', type=int, default=15,
                       help='Context 最大对话轮次数（默认: 15）')
    parser.add_argument('--max_chars_per_turn', type=int, default=300,
                       help='每轮对话最大字符数（默认: 300）')
    
    args = parser.parse_args()
    
    # 推断 scenario_path
    if args.scenario_path is None:
        args.scenario_path = '/mnt/parallel/GIDigitalTwinBench/IdealSelf/RealPersonaChat'
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
