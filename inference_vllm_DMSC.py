"""
使用 vLLM 进行高性能推理 - DMSC (豆瓣影评) 专用版本

vLLM 优势:
- 速度提升 2-24x (相比 HuggingFace Transformers)
- 内存效率更高 (PagedAttention + Continuous Batching)
- 支持 Tensor Parallelism (多GPU并行)
- 自动批处理优化

环境要求:
pip install vllm

使用方法:
python inference_vllm_DMSC.py \
    --checkpoint_dir outputs/DMSC_history_0221_1 \
    --ablation_config history_only \
    --num_samples 5 \
    --output_dir outputs/leaderboards/DMSC_vllm \
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
import re

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


def clean_generated_text(text: str, max_length: int = 500) -> str:
    """
    清洗 DMSC 生成的影评文本
    
    清洗逻辑:
    1. 提取 [ANSWER] 标签内容
    2. 移除指令性文本和元数据标签
    3. 移除思考过程和解释
    4. 截断到合理长度
    
    Args:
        text: 原始生成文本
        max_length: 最大输出长度（字符数，默认500）
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    original_text = text
    
    # 1. 提取 [ANSWER]...[/ANSWER] 内容（最高优先级）
    if '[ANSWER]' in text:
        if '[/ANSWER]' in text:
            # 匹配所有 [ANSWER]...[/ANSWER] 对
            answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                # 找到第一个非空的匹配
                for match in matches:
                    content = match.strip()
                    # 移除中间的 [ANSWER_FORMAT=...] 等标签
                    content = re.sub(r'\[ANSWER[^\]]*\]', '', content).strip()
                    if content and len(content) >= 2:
                        text = content
                        break
        else:
            # 如果没有找到 [/ANSWER]，提取 [ANSWER] 之后的内容
            answer_pattern = r'\[ANSWER\](.*?)(?=\[ANSWER\]|$)'
            matches = re.findall(answer_pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    content = match.strip()
                    content = re.sub(r'\[/?ANSWER[^\]]*\]', '', content).strip()
                    if content and len(content) >= 2:
                        text = content
                        break
    
    # 2. 移除残留的 ANSWER 标签
    text = re.sub(r'\[/?ANSWER[^\]]*\]', '', text).strip()
        
    # 3. 移除元数据标签
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 4. 移除对话格式标记
    text = re.sub(r'User:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    
    # 5. 移除指令性文本（先截断，再删除）
    # 这些模式会匹配并移除从指令开始到文本结尾的所有内容
    truncate_from_instruction_patterns = [
        r'注意[：:].*',
        r'请直接给出.*',
        r'用 \[ANSWER\] 和 \[/ANSWER\] 标签包裹.*',
        r'不需要解释或思考过程.*',
        r'Make sure.*formatted\..*',
        r'Your response should.*',
        r'Please verify.*',
        r'Ensure your response.*',
    ]
    
    # 先尝试截断（移除指令及其之后的所有内容）
    for pattern in truncate_from_instruction_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # 截断到指令开始的位置
            text = text[:match.start()].strip()
            break
    
    # 然后移除其他指令性文本模式
    instruction_patterns = [
        r'注意[：:].*?不需要解释或思考过程[。.]?',
        r'请直接给出.*?评价[。.]?',
        r'用 \[ANSWER\] 和 \[/ANSWER\] 标签包裹.*?[。.]?',
        r'不需要解释或思考过程[。.]?',
        r'直接输出.*?[。.]?',
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 6. 移除开头的标点符号和空白
    text = text.lstrip(r'.!?,\s\-：:')
    text = re.sub(r'^[.!?,:;\-：:]\s+', '', text)
    
    # 7. 清理重复的标点符号
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)
    
    # 8. 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 9. 移除只有标点符号的文本
    if text and len(text) <= 3 and all(c in '.。！？,，、：:' for c in text):
        text = ""
    
    # 10. 截断到最大长度（保留完整句子）
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
    
    # 11. 最终清理
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def build_inference_prompt(
    user_info: Dict[str, Any],
    movie_name: str,
    use_profile: bool = True,
    use_history: bool = True,
    max_history_reviews: int = 100
) -> str:
    """
    构建 DMSC 推理 prompt（与训练时的格式一致）
    
    对应 train_distributed_MovieReview.py 中的 build_prompt 方法
    
    Args:
        user_info: 用户信息
        movie_name: 电影名称
        use_profile: 是否使用用户画像
        use_history: 是否使用历史影评
        max_history_reviews: 最大历史影评数量（默认100，避免 prompt 过长）
    """
    parts = []
    
    # 1. 用户Profile
    if use_profile and user_info.get('user_profile'):
        profile = user_info['user_profile']
        parts.append(f"用户: {profile.get('name', 'Unknown')}")
    
    # 2. 历史影评
    if use_history and user_info.get('user_train_samples'):
        train_samples = user_info['user_train_samples']
        if train_samples:
            # 构建历史影评列表
            history_reviews = []
            for sample in train_samples:
                # 从 train.json 中提取影评数据
                task = sample.get('task', {})
                collections = task.get('task_behavior_collections', [])
                for collection in collections:
                    data_items = collection.get('data', [])
                    for data_item in data_items:
                        movie = data_item.get('continuation_prefix', '').rstrip(': ')
                        review = data_item.get('continuation', '')
                        timestamp = data_item.get('timestamp', '')
                        if movie and review:
                            history_reviews.append({
                                'movie': movie,
                                'review': review,
                                'timestamp': timestamp
                            })
            
            # 限制历史影评数量（取最近的 N 条）
            if len(history_reviews) > max_history_reviews:
                history_reviews = history_reviews[-max_history_reviews:]
            
            if history_reviews:
                parts.append(f"\n历史影评记录 ({len(history_reviews)}条):")
                for h in history_reviews:
                    # 限制每条影评的长度，避免过长
                    review_text = h['review']
                    if len(review_text) > 200:
                        review_text = review_text[:197] + "..."
                    parts.append(f"  电影《{h['movie']}》: {review_text}")
    
    # 3. 当前电影（中文提示）
    parts.append(f"\n预测用户对该电影的评价：")
    parts.append("注意：请直接给出用户对该电影的评价，用 [ANSWER] 和 [/ANSWER] 标签包裹答案内容，不需要解释或思考过程。")
    
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
    use_history: bool,
    num_samples: int,
    output_dir: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16384,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_tokens: int = 256,
    seed: int = 42,
    max_history_reviews: int = 100
):
    """
    使用 vLLM 运行 DMSC 推理
    """
    print("=" * 80)
    print("vLLM 推理配置 - DMSC (豆瓣影评)")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: DMSC (豆瓣影评)")
    print(f"数据路径: {scenario_path}")
    print(f"Ablation: {ablation_config}")
    print(f"  use_profile: {use_profile}")
    print(f"  use_history: {use_history}")
    print(f"Samples per user: {num_samples}")
    print(f"Output: {output_dir}")
    print(f"Tensor Parallel: {tensor_parallel_size} GPU(s)")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Model Length: {max_model_len}")
    print(f"Max History Reviews: {max_history_reviews}")
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
    enhanced_temperature = min(temperature, 1.0)
    enhanced_top_p = min(top_p, 0.95)
    actual_repetition_penalty = repetition_penalty if repetition_penalty > 0 else 1.1
    actual_max_tokens = min(max_tokens, 256)
    
    print(f"\n采样参数 (DMSC 优化):")
    print(f"  temperature: {enhanced_temperature} (影评生成推荐 0.9-1.0)")
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
                # 提取电影名称
                movie_name = data_item.get('continuation_prefix', '').rstrip(': ')
                
                if not movie_name:
                    continue
                
                # 为每个 data_item 生成样本
                for sample_idx in range(num_samples):
                    prompt = build_inference_prompt(
                        user_info=user_info,
                        movie_name=movie_name,
                        use_profile=use_profile,
                        use_history=use_history,
                        max_history_reviews=max_history_reviews
                    )
                    
                    all_prompts.append(prompt)
                    all_metadata.append({
                        'test_sample_idx': test_sample_idx,
                        'collection_idx': collection_idx,
                        'data_item_idx': data_item_idx,
                        'sample_idx': sample_idx,
                        'movie_name': movie_name,
                        'user_hash': user_hash
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
    examples_content.append(f"数据集: DMSC (豆瓣影评)")
    examples_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    examples_content.append(f"总样本数: {len(all_prompts)}")
    examples_content.append("=" * 80)
    examples_content.append("")
    
    num_examples = min(5, len(all_prompts))
    for i in range(num_examples):
        metadata = all_metadata[i]
        example_header = f"\n【示例 {i+1}/{num_examples}】"
        print(example_header)
        examples_content.append(example_header)
        
        separator = "-" * 80
        print(separator)
        examples_content.append(separator)
        
        # 打印元信息
        meta_info = f"用户: {metadata['user_hash'][:16]}... | 电影: {metadata['movie_name']} | Sample: {metadata['sample_idx']}"
        print(meta_info)
        examples_content.append(meta_info)
        examples_content.append("")
        
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
    data_item_samples = {}
    
    cleaned_count = 0
    empty_count = 0
    
    # 保存原始输出和清洗后的结果
    processing_log_content = []
    processing_log_content.append("=" * 80)
    processing_log_content.append("推理阶段原始输出和处理结果")
    processing_log_content.append("=" * 80)
    processing_log_content.append(f"数据集: DMSC (豆瓣影评)")
    processing_log_content.append(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    processing_log_content.append("=" * 80)
    processing_log_content.append("")
    
    # 按 data_item 和 sample_idx 分组处理
    for idx, (metadata, generated_text) in enumerate(zip(all_metadata, generated_texts)):
        key = (metadata['test_sample_idx'], metadata['collection_idx'], metadata['data_item_idx'])
        sample_idx = metadata['sample_idx']
        
        if key not in data_item_samples:
            data_item_samples[key] = {}
        
        cleaned_text = clean_generated_text(generated_text, max_length=500)
        
        if cleaned_text != generated_text.strip():
            cleaned_count += 1
        
        # 保存原始输出和清洗后的结果
        test_sample = test_leaderboard[metadata['test_sample_idx']]
        collection = test_sample['task']['task_behavior_collections'][metadata['collection_idx']]
        data_item = collection['data'][metadata['data_item_idx']]
        
        processing_log_content.append(f"样本 #{idx + 1}")
        processing_log_content.append(f"用户: {metadata['user_hash']}")
        processing_log_content.append(f"电影: {metadata['movie_name']}")
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
            if sample_idx not in data_item_samples[key]:
                data_item_samples[key][sample_idx] = cleaned_text
        else:
            empty_count += 1
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
        
        # 按 sample_idx 顺序填充 continuations
        final_continuations = []
        for sample_idx in range(num_samples):
            if sample_idx in samples_dict and samples_dict[sample_idx]:
                final_continuations.append(samples_dict[sample_idx])
            else:
                if valid_answers:
                    copy_idx = len(final_continuations) % len(valid_answers)
                    final_continuations.append(valid_answers[copy_idx])
                    copied_count += 1
                else:
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
        'dataset': 'DMSC',
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
    parser = argparse.ArgumentParser(description='vLLM 高性能推理 - DMSC (豆瓣影评) 专用')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型 checkpoint 目录')
    parser.add_argument('--scenario_path', type=str, default=None,
                       help='场景数据路径（默认自动推断）')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history', 'profile_only', 'history_only'],
                       help='消融实验配置')
    
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个电影生成的影评数（默认: 5）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor Parallel 大小（使用多少张 GPU，默认: 1）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU 内存利用率（0.0-1.0，默认: 0.9）')
    parser.add_argument('--max_model_len', type=int, default=16384,
                       help='最大模型序列长度（默认: 16384）')
    
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='采样温度（默认: 0.9，影评生成推荐 0.9-1.0）')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p 采样（默认: 0.9）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k 采样（默认: 50）')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='重复惩罚（默认: 1.1）')
    parser.add_argument('--max_tokens', type=int, default=256,
                       help='最大生成 token 数（默认: 256）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--max_history_reviews', type=int, default=100,
                       help='最大历史影评数量（默认: 100，避免 prompt 过长）')
    
    args = parser.parse_args()
    
    # 推断 scenario_path
    if args.scenario_path is None:
        args.scenario_path = '/mnt/parallel/GIDigitalTwinBench/RealSelf/DMSC'
        print(f"自动推断数据路径: {args.scenario_path}")
    
    # 从 ablation_config 推断配置
    config_mapping = {
        'profile_and_history': (True, True),
        'profile_only': (True, False),
        'history_only': (False, True),
    }
    use_profile, use_history = config_mapping[args.ablation_config]
    
    # 运行推理
    run_inference_vllm(
        checkpoint_dir=args.checkpoint_dir,
        scenario_path=args.scenario_path,
        ablation_config=args.ablation_config,
        use_profile=use_profile,
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
        max_history_reviews=args.max_history_reviews
    )


if __name__ == '__main__':
    main()
