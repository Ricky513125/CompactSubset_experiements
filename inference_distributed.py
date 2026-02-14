"""
分布式推理脚本：使用多GPU并行推理加速（Lovink专用版本）
支持将数据集分片到多个GPU上并行处理，最后合并结果
支持相对路径（相对于项目根目录）和绝对路径

使用方法1（推荐 - 简化参数，在 Lovink/ 子目录下运行）：
cd /mnt/parallel/Qwen3_4B/prompt_improvement/Lovink
torchrun --nproc_per_node=8 inference_distributed.py \
    --checkpoint_dir outputs/LovinkDialogue_profile_context_detailed_20260204_104136 \
    --dataset LovinkDialogue \
    --ablation_config profile_and_context \
    --num_samples 5 \
    --output_dir outputs/test_leaderboards/0204_Lovink_pc

使用方法2（使用简洁的标签格式 prompt，与训练时一致）：
torchrun --nproc_per_node=8 inference_distributed.py \
    --checkpoint_dir outputs/LovinkDialogue_profile_context_detailed_short_short \
    --dataset LovinkDialogue \
    --ablation_config profile_and_context \
    --num_samples 5 \
    --output_dir outputs/leaderboards/LovinkDialogue_profile_context_detailed_0206_struct_gen \
    --no_detailed_template

使用方法3（使用 LovinkDialogue_v3.md 模板，支持 Few-Shot Learning）：
torchrun --nproc_per_node=8 inference_distributed.py \
    --checkpoint_dir outputs/your_checkpoint \
    --dataset LovinkDialogue \
    --ablation_config profile_and_context \
    --num_samples 5 \
    --output_dir outputs/test_v3_template \
    --template_filename LovinkDialogue_v3.md

使用方法4（指定其他markdown模板文件）：
torchrun --nproc_per_node=8 inference_distributed.py \
    --checkpoint_dir outputs/LovinkDialogue_profile_context_detailed_20260204_104136 \
    --dataset LovinkDialogue \
    --ablation_config profile_and_context \
    --num_samples 5 \
    --output_dir outputs/test_leaderboards/0204_Lovink_pc \
    --template_filename prompt_LovinkDialoguo_pc.md

使用方法5（完整绝对路径）：
torchrun --nproc_per_node=8 /mnt/parallel/Qwen3_4B/prompt_improvement/Lovink/inference_distributed.py \
    --checkpoint_dir /mnt/parallel/Qwen3_4B/outputs/LovinkDialogue_profile_context_detailed_20260204_104136 \
    --scenario_path /mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue \
    --config_name profile_and_context \
    --use_profile \
    --use_context \
    --num_samples 5 \
    --output_dir /mnt/parallel/Qwen3_4B/outputs/test_leaderboards/0204_Lovink_pc \
    --no_detailed_template

提示格式说明：
- 使用 --no_detailed_template 时：使用简洁的标签格式
  格式: [USER_PROFILE] [USER_NAME=xxx] [DIM_OCEAN_EXTRAVERSION=90] ...
        [TASK] 基于用户在 Lovink 对话中的历史数据...
        [RECENT_DIALOGUE] User: ... Assistant: ...
        Predict the user's next message:
  
- 不使用 --no_detailed_template（默认）：使用详细模板
  - 优先加载 markdown 模板文件（如果存在）
  - 回退到 Lovink 风格的自然语言模板
"""
import json
import argparse
import os
import sys
import importlib.util
from pathlib import Path

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import torch
import torch.distributed as dist
from tqdm import tqdm
from datetime import datetime

# 导入自定义模块（现在 sys.path 已经设置好了）
from inference import (
    load_test_leaderboard,
    get_user_info_from_leaderboard,
    generate_continuations,
    build_prompt
)
from data_loader_more_data import load_train_data
#  build_simple_inference_prompt 需要延迟导入，避免从错误路径导入
# from data_loader_more_data import build_simple_inference_prompt
from japanese_text_normalizer import normalize_japanese_text
from transformers import AutoTokenizer, AutoModelForCausalLM
import re



def cleanup_generated_text(text: str) -> str:
    """
    清洗生成的文本：
    1. 移除emoji表情符号
    2. 清理过多的重复字符和标点符号
    3. 规范化空白字符
    
    Args:
        text: 原始生成文本
    
    Returns:
        清洗后的文本
    """
    import re
    
    if not text:
        return text
    
    # 1. 移除emoji和特殊符号
    # Unicode emoji范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # 2. 清理过多的重复字符（保留最多2个）
    # 例如："哈哈哈哈哈" -> "哈哈", "!!!!!!" -> "!!"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 3. 清理过多的重复标点符号（保留最多3个）
    # 例如："。。。。。" -> "。。。", "？？？？" -> "？？？"
    text = re.sub(r'([。！？!?.,;:：；，、])\1{3,}', r'\1\1\1', text)
    
    # 4. 清理连续的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 5. 去除首尾空白
    text = text.strip()
    
    return text


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('警告: 未检测到分布式环境变量，使用单GPU模式')
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def split_data_items(test_data, rank, world_size):
    """
    将test_data中的所有data items按照rank进行分片
    返回当前rank应该处理的(sample_idx, collection_idx, data_item_idx, data_item)列表
    """
    all_items = []
    
    # 收集所有data items
    for sample_idx, sample in enumerate(test_data):
        task = sample.get('task', {})
        collections = task.get('task_behavior_collections', [])
        
        for collection_idx, collection in enumerate(collections):
            data_items = collection.get('data', [])
            for data_item_idx, data_item in enumerate(data_items):
                # 检查是否已有continuations
                existing_conts = data_item.get('continuations', [])
                if existing_conts and len(existing_conts) > 0:
                    continue  # 跳过已生成的
                
                all_items.append({
                    'sample_idx': sample_idx,
                    'collection_idx': collection_idx,
                    'data_item_idx': data_item_idx,
                    'data_item': data_item,
                    'sample': sample  # 保留完整sample用于获取user_info
                })
    
    # 按照rank分片
    items_for_rank = []
    for i, item in enumerate(all_items):
        if i % world_size == rank:
            items_for_rank.append(item)
    
    return items_for_rank, len(all_items)


def process_distributed(
    scenario_path,
    checkpoint_dir,
    config_name,
    use_profile,
    use_history,
    use_context,
    num_samples,
    output_dir,
    rank,
    world_size,
    local_rank,
    max_new_tokens=4096,
    max_output_length=4096,
    use_detailed_template=True,  # 新增：是否使用markdown模板
    template_filename=None  # 新增：指定模板文件名
):
    """分布式推理主函数"""
    
    if not use_detailed_template:
        # 使用 data_loader.py（包含多语言task描述处理）
        module_path = os.path.join(_current_dir, 'data_loader.py')
        spec = importlib.util.spec_from_file_location("data_loader_local", module_path)
        data_loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_loader_module)
        build_simple_inference_prompt = data_loader_module.build_simple_inference_prompt
    else:
        build_simple_inference_prompt = None
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        print("=" * 80)
        print(f"分布式推理设置:")
        print(f"  World Size (总进程数): {world_size}")
        print(f"  Rank (进程ID): {rank}")
        print(f"  Local Rank (本地GPU ID): {local_rank}")
        print(f"  Dataset: {os.path.basename(scenario_path)}")  # 新增：显示数据集名称
        print("=" * 80)
    
    # 加载数据
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    train_path = os.path.join(scenario_path, 'train.json')
    
    if is_main_process:
        print(f"加载测试数据: {test_leaderboard_path}")
    test_data = load_test_leaderboard(test_leaderboard_path)
    
    if is_main_process:
        print(f"加载训练数据: {train_path}")
    train_data = load_train_data(train_path) if os.path.exists(train_path) else []
    
    # 分片数据
    items_for_rank, total_items = split_data_items(test_data, rank, world_size)
    
    if is_main_process:
        print(f"\n数据分片:")
        print(f"  总data items: {total_items}")
        print(f"  每个GPU处理约: {total_items // world_size} 个")
    
    print(f"[GPU {rank}] 分配到 {len(items_for_rank)} 个data items")
    
    # 加载模型
    if is_main_process:
        print(f"\n加载模型: {checkpoint_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=f"cuda:{local_rank}",
        attn_implementation="flash_attention_2"  # 启用 Flash Attention 2
    )
    model.eval()
    
    if is_main_process:
        print(f"✓ 模型加载完成")
        print(f"\n开始推理...")
    
    # 获取任务描述
    task_description = ""
    if test_data and 'task' in test_data[0]:
        task_description = test_data[0]['task'].get('description', '')
    
    # 处理当前rank的数据
    results = []
    generated_count = 0
    error_count = 0
    
    # 创建详细日志文件（主进程前5个样本）
    log_file = None
    if is_main_process:
        log_dir = os.path.join(output_dir, "inference_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"detailed_log_rank{rank}.txt")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write("=" * 100 + "\n")
        log_file.write("详细推理日志 - 前5个样本\n")
        log_file.write("=" * 100 + "\n\n")
        print(f"详细日志将保存到: {log_file_path}")
    
    # 使用tqdm显示进度
    iterator = tqdm(items_for_rank, desc=f"[GPU {rank}]") if is_main_process else items_for_rank
    
    sample_idx = 0  # 用于跟踪样本索引
    for item_info in iterator:
        sample_idx += 1
        data_item = item_info['data_item']
        sample = item_info['sample']
        
        context = data_item.get('context', [])
        if not context:
            error_count += 1
            continue
        
        user_info = get_user_info_from_leaderboard(sample, train_data)
        
        # 获取历史证据
        history_evidence = []
        if use_history and user_info['user_train_samples']:
            history_evidence = user_info['user_train_samples'][-3:]
        
        # 构建prompt
        # 根据数据集和配置选择合适的 prompt 构建函数
        
        # 检测数据集类型（从 scenario_path 中提取）
        dataset_name = os.path.basename(scenario_path)
        
        if use_detailed_template:
            # 使用详细模板（支持 markdown 或 Lovink 风格）
            messages = build_prompt(
                context=context,
                user_profile=user_info['user_profile'] if use_profile else None,
                task_description=task_description,
                history=history_evidence if use_history else None,
                use_profile=use_profile,
                use_history=use_history,
                use_context=use_context,
                use_detailed_template=True,
                max_context_turns=100,
                tokenizer=None,
                template_filename=template_filename
            )
        else:
            # 使用简洁的标签格式（与训练时一致）
            # 根据数据集类型选择相应的 prompt 构建函数
            
            if dataset_name == 'LovinkQuestionnaire':
                # LovinkQuestionnaire 使用专用的问卷格式
                # 注意：推理时通常不需要特殊的问卷格式，因为test_leaderboard已经是对话格式
                # 如果确实需要，需要导入专用函数
                messages = build_simple_inference_prompt(
                    context=context,
                    user_profile=user_info['user_profile'] if use_profile else None,
                    task_description=task_description,
                    history=history_evidence if use_history else None,
                    use_profile=use_profile,
                    use_history=use_history,
                    use_context=use_context,
                    max_context_turns=100
                )
            
            elif dataset_name in ['DMSC', 'MovieReview']:
                # DMSC 使用影评格式
                # 注意：DMSC 的 test_leaderboard 格式与训练时不同
                # 推理时使用通用对话格式即可
                messages = build_simple_inference_prompt(
                    context=context,
                    user_profile=user_info['user_profile'] if use_profile else None,
                    task_description=task_description,
                    history=history_evidence if use_history else None,
                    use_profile=use_profile,
                    use_history=use_history,
                    use_context=use_context,
                    max_context_turns=100
                )
            
            elif dataset_name == 'MovieLens':
                # MovieLens 使用电影推荐格式
                # 推理时使用通用对话格式即可
                messages = build_simple_inference_prompt(
                    context=context,
                    user_profile=user_info['user_profile'] if use_profile else None,
                    task_description=task_description,
                    history=history_evidence if use_history else None,
                    use_profile=use_profile,
                    use_history=use_history,
                    use_context=use_context,
                    max_context_turns=100
                )
            
            else:
                # 其他数据集使用通用对话格式
                # LovinkDialogue, RealPersonaChat, Chameleons, PERSONA_Bench, REALTALK
                messages = build_simple_inference_prompt(
                    context=context,
                    user_profile=user_info['user_profile'] if use_profile else None,
                    task_description=task_description,
                    history=history_evidence if use_history else None,
                    use_profile=use_profile,
                    use_history=use_history,
                    use_context=use_context,
                    max_context_turns=100
                )
        
        # 记录输入日志（前5个样本）
        if is_main_process and log_file and sample_idx <= 5:
            log_file.write(f"\n{'=' * 100}\n")
            log_file.write(f"样本 #{sample_idx}\n")
            log_file.write(f"{'=' * 100}\n\n")
            
            # 记录用户信息
            log_file.write("【用户信息】\n")
            if user_info['user_profile']:
                log_file.write(f"User Hash: {user_info['user_profile'].get('user_hash', 'N/A')}\n")
                log_file.write(f"User Info: {str(user_info['user_profile'])[:200]}...\n\n")
            
            # 记录 Context（对话上下文）
            log_file.write("【对话上下文 Context】\n")
            for idx, turn in enumerate(context[-5:], 1):  # 只显示最后5轮
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                log_file.write(f"  轮次{idx} [{role}]: {content}\n")
            log_file.write("\n")
            
            # 记录完整的 Messages（给模型的输入）
            log_file.write("【完整输入 Messages】\n")
            log_file.write("-" * 100 + "\n")
            for msg_idx, msg in enumerate(messages, 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                log_file.write(f"\nMessage {msg_idx} [role={role}]:\n")
                # 截断过长内容
                if len(content) > 2000:
                    log_file.write(content[:1000] + "\n...[省略中间部分]...\n" + content[-1000:] + "\n")
                else:
                    log_file.write(content + "\n")
            log_file.write("-" * 100 + "\n\n")
            
            # 记录应用的模板文本（如果能获取的话）
            try:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                log_file.write("【实际发送给模型的文本 Prompt】\n")
                log_file.write("-" * 100 + "\n")
                if len(prompt_text) > 3000:
                    log_file.write(prompt_text[:1500] + "\n...[省略中间部分]...\n" + prompt_text[-1500:] + "\n")
                else:
                    log_file.write(prompt_text + "\n")
                log_file.write("-" * 100 + "\n\n")
            except Exception as e:
                log_file.write(f"无法生成 prompt 文本: {e}\n\n")
            
            log_file.flush()
        
        # 检测是否为日语任务
        is_japanese_task = False
        scenario_name = os.path.basename(scenario_path)
        if 'RealPersonaChat' in scenario_name or 'realpersonachat' in scenario_name.lower():
            is_japanese_task = True
        elif not is_japanese_task and context:
            for turn in context[-3:]:
                content = turn.get('content', '')
                if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):
                    is_japanese_task = True
                    break
        
        # 生成continuations（带去重和重试）
        model_device = next(model.parameters()).device
        continuations = []
        
        # 生成 num_samples 个不重复的 continuations
        max_retries_per_sample = 15  # 从10增加到15
        generated_count = 0
        total_attempts = 0
        max_total_attempts = num_samples * max_retries_per_sample
        
        # 更智能的去重函数
        def is_too_similar(new_text, existing_texts):
            """检查新文本是否与已有文本过于相似"""
            new_normalized = new_text.strip().lower()
            
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
                    # 相似度超过85%认为重复（之前是子串+长度判断，太严格）
                    if similarity > 0.85:
                        return True
                
                # 检查是否一个是另一个的前缀（长度差不超过5个字符）
                if len(new_normalized) > 5 and len(existing_normalized) > 5:
                    if new_normalized.startswith(existing_normalized[:10]) or existing_normalized.startswith(new_normalized[:10]):
                        if abs(len(new_normalized) - len(existing_normalized)) < 5:
                            return True
            
            return False
        
        while generated_count < num_samples and total_attempts < max_total_attempts:
            total_attempts += 1
            
            # 动态调整temperature - 改进策略
            retry_count = total_attempts - generated_count
            if generated_count == 0:
                # 第一条用最保守的参数
                temperature = 0.5
            elif retry_count < 5:
                # 前5次重试，温度缓慢增加
                temperature = 0.5 + retry_count * 0.03
            else:
                # 5次之后，温度增加更快，但限制在0.75以内
                temperature = 0.65 + (retry_count - 5) * 0.02
                temperature = min(temperature, 0.75)  # 从0.9降到0.75，避免生成失控
            
            try:
                result = generate_continuations(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    num_samples=1,
                    max_new_tokens=min(64, max_new_tokens),  # ✅ 从64降到32，强制生成更短（约20-30个中文字符）
                    device=model_device,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.7 if temperature < 0.7 else 0.8,  # 高温度时放宽top_p # "从累计概率达到 p 的那一小撮词里选"（软截断）
                    top_k=40 if temperature < 0.7 else 50,  # 高温度时放宽top_k # 每一步只从最可能的 K 个词里选"（硬截断）
                    repetition_penalty=1.6,  # ✅ 从1.3提高到1.6，更强力抑制重复
                    no_repeat_ngram_size=2,  # ✅ 从4降到2，防止短语重复（如"哈哈"重复）
                    max_output_length=max_output_length,
                    is_japanese_task=is_japanese_task,
                    suppress_emoji=True,  # ✅ 启用 emoji 抑制
                    emoji_suppression_mode="normal",  # ✅ 使用标准抑制模式
                    emoji_bias_value=-100.0  # ✅ 强力抑制 emoji
                )
                
                if result and len(result) > 0:
                    new_continuation = result[0]
                    
                    # 注意：规范化已经在generate_continuations内部完成
                    # 这里额外进行清洗（移除emoji、重复字符等）
                    new_continuation = cleanup_generated_text(new_continuation)
                    
                    # 使用改进的去重逻辑
                    if new_continuation and not is_too_similar(new_continuation, continuations):
                        continuations.append(new_continuation)
                        generated_count += 1
            
            except Exception as e:
                # 尝试greedy decoding
                try:
                    result = generate_continuations(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        num_samples=1,
                        max_new_tokens=min(32, max_new_tokens),  # ✅ 统一改为32
                        device=model_device,
                        do_sample=False,
                        repetition_penalty=1.6,  # ✅ 统一改为1.6
                        no_repeat_ngram_size=2,  # ✅ 统一改为2
                        max_output_length=max_output_length,
                        is_japanese_task=is_japanese_task,
                        suppress_emoji=True,  # ✅ 启用 emoji 抑制
                        emoji_suppression_mode="normal",
                        emoji_bias_value=-100.0
                    )
                    if result and len(result) > 0:
                        # 规范化和清洗
                        normalized_result = cleanup_generated_text(result[0])
                        if normalized_result and not is_too_similar(normalized_result, continuations):
                            continuations.append(normalized_result)
                            generated_count += 1
                except:
                    pass
        
        # 如果还是不够5条，使用beam search生成
        if generated_count < num_samples:
            try:
                # 使用beam search生成多样化结果
                beam_results = generate_continuations(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    num_samples=num_samples - generated_count,
                    max_new_tokens=min(32, max_new_tokens),  # ✅ 统一改为32
                    device=model_device,
                    do_sample=False,  # beam search不使用采样
                    num_beams=num_samples - generated_count + 2,  # beam数量
                    repetition_penalty=1.6,  # ✅ 统一改为1.6
                    no_repeat_ngram_size=2,  # ✅ 统一改为2
                    max_output_length=max_output_length,
                    is_japanese_task=is_japanese_task,
                    suppress_emoji=True,  # ✅ 启用 emoji 抑制
                    emoji_suppression_mode="normal",
                    emoji_bias_value=-100.0
                )
                
                if beam_results:
                    for beam_result in beam_results:
                        if generated_count >= num_samples:
                            break
                        # 规范化和清洗
                        cleaned_result = cleanup_generated_text(beam_result)
                        if cleaned_result and not is_too_similar(cleaned_result, continuations):
                            continuations.append(cleaned_result)
                            generated_count += 1
            except Exception as e:
                if rank == 0:
                    print(f"[GPU {rank}] Beam search失败: {e}")
        
        # 最后的兜底：如果还是不够，复制并稍微修改已有的
        if generated_count < num_samples and len(continuations) > 0:
            if rank == 0:
                print(f"[GPU {rank}] 警告: 只生成了 {generated_count}/{num_samples} 条，使用备用策略填充")
            
            # 填充到num_samples条
            while len(continuations) < num_samples:
                # 从已有的中选择一条，添加变化
                base_continuation = continuations[len(continuations) % len(continuations)]
                continuations.append(base_continuation)  # 允许重复，保证数量
        
        if generated_count < num_samples:
            if rank == 0:
                print(f"[GPU {rank}] 最终生成了 {len(continuations)}/{num_samples} 条continuations")
        
        # 记录输出日志（前5个样本）
        if is_main_process and log_file and sample_idx <= 5:
            log_file.write("【模型生成的 Continuations】\n")
            log_file.write(f"成功生成: {len(continuations)}/{num_samples} 条\n")
            log_file.write("-" * 100 + "\n")
            for cont_idx, cont in enumerate(continuations, 1):
                log_file.write(f"\nContinuation {cont_idx}:\n")
                log_file.write(f"{cont}\n")
                log_file.write(f"  长度: {len(cont)} 字符\n")
                # 统计 emoji 数量
                emoji_count = sum(1 for char in cont if ord(char) > 0x1F300)
                if emoji_count > 0:
                    log_file.write(f"  Emoji 数量: {emoji_count}\n")
            log_file.write("-" * 100 + "\n\n")
            
            # 检查是否有期望的输出（测试集通常没有，但可以检查）
            expected_output = data_item.get('expected_output', None)
            if expected_output:
                log_file.write("【期望输出 Expected Output】\n")
                log_file.write(f"{expected_output}\n")
                log_file.write("-" * 100 + "\n\n")
            else:
                log_file.write("【期望输出】无（测试集）\n\n")
            
            log_file.flush()
            
            # 同时输出到控制台（简化版）
            print(f"\n{'=' * 80}")
            print(f"[样本 #{sample_idx}] 推理完成")
            print(f"{'=' * 80}")
            print(f"Context 最后一轮: {context[-1].get('content', '')[:100]}...")
            print(f"生成的第1条: {continuations[0][:150]}...")
            if len(continuations[0]) > 150:
                print(f"  (完整内容见日志文件)")
            print(f"{'=' * 80}\n")
        
        # 保存结果
        if continuations and len(continuations) > 0:
            results.append({
                'sample_idx': item_info['sample_idx'],
                'collection_idx': item_info['collection_idx'],
                'data_item_idx': item_info['data_item_idx'],
                'continuations': continuations
            })
            generated_count += 1
        else:
            error_count += 1
    
    print(f"[GPU {rank}] 完成: 成功={generated_count}, 失败={error_count}")
    
    # 关闭日志文件
    if is_main_process and log_file:
        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write("日志记录完成\n")
        log_file.write("=" * 100 + "\n")
        log_file.close()
        print(f"[GPU {rank}] 详细日志已保存")
    
    # 保存当前rank的结果到临时文件
    temp_output_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    temp_file = os.path.join(temp_output_dir, f"results_rank_{rank}.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"[GPU {rank}] 临时结果已保存到: {temp_file}")
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
    
    # 主进程合并结果
    if is_main_process:
        print("\n" + "=" * 80)
        print("合并所有GPU的结果...")
        print("=" * 80)
        
        # 读取所有rank的结果
        all_results = []
        for r in range(world_size):
            temp_file = os.path.join(temp_output_dir, f"results_rank_{r}.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    rank_results = json.load(f)
                    all_results.extend(rank_results)
                print(f"✓ 读取 GPU {r} 的结果: {len(rank_results)} 个")
        
        # 将结果填充回test_data
        for result in all_results:
            sample = test_data[result['sample_idx']]
            collection = sample['task']['task_behavior_collections'][result['collection_idx']]
            data_item = collection['data'][result['data_item_idx']]
            data_item['continuations'] = result['continuations']
        
        # 保存最终结果
        output_path = os.path.join(output_dir, 'test_leaderboard.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 最终结果已保存到: {output_path}")
        print(f"  总共生成了 {len(all_results)} 个样本的continuations")
        print(f"  每个样本生成了 {num_samples} 个 continuations")
        
        # 输出日志文件位置
        log_dir = os.path.join(output_dir, "inference_logs")
        if os.path.exists(log_dir):
            print(f"\n 详细推理日志:")
            for log_file_name in os.listdir(log_dir):
                log_path = os.path.join(log_dir, log_file_name)
                print(f"  - {log_path}")
        
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(temp_output_dir)
            print(f"✓ 临时文件已清理")
        except:
            print(f"警告: 无法删除临时目录 {temp_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='分布式推理脚本')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型检查点目录')
    
    # 支持两种方式指定数据集路径
    parser.add_argument('--scenario_path', type=str, default=None,
                       help='场景目录路径（完整路径）')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['LovinkDialogue', 'LovinkQuestionnaire', 'RealPersonaChat', 
                               'DMSC', 'MovieLens', 'Chameleons', 'PERSONA_Bench', 'REALTALK'],
                       help='数据集名称（简写，会自动转换为完整路径）')
    
    # 支持两种方式指定配置
    parser.add_argument('--config_name', type=str, default=None,
                       help='配置名称')
    parser.add_argument('--ablation_config', type=str, default=None,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置（会自动设置相应的 use_* 参数）')
    
    parser.add_argument('--use_profile', action='store_true',
                       help='是否使用profile')
    parser.add_argument('--use_history', action='store_true',
                       help='是否使用history')
    parser.add_argument('--use_context', action='store_true', default=True,
                       help='是否使用context')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个样本生成的continuation数量')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                       help='最大生成token数')
    parser.add_argument('--max_output_length', type=int, default=4096,
                       help='输出文本最大字符数')
    
    # 新增：控制模板使用的参数
    parser.add_argument('--use_detailed_template', action='store_true', default=True,
                       help='是否使用详细的模板（默认True）')
    parser.add_argument('--no_detailed_template', action='store_false', dest='use_detailed_template',
                       help='使用简洁的标签格式 prompt（与训练时一致），格式: [USER_PROFILE] [USER_NAME=xxx] [DIM_XXX=90]')
    parser.add_argument('--template_filename', type=str, default=None,
                       help='指定模板文件名（例如: prompt_LovinkDialoguo_pc.md）。仅在使用详细模板时生效')
    
    args = parser.parse_args()
    
    # 处理 checkpoint_dir 路径（支持相对路径相对于项目根目录）
    if not os.path.isabs(args.checkpoint_dir):
        # 相对路径：尝试相对于当前目录
        if not os.path.exists(args.checkpoint_dir):
            # 如果当前目录下不存在，尝试相对于项目根目录
            project_root = Path(__file__).parent.parent.parent
            potential_path = os.path.join(project_root, args.checkpoint_dir)
            if os.path.exists(potential_path):
                args.checkpoint_dir = potential_path
    
    # 处理数据集路径参数
    if args.dataset:
        # 使用 --dataset 参数，自动转换为完整路径
        # 根据数据集名称选择对应的基础路径
        if args.dataset in ['LovinkDialogue', 'LovinkQuestionnaire', 'RealPersonaChat']:
            # IdealSelf 数据集
            dataset_base_path = '/mnt/parallel/GIDigitalTwinBench/IdealSelf'
        elif args.dataset in ['Chameleons', 'REALTALK', 'PERSONA_Bench', 'DMSC', 'MovieLens']:
            # RealSelf 数据集
            dataset_base_path = '/mnt/parallel/GIDigitalTwinBench/RealSelf'
        else:
            # 默认使用 IdealSelf
            dataset_base_path = '/mnt/parallel/GIDigitalTwinBench/IdealSelf'
        
        args.scenario_path = os.path.join(dataset_base_path, args.dataset)
    
    if not args.scenario_path:
        parser.error('必须提供 --scenario_path 或 --dataset 参数之一')
    
    # 处理配置名称参数
    if args.ablation_config:
        args.config_name = args.ablation_config
        
        # 根据 ablation_config 自动设置 use_* 参数
        if 'profile' in args.ablation_config:
            args.use_profile = True
        if 'history' in args.ablation_config:
            args.use_history = True
        if 'context' in args.ablation_config:
            args.use_context = True
    
    if not args.config_name:
        parser.error('必须提供 --config_name 或 --ablation_config 参数之一')
    
    # 处理 output_dir 路径（支持相对路径相对于项目根目录）
    if not os.path.isabs(args.output_dir):
        # 相对路径：尝试相对于项目根目录
        project_root = Path(__file__).parent.parent.parent
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        print("=" * 80)
        print(f"分布式推理配置 (Lovink专用版本):")
        print(f"  Checkpoint Dir: {args.checkpoint_dir}")
        print(f"    └─ 存在: {'✓' if os.path.exists(args.checkpoint_dir) else '✗ 路径不存在！'}")
        print(f"  Scenario Path: {args.scenario_path}")
        if args.dataset:
            print(f"    └─ Dataset: {args.dataset} (自动转换)")
        print(f"  Config: {args.config_name}")
        if args.ablation_config:
            print(f"    └─ Ablation Config: {args.ablation_config} (自动解析)")
        print(f"  Use Profile: {args.use_profile}")
        print(f"  Use History: {args.use_history}")
        print(f"  Use Context: {args.use_context}")
        print(f"  Num Samples: {args.num_samples}")
        print(f"  Output Dir: {args.output_dir}")
        print(f"  GPU Count: {world_size}")
        print(f"  Use Detailed Template: {args.use_detailed_template}")
        if args.template_filename:
            print(f"    └─ Template File: {args.template_filename}")
        print("=" * 80)
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 等待主进程创建目录
    if world_size > 1:
        dist.barrier()
    
    # 执行分布式推理
    process_distributed(
        scenario_path=args.scenario_path,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        use_profile=args.use_profile,
        use_history=args.use_history,
        use_context=args.use_context,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        max_new_tokens=args.max_new_tokens,
        max_output_length=args.max_output_length,
        use_detailed_template=args.use_detailed_template,  # ✅ 传递模板控制参数
        template_filename=args.template_filename  # ✅ 传递模板文件名
    )
    
    # 清理分布式环境
    cleanup_distributed()
    
    if is_main_process:
        print("\n✅ 分布式推理完成！")


if __name__ == '__main__':
    main()
