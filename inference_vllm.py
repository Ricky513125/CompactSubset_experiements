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

# 日本语规范化（如果可用）
try:
    from japanese_text_normalizer import normalize_japanese_text
except ImportError:
    def normalize_japanese_text(text):
        return text


def build_inference_prompt(
    user_info: Dict[str, Any],
    use_profile: bool = True,
    use_context: bool = True,
    use_history: bool = False,
    dataset_name: str = "Unknown"
) -> str:
    """
    构建推理 prompt
    
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
        parts.append("[USER_PROFILE]")
        if 'name' in profile:
            parts.append(f"[USER_NAME={profile['name']}]")
        if 'age' in profile:
            parts.append(f"[USER_AGE={profile['age']}]")
        if 'gender' in profile:
            parts.append(f"[USER_GENDER={profile['gender']}]")
        
        # 人格维度
        for key, value in profile.items():
            if key.startswith('DIM_') or key.startswith('dim_'):
                parts.append(f"[{key.upper()}={value}]")
        parts.append("")
    
    # 2. 任务描述
    parts.append(f"[TASK] 基于用户在 {dataset_name} 中的历史数据，预测用户的下一条回复。")
    parts.append("")
    
    # 3. 历史信息
    if use_history and user_info.get('history'):
        parts.append("[HISTORY]")
        for i, hist_item in enumerate(user_info['history'][-5:], 1):  # 最近5条
            parts.append(f"  {i}. {hist_item}")
        parts.append("")
    
    # 4. 对话上下文
    if use_context and user_info.get('context'):
        parts.append("[DIALOGUE_CONTEXT]")
        for turn in user_info['context']:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
        parts.append("")
    
    # 5. 生成提示
    parts.append("根据以上信息，预测用户的下一条回复:")
    
    return "\n".join(parts)


def generate_with_vllm(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    show_progress: bool = True
) -> List[str]:
    """
    使用 vLLM 批量生成
    
    Args:
        llm: vLLM LLM 实例
        prompts: 待生成的 prompts 列表
        sampling_params: 采样参数
        show_progress: 是否显示进度条
    
    Returns:
        生成的文本列表
    """
    if show_progress:
        print(f"使用 vLLM 生成 {len(prompts)} 个样本...")
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
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 512,
    seed: int = 42
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
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        skip_special_tokens=True,
    )
    
    print(f"\n采样参数:")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  max_tokens: {max_tokens}")
    
    # 准备所有推理请求
    print("\n准备推理请求...")
    all_prompts = []
    all_metadata = []  # 存储每个 prompt 的元信息
    
    dataset_name = os.path.basename(scenario_path)
    
    for test_sample in tqdm(test_leaderboard, desc="构建 prompts"):
        # 从测试样本中获取用户信息
        user_info = get_user_info_from_leaderboard(
            sample=test_sample,
            train_data=train_data
        )
        
        if not user_info:
            continue
        
        # 获取 user_hash
        user_hash = test_sample.get('user_hash', test_sample.get('user', {}).get('hash', 'unknown'))
        
        # 为每个用户生成 num_samples 个样本
        for sample_idx in range(num_samples):
            prompt = build_inference_prompt(
                user_info=user_info,
                use_profile=use_profile,
                use_context=use_context,
                use_history=use_history,
                dataset_name=dataset_name
            )
            
            all_prompts.append(prompt)
            all_metadata.append({
                'user_hash': user_hash,
                'sample_idx': sample_idx,
                'user_profile': user_info.get('user_profile'),
                'context': user_info.get('context'),
                'history': user_info.get('history', [])
            })
    
    print(f"总推理请求数: {len(all_prompts)}")
    
    # 批量推理
    print("\n开始批量推理...")
    inference_start = time.time()
    
    generated_texts = generate_with_vllm(
        llm=llm,
        prompts=all_prompts,
        sampling_params=sampling_params,
        show_progress=True
    )
    
    inference_time = time.time() - inference_start
    throughput = len(all_prompts) / inference_time
    
    print(f"\n✓ 推理完成")
    print(f"  总样本数: {len(all_prompts)}")
    print(f"  推理时间: {inference_time:.2f}s")
    print(f"  吞吐量: {throughput:.2f} samples/sec ({throughput * 60:.0f} samples/min)")
    
    # 保存结果
    print(f"\n保存结果到: {output_dir}")
    
    # 按用户组织结果
    user_results = {}
    for metadata, generated_text in zip(all_metadata, generated_texts):
        user_hash = metadata['user_hash']
        if user_hash not in user_results:
            user_results[user_hash] = {
                'user_hash': user_hash,
                'user_profile': metadata['user_profile'],
                'context': metadata['context'],
                'history': metadata['history'],
                'generated_samples': []
            }
        
        user_results[user_hash]['generated_samples'].append({
            'sample_idx': metadata['sample_idx'],
            'generated_text': generated_text
        })
    
    # 保存每个用户的结果
    for user_hash, result in user_results.items():
        user_output_file = os.path.join(output_dir, f"{user_hash}.json")
        with open(user_output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 保存汇总信息
    summary = {
        'checkpoint': checkpoint_dir,
        'scenario': scenario_path,
        'ablation_config': ablation_config,
        'num_users': len(user_results),
        'num_samples_per_user': num_samples,
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
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(output_dir, 'inference_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 结果已保存")
    print(f"  用户数: {len(user_results)}")
    print(f"  汇总文件: {summary_file}")
    
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
                       help='最大模型序列长度（默认: 8192）')
    
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
