#!/bin/bash

# 使用 vLLM 进行高性能推理的辅助脚本
# 用法: bash inference_with_vllm.sh

set -euo pipefail

# ===== 配置 =====
WORK_DIR="/mnt/parallel/CompactSubset_experiement"

echo "===== vLLM 推理脚本 ====="
echo "Date: $(date)"
echo "工作目录: $WORK_DIR"
echo

# ===== 验证环境 =====
echo "===== 验证环境 ====="
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    echo "错误: vLLM 未安装"
    echo "请运行: pip install vllm"
    exit 1
}
echo "✓ vLLM 已安装"
echo

# ===== 检查 GPU =====
echo "===== GPU 信息 ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
    echo "错误: 无法获取 GPU 信息"
    exit 1
}
echo

# ===== 切换到工作目录 =====
cd "$WORK_DIR"

# ===== 定义 vLLM 推理函数 =====
run_vllm_inference() {
    local task_name=$1
    local checkpoint_dir=$2
    local dataset=$3
    local ablation_config=$4
    local output_dir=$5
    local tensor_parallel_size=${6:-1}  # 默认单GPU
    local num_samples=${7:-5}           # 默认5个样本
    
    echo "=========================================="
    echo "vLLM 推理任务: $task_name"
    echo "时间: $(date)"
    echo "Checkpoint: $checkpoint_dir"
    echo "Dataset: $dataset"
    echo "Ablation: $ablation_config"
    echo "Tensor Parallel: $tensor_parallel_size GPU(s)"
    echo "=========================================="
    
    # 检查 checkpoint 是否存在
    if [ ! -d "$checkpoint_dir" ]; then
        echo "错误: Checkpoint 不存在: $checkpoint_dir"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 运行 vLLM 推理
    local log_file="logs/vllm_inference_${task_name}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs
    
    if python inference_vllm.py \
        --checkpoint_dir "$checkpoint_dir" \
        --dataset "$dataset" \
        --ablation_config "$ablation_config" \
        --num_samples "$num_samples" \
        --output_dir "$output_dir" \
        --tensor_parallel_size "$tensor_parallel_size" \
        --gpu_memory_utilization 0.9 \
        --max_model_len 8192 \
        --temperature 1.0 \
        --top_p 0.9 \
        --top_k 50 \
        --max_tokens 512 \
        2>&1 | tee "$log_file"; then
        
        echo "✓ vLLM 推理任务 $task_name 完成"
        echo "完成时间: $(date)"
        echo "日志文件: $log_file"
        
        # 检查输出
        if [ -f "$output_dir/inference_summary.json" ]; then
            echo "✓ 推理结果已生成"
            python -c "
import json
with open('$output_dir/inference_summary.json', 'r') as f:
    summary = json.load(f)
    print(f\"  用户数: {summary['num_users']}\")
    print(f\"  总样本数: {summary['total_samples']}\")
    print(f\"  吞吐量: {summary['throughput_samples_per_sec']:.2f} samples/sec\")
"
            return 0
        else
            echo "警告: inference_summary.json 未生成"
            return 1
        fi
    else
        local exit_code=$?
        echo "✗ vLLM 推理任务 $task_name 失败，退出码: $exit_code"
        echo "查看日志: $log_file"
        exit $exit_code
    fi
    echo
}

# ===== 示例：运行推理任务 =====

# 示例 1: Chameleons (单GPU)
run_vllm_inference \
    "Chameleons_single_gpu" \
    "outputs/Chameleons_8B_context_sampled_seed42" \
    "Chameleons" \
    "context_only" \
    "outputs/leaderboards/Chameleons_8B_context_vllm" \
    1 \
    5

# 示例 2: DMSC (4 GPU Tensor Parallel)
# run_vllm_inference \
#     "DMSC_4gpu_tp" \
#     "outputs/DMSC_8B_one_per_user_0213" \
#     "DMSC" \
#     "profile_and_history" \
#     "outputs/leaderboards/DMSC_8B_vllm_4gpu" \
#     4 \
#     5

echo
echo "===== 所有推理任务完成 ====="
echo "Date: $(date)"
