#!/bin/bash

# 快速测试 vLLM 推理脚本

set -e

echo "===== 测试 vLLM 推理脚本 ====="
echo "Date: $(date)"
echo

# 测试数据路径推断
echo "测试 1: 验证数据路径映射"
echo "---"

datasets=("DMSC" "Chameleons" "LovinkDialogue" "LovinkQuestionnaire")

for dataset in "${datasets[@]}"; do
    echo -n "  $dataset: "
    
    # 根据 dataset 推断路径
    case $dataset in
        "DMSC"|"Chameleons"|"MovieReview")
            base_path="/mnt/parallel/GIDigitalTwinBench/RealSelf"
            ;;
        *)
            base_path="/mnt/parallel/GIDigitalTwinBench/IdealSelf"
            ;;
    esac
    
    data_path="$base_path/$dataset"
    
    if [ -f "$data_path/test_leaderboard.json" ]; then
        echo "✓ $data_path"
    else
        echo "✗ 路径不存在: $data_path"
    fi
done

echo
echo "测试 2: 验证 vLLM 安装"
echo "---"
if python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')" 2>&1; then
    echo "  ✓ vLLM 已安装"
else
    echo "  ✗ vLLM 未安装"
    echo "  请运行: pip install vllm"
    exit 1
fi

echo
echo "测试 3: 验证 checkpoint"
echo "---"
checkpoint="outputs/DMSC_8B_one_per_user_0213"
if [ -d "$checkpoint" ]; then
    echo "  ✓ Checkpoint 存在: $checkpoint"
    # 检查必要文件
    if [ -f "$checkpoint/config.json" ]; then
        echo "    ✓ config.json"
    else
        echo "    ✗ config.json 缺失"
    fi
    if [ -f "$checkpoint/tokenizer.json" ] || [ -f "$checkpoint/tokenizer_config.json" ]; then
        echo "    ✓ tokenizer"
    else
        echo "    ✗ tokenizer 缺失"
    fi
else
    echo "  ✗ Checkpoint 不存在: $checkpoint"
    echo "  可用的 checkpoints:"
    ls -d outputs/*_8B_* 2>/dev/null | head -5 || echo "    (无)"
fi

echo
echo "测试 4: 快速推理测试（1个样本）"
echo "---"
echo "运行快速推理测试..."

# 使用最小配置测试
if python inference_vllm.py \
    --checkpoint_dir outputs/DMSC_8B_one_per_user_0213 \
    --dataset DMSC \
    --ablation_config profile_and_history \
    --num_samples 1 \
    --output_dir outputs/test_vllm_quick \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.7 \
    --max_model_len 4096 \
    2>&1 | head -50; then
    
    echo
    echo "✓ 快速测试成功"
else
    echo
    echo "✗ 快速测试失败"
    echo "查看完整日志以获取详细错误信息"
    exit 1
fi

echo
echo "===== 所有测试完成 ====="
echo "可以开始正式推理了！"
echo
echo "运行完整推理:"
echo "  python inference_vllm.py \\"
echo "    --checkpoint_dir outputs/DMSC_8B_one_per_user_0213 \\"
echo "    --dataset DMSC \\"
echo "    --ablation_config profile_and_history \\"
echo "    --num_samples 5 \\"
echo "    --output_dir outputs/leaderboards/DMSC_vllm_8gpu \\"
echo "    --tensor_parallel_size 8 \\"
echo "    --gpu_memory_utilization 0.9"
