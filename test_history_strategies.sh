#!/bin/bash

# 测试不同历史策略的影响
# 用法: bash test_history_strategies.sh <data_file>

DATA_FILE=${1:-"example_lovink_questionnaire.json"}
OUTPUT_BASE="outputs/history_strategy_test"

echo "================================"
echo "测试Lovink问卷历史策略"
echo "================================"
echo "数据文件: $DATA_FILE"
echo "输出目录: $OUTPUT_BASE"
echo "================================"

# 测试不同策略（只训练1个epoch，快速验证）
strategies=("all_previous" "fixed_ratio" "fixed_count" "random" "none")

for strategy in "${strategies[@]}"; do
    echo ""
    echo "测试策略: $strategy"
    echo "--------------------------------"
    
    # 构建命令
    cmd="python train_distributed_LovinkQuestionnaire.py \
        --config config_LovinkQuestionnaire.json \
        --ablation_config profile_and_history \
        --history_strategy $strategy \
        --output_dir ${OUTPUT_BASE}/${strategy} \
        --max_epochs 1"
    
    # 根据策略添加额外参数
    if [ "$strategy" == "fixed_ratio" ]; then
        cmd="$cmd --history_ratio 0.5"
    elif [ "$strategy" == "fixed_count" ]; then
        cmd="$cmd --fixed_history_count 3"
    fi
    
    echo "运行: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ 策略 $strategy 测试成功"
    else
        echo "✗ 策略 $strategy 测试失败"
    fi
done

echo ""
echo "================================"
echo "测试完成！"
echo "================================"
echo "查看结果:"
echo "  ls -lh ${OUTPUT_BASE}/*/training_samples_log.txt"
echo ""
echo "对比不同策略的样本："
for strategy in "${strategies[@]}"; do
    log_file="${OUTPUT_BASE}/${strategy}/training_samples_log.txt"
    if [ -f "$log_file" ]; then
        echo ""
        echo "=== 策略: $strategy ==="
        head -n 30 "$log_file"
    fi
done
