#!/bin/bash

# Lovink问卷训练作业提交脚本
# 用法: bash submit_lovink_job.sh [ablation_config] [history_strategy] [history_ratio]

# 默认参数
ABLATION_CONFIG=${1:-"profile_and_history"}
HISTORY_STRATEGY=${2:-"fixed_ratio"}
HISTORY_RATIO=${3:-"0.5"}

echo "================================"
echo "提交Lovink问卷训练作业"
echo "================================"
echo "消融配置: $ABLATION_CONFIG"
echo "历史策略: $HISTORY_STRATEGY"
echo "历史比例: $HISTORY_RATIO"
echo "================================"

# 确保logs目录存在
WORK_DIR="/mnt/parallel/CompactSubset_experiement"
mkdir -p "$WORK_DIR/logs"
echo "✓ 日志目录已创建: $WORK_DIR/logs"

# 修改sbatch脚本中的参数
SBATCH_FILE="$WORK_DIR/train_lovink_questionnaire.sbatch"
TEMP_SBATCH=$(mktemp)

# 复制原文件并修改参数
cp "$SBATCH_FILE" "$TEMP_SBATCH"

sed -i "s/^ABLATION_CONFIG=.*/ABLATION_CONFIG=\"$ABLATION_CONFIG\"/" "$TEMP_SBATCH"
sed -i "s/^HISTORY_STRATEGY=.*/HISTORY_STRATEGY=\"$HISTORY_STRATEGY\"/" "$TEMP_SBATCH"
sed -i "s/^HISTORY_RATIO=.*/HISTORY_RATIO=$HISTORY_RATIO/" "$TEMP_SBATCH"

echo ""
echo "修改后的配置:"
echo "---"
grep "^ABLATION_CONFIG=" "$TEMP_SBATCH"
grep "^HISTORY_STRATEGY=" "$TEMP_SBATCH"
grep "^HISTORY_RATIO=" "$TEMP_SBATCH"
echo "---"

echo ""
read -p "确认提交作业? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    JOB_ID=$(sbatch "$TEMP_SBATCH" | awk '{print $NF}')
    echo ""
    echo "✓ 作业已提交！"
    echo "  Job ID: $JOB_ID"
    echo "  输出日志: $WORK_DIR/logs/lovink_questionnaire_${JOB_ID}.out"
    echo "  错误日志: $WORK_DIR/logs/lovink_questionnaire_${JOB_ID}.err"
    echo ""
    echo "查看日志："
    echo "  tail -f $WORK_DIR/logs/lovink_questionnaire_${JOB_ID}.out"
    echo ""
    echo "查看作业状态："
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "取消作业："
    echo "  scancel $JOB_ID"
else
    echo "作业提交已取消"
fi

# 清理临时文件
rm -f "$TEMP_SBATCH"
