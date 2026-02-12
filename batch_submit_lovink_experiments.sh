#!/bin/bash

# 批量提交Lovink问卷消融实验
# 用法: bash batch_submit_lovink_experiments.sh

WORK_DIR="/mnt/parallel/CompactSubset_experiement"

echo "================================"
echo "批量提交Lovink问卷消融实验"
echo "================================"

# 确保logs目录存在
mkdir -p "$WORK_DIR/logs"
echo "✓ 日志目录已创建"

# 定义实验配置
declare -a EXPERIMENTS=(
    # 格式: "ablation_config|history_strategy|history_ratio|描述"
    "profile_and_history|fixed_ratio|0.5|完整模型-固定比例50%"
    "profile_and_history|fixed_ratio|0.3|完整模型-固定比例30%"
    "profile_and_history|fixed_ratio|0.7|完整模型-固定比例70%"
    "profile_and_history|random|0.5|完整模型-随机历史"
    "profile_and_history|all_previous|0.5|完整模型-所有历史"
    "profile_only|none|0.5|仅Profile"
    "history_only|fixed_ratio|0.5|仅历史"
    "context_only|none|0.5|仅问题"
)

echo ""
echo "将提交 ${#EXPERIMENTS[@]} 个实验:"
echo "---"
for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r ablation strategy ratio desc <<< "${EXPERIMENTS[$i]}"
    echo "$((i+1)). $desc"
    echo "   ablation=$ablation, strategy=$strategy, ratio=$ratio"
done
echo "---"

echo ""
read -p "确认提交所有实验? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "批量提交已取消"
    exit 0
fi

echo ""
echo "开始提交作业..."
echo ""

SBATCH_TEMPLATE="$WORK_DIR/train_lovink_questionnaire.sbatch"
SUBMITTED_JOBS=()

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r ablation strategy ratio desc <<< "$exp"
    
    # 创建临时sbatch文件
    TEMP_SBATCH=$(mktemp)
    cp "$SBATCH_TEMPLATE" "$TEMP_SBATCH"
    
    # 修改参数
    sed -i "s/^ABLATION_CONFIG=.*/ABLATION_CONFIG=\"$ablation\"/" "$TEMP_SBATCH"
    sed -i "s/^HISTORY_STRATEGY=.*/HISTORY_STRATEGY=\"$strategy\"/" "$TEMP_SBATCH"
    sed -i "s/^HISTORY_RATIO=.*/HISTORY_RATIO=$ratio/" "$TEMP_SBATCH"
    
    # 修改job名称以区分不同实验
    sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=lovink_${ablation}_${strategy}/" "$TEMP_SBATCH"
    
    # 提交作业
    JOB_OUTPUT=$(sbatch "$TEMP_SBATCH")
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
    
    echo "✓ [$desc]"
    echo "  Job ID: $JOB_ID"
    echo "  日志: logs/lovink_${ablation}_${strategy}_${JOB_ID}.out"
    echo ""
    
    SUBMITTED_JOBS+=("$JOB_ID|$desc")
    
    # 清理临时文件
    rm -f "$TEMP_SBATCH"
    
    # 短暂延迟，避免同时提交过多作业
    sleep 2
done

echo "================================"
echo "所有作业已提交！"
echo "================================"
echo ""
echo "已提交的作业:"
for job_info in "${SUBMITTED_JOBS[@]}"; do
    IFS='|' read -r job_id job_desc <<< "$job_info"
    echo "  Job $job_id: $job_desc"
done

echo ""
echo "查看所有作业状态:"
echo "  squeue -u $USER"
echo ""
echo "查看特定作业日志:"
echo "  tail -f $WORK_DIR/logs/lovink_*_<JOB_ID>.out"
echo ""
echo "取消所有作业:"
echo "  scancel ${SUBMITTED_JOBS[@]/%|*/}"
