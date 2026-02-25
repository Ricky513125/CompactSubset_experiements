#!/bin/bash

# 批量对所有数据集进行采样
# 每个用户最多保留 N 个样本，生成 train_N.json

set -e

BASE_DIR="/mnt/parallel/GIDigitalTwinBench/RealSelf"
OUTPUT_BASE_DIR="./sampled_data"
MAX_SAMPLES=3
SEED=42
FORCE_OVERWRITE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --force|-f)
            FORCE_OVERWRITE=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --max_samples N    每用户最多样本数 (默认: 3)"
            echo "  --seed N           随机种子 (默认: 42)"
            echo "  --force, -f        强制覆盖已存在的文件"
            echo "  --help, -h         显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "批量数据集采样"
echo "================================================================================"
echo "输入基础目录: ${BASE_DIR}"
echo "输出基础目录: ${OUTPUT_BASE_DIR}"
echo "每用户最多样本数: ${MAX_SAMPLES}"
echo "随机种子: ${SEED}"
echo "强制覆盖: ${FORCE_OVERWRITE}"
echo "================================================================================"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}"

# 数据集列表
DATASETS=(
    "Chameleons"
    "LovinkDialogue"
    "LovinkQuestionnaire"
    "MovieLens"
    "PERSONA_Bench"
    "RealPersonaChat"
    "REALTALK"
    "DMSC"
)

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

for dataset in "${DATASETS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "处理数据集: ${dataset}"
    echo "--------------------------------------------------------------------------------"
    
    INPUT_FILE="${BASE_DIR}/${dataset}/train.json"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${dataset}"
    OUTPUT_FILE="${OUTPUT_DIR}/train_${MAX_SAMPLES}.json"
    
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "⚠️  跳过: 输入文件不存在: ${INPUT_FILE}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_DATASETS+=("${dataset} (文件不存在)")
        echo ""
        continue
    fi
    
    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}"
    
    if [ -f "${OUTPUT_FILE}" ] && [ "${FORCE_OVERWRITE}" = false ]; then
        echo "⚠️  输出文件已存在: ${OUTPUT_FILE}"
        read -p "是否覆盖? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  跳过"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_DATASETS+=("${dataset} (用户跳过)")
            echo ""
            continue
        fi
    fi
    
    # 执行采样
    if python sample_dataset.py \
        "${INPUT_FILE}" \
        "${OUTPUT_FILE}" \
        --max_samples ${MAX_SAMPLES} \
        --seed ${SEED}; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✅ 成功"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_DATASETS+=("${dataset} (采样失败)")
        echo "❌ 失败"
    fi
    
    echo ""
done

echo "================================================================================"
echo "采样完成"
echo "================================================================================"
echo "总数据集数: ${#DATASETS[@]}"
echo "成功: ${SUCCESS_COUNT}"
echo "失败/跳过: ${FAIL_COUNT}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "失败/跳过的数据集:"
    for failed in "${FAILED_DATASETS[@]}"; do
        echo "  - ${failed}"
    done
fi

echo "================================================================================"

exit 0
