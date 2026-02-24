#!/bin/bash

set -euo pipefail

# ===== 配置 =====
# 工作目录：使用脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR}"
# 生成一个唯一的 JOB_ID 用于日志文件命名
JOB_ID="$(date +%Y%m%d_%H%M%S)_$$"

# ===== 邮件配置 =====
# 发送邮箱（Gmail）
SENDER_EMAIL="${SENDER_EMAIL:-lingyuli513125@gmail.com}"
# 发送邮箱密码（Gmail应用密码，通过环境变量设置，建议使用 Gmail App Password）
SENDER_PASSWORD="${SENDER_PASSWORD:-eogu yaxv bhmn xluf}"
# 收件邮箱
RECIPIENT_EMAIL="${RECIPIENT_EMAIL:-lilingyu513125@163.com}"
# 如果未设置发送邮箱密码，将跳过邮件发送

# ===== GPU 检查配置 =====
# 检查间隔（秒），默认10分钟
CHECK_INTERVAL="${CHECK_INTERVAL:-600}"
# GPU利用率阈值（%），低于此值认为GPU空闲
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"
# GPU内存使用阈值（MB），低于此值认为GPU空闲（默认1GB）
GPU_MEM_THRESHOLD="${GPU_MEM_THRESHOLD:-1024}"
# 需要检查的GPU数量（默认检查所有GPU）
NUM_GPUS="${NUM_GPUS:-8}"

# ===== 创建日志目录 =====
mkdir -p logs

echo "===== JOB INFO ====="
echo "Date: $(date)"
echo "User: $USER"
echo "JobID: ${JOB_ID}"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "发送邮箱: ${SENDER_EMAIL}"
echo "收件邮箱: ${RECIPIENT_EMAIL}"
echo "邮件通知: ${SENDER_PASSWORD:+已启用}${SENDER_PASSWORD:-未启用（需要设置 SENDER_PASSWORD）}"
echo "GPU检查间隔: ${CHECK_INTERVAL}秒（${CHECK_INTERVAL}s = $((CHECK_INTERVAL / 60))分钟）"
echo "GPU利用率阈值: ${GPU_UTIL_THRESHOLD}%"
echo "GPU内存使用阈值: ${GPU_MEM_THRESHOLD}MB"
echo "需要检查的GPU数量: ${NUM_GPUS}"
echo

# ===== 设置环境变量，让 transformers 只使用 PyTorch 后端 =====
export USE_TF=0
export USE_TORCH=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_USE_TF=0
echo "已设置环境变量: USE_TF=0, USE_TORCH=1, TRANSFORMERS_NO_TF=1, TRANSFORMERS_USE_TF=0"
echo

# ===== 验证环境 =====
echo "===== 验证环境 ====="
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "警告: PyTorch 未安装或无法导入"

# 验证 huggingface-hub 版本
echo "检查 huggingface-hub..."
HF_HUB_CHECK=$(python -c "import huggingface_hub; print(f'huggingface-hub version: {huggingface_hub.__version__}'); print(f'Location: {huggingface_hub.__file__}')" 2>&1)
if [ $? -eq 0 ]; then
    echo "$HF_HUB_CHECK"
    echo "✓ huggingface-hub 版本正确"
else
    echo "⚠ huggingface-hub 检查失败"
    echo "$HF_HUB_CHECK"
fi
echo

# ===== 检查 GPU =====
echo "===== GPU 信息 ====="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo

# ===== 切换到工作目录 =====
echo "===== 切换到工作目录 ====="
echo "工作目录: $WORK_DIR"
cd "$WORK_DIR"

echo "Python 路径: $(which python || which python3 || echo 'not found')"
echo "torchrun 路径: $(which torchrun || echo 'not found')"
echo

# ===== 定义邮件发送函数 =====
send_email() {
    local subject=$1
    local body=$2
    
    # 如果未设置发送邮箱密码，跳过邮件发送
    if [ -z "$SENDER_PASSWORD" ]; then
        echo "提示: 未设置 SENDER_PASSWORD，跳过邮件通知"
        echo "提示: 请设置 Gmail 应用密码 (App Password) 来启用邮件通知"
        return 0
    fi
    
    # 使用 Python 通过 Gmail SMTP 发送邮件
    python3 << EOF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

try:
    # 创建邮件
    msg = MIMEMultipart()
    msg['From'] = "$SENDER_EMAIL"
    msg['To'] = "$RECIPIENT_EMAIL"
    msg['Subject'] = "$subject"
    
    # 添加邮件正文
    msg.attach(MIMEText("""$body""", 'plain', 'utf-8'))
    
    # 连接 Gmail SMTP 服务器
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("$SENDER_EMAIL", "$SENDER_PASSWORD")
    
    # 发送邮件
    text = msg.as_string()
    server.sendmail("$SENDER_EMAIL", "$RECIPIENT_EMAIL", text)
    server.quit()
    
    print("✓ 邮件已发送到: $RECIPIENT_EMAIL")
    sys.exit(0)
except Exception as e:
    print(f"⚠ 邮件发送失败: {e}")
    print("提示: 请检查 Gmail 应用密码是否正确设置")
    sys.exit(1)
EOF
    
    local email_result=$?
    if [ $email_result -ne 0 ]; then
        echo "⚠ 邮件发送失败，请检查配置"
    fi
}

# ===== 定义发送训练进度邮件函数 =====
send_training_progress_email() {
    local task_name=$1
    local output_dir=$2
    local task_log_dir=$3
    
    # 如果未设置发送邮箱密码，跳过邮件发送
    if [ -z "$SENDER_PASSWORD" ]; then
        echo "提示: 未设置 SENDER_PASSWORD，跳过邮件通知"
        echo "提示: 请设置 Gmail 应用密码 (App Password) 来启用邮件通知"
        return 0
    fi
    
    # 查找 training_progress.txt 文件
    local progress_file="${output_dir}/training_logs/training_progress.txt"
    
    if [ ! -f "$progress_file" ]; then
        echo "⚠ 训练进度文件不存在: $progress_file"
        echo "跳过邮件发送"
        return 0
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local subject="[训练任务] $task_name 训练完成 - $timestamp"
    
    # 读取训练进度文件的最后部分（避免邮件过大）
    local progress_content=$(tail -n 500 "$progress_file" 2>/dev/null || cat "$progress_file")
    
    local body="训练任务已成功完成

任务名称: $task_name
完成时间: $timestamp
状态: 成功

模型输出目录: $output_dir
日志目录: $task_log_dir

---
Job ID: $JOB_ID
主机名: $(hostname)
工作目录: $WORK_DIR

---
训练进度日志（最后500行）:
==========================================
$progress_content
"
    
    send_email "$subject" "$body"
}

# ===== 检查GPU是否被占用 =====
check_gpu_available() {
    # 检查 nvidia-smi 是否可用
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "⚠ nvidia-smi 不可用，假设GPU可用"
        return 0
    fi
    
    # 获取GPU利用率信息
    local gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$gpu_info" ]; then
        echo "⚠ 无法获取GPU信息，假设GPU可用"
        return 0
    fi
    
    # 检查前 NUM_GPUS 个GPU
    local all_available=true
    local gpu_count=0
    
    while IFS=, read -r index util mem_used mem_total; do
        # 去除空格
        index=$(echo "$index" | xargs)
        util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        
        if [ "$gpu_count" -ge "$NUM_GPUS" ]; then
            break
        fi
        
        gpu_count=$((gpu_count + 1))
        
        # 检查GPU利用率和内存使用情况（任一超过阈值即认为被占用）
        local is_occupied=false
        local reason=""
        
        if [ "$util" -gt "$GPU_UTIL_THRESHOLD" ]; then
            is_occupied=true
            reason="利用率 ${util}% > ${GPU_UTIL_THRESHOLD}%"
        fi
        
        if [ "$mem_used" -gt "$GPU_MEM_THRESHOLD" ]; then
            is_occupied=true
            if [ -n "$reason" ]; then
                reason="${reason}, 内存 ${mem_used}MB > ${GPU_MEM_THRESHOLD}MB"
            else
                reason="内存 ${mem_used}MB > ${GPU_MEM_THRESHOLD}MB"
            fi
        fi
        
        if [ "$is_occupied" = true ]; then
            echo "  GPU $index: $reason - 被占用"
            all_available=false
        else
            echo "  GPU $index: 利用率 ${util}%, 内存 ${mem_used}MB - 空闲"
        fi
    done <<< "$gpu_info"
    
    if [ "$all_available" = true ]; then
        return 0  # GPU可用
    else
        return 1  # GPU被占用
    fi
}

# ===== 等待GPU空闲 =====
wait_for_gpu() {
    echo "=========================================="
    echo "等待GPU空闲..."
    echo "检查间隔: ${CHECK_INTERVAL}秒"
    echo "GPU利用率阈值: ${GPU_UTIL_THRESHOLD}%"
    echo "GPU内存使用阈值: ${GPU_MEM_THRESHOLD}MB"
    echo "需要检查的GPU数量: ${NUM_GPUS}"
    echo "=========================================="
    
    local check_count=0
    
    while true; do
        check_count=$((check_count + 1))
        local current_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo
        echo "[检查 #${check_count}] $current_time"
        echo "检查GPU状态..."
        
        if check_gpu_available; then
            echo "✓ GPU已空闲，可以开始训练"
            return 0
        else
            echo "⚠ GPU仍被占用，等待 ${CHECK_INTERVAL}秒后再次检查..."
            sleep "$CHECK_INTERVAL"
        fi
    done
}

# ===== 发送GPU空闲通知 =====
send_gpu_available_notification() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local subject="[GPU通知] GPU已空闲，开始训练任务 - $timestamp"
    
    # 获取当前GPU状态
    local gpu_status=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_status=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n "$NUM_GPUS" || echo "无法获取GPU状态")
    else
        gpu_status="nvidia-smi 不可用"
    fi
    
    local body="GPU已空闲，开始训练任务

任务名称: DMSC (history_only)
开始时间: $timestamp
Job ID: $JOB_ID
主机名: $(hostname)
工作目录: $WORK_DIR

当前GPU状态:
$gpu_status

训练命令:
torchrun --nproc_per_node=8 --master_port=29500 \\
    train_distributed_MovieReview.py \\
    --config config_DMSC.json \\
    --deepspeed ds_config_zero2.json \\
    --ablation_config history_only \\
    --output_dir outputs/DMSC_history_0221_1 \\
    --max_epochs 50 \\
    --wandb_project Qwen3-DMSC \\
    --wandb_run_name history_0221_1 \\
    --prompt_style simple
"
    
    send_email "$subject" "$body"
}

# ===== 定义训练函数 =====
run_training() {
    local task_name=$1
    local master_port=$2
    shift 2
    local train_args=("$@")
    
    echo "=========================================="
    echo "开始训练任务: $task_name"
    echo "时间: $(date)"
    echo "Master Port: $master_port"
    echo "=========================================="
    
    # 创建任务特定的日志文件
    local task_log_dir="logs/task_${task_name}_${JOB_ID}"
    mkdir -p "$task_log_dir"
    
    # 从训练参数中提取 output_dir
    local output_dir=""
    for i in "${!train_args[@]}"; do
        if [ "${train_args[$i]}" = "--output_dir" ] && [ $((i+1)) -lt ${#train_args[@]} ]; then
            output_dir="${train_args[$((i+1))]}"
            break
        fi
    done
    
    # 运行训练（使用 set -e，失败会自动退出）
    if ! PYTHONNOUSERSITE=1 \
        USE_TF=0 \
        USE_TORCH=1 \
        TRANSFORMERS_NO_TF=1 \
        torchrun \
        --nproc_per_node=8 \
        --master_port="$master_port" \
        "${train_args[@]}" \
        > "$task_log_dir/train_stdout.log" 2> "$task_log_dir/train_stderr.log"; then
        echo "✗ 训练任务 $task_name 失败"
        echo "失败时间: $(date)"
        echo "查看日志: $task_log_dir/train_stderr.log"
        
        # 发送失败通知
        local fail_subject="[训练任务] $task_name 训练失败 - $(date '+%Y-%m-%d %H:%M:%S')"
        local fail_body="训练任务失败

任务名称: $task_name
失败时间: $(date '+%Y-%m-%d %H:%M:%S')
状态: 失败

日志目录: $task_log_dir
错误日志: $task_log_dir/train_stderr.log

Job ID: $JOB_ID
主机名: $(hostname)
工作目录: $WORK_DIR
"
        send_email "$fail_subject" "$fail_body"
        exit 1
    fi
    
    # 如果到这里说明训练成功
    echo "✓ 训练任务 $task_name 完成"
    echo "完成时间: $(date)"
    
    # 发送训练进度邮件
    if [ -n "$output_dir" ]; then
        echo "发送训练进度邮件..."
        send_training_progress_email "$task_name" "$output_dir" "$task_log_dir"
    else
        echo "⚠ 无法找到 output_dir，跳过邮件发送"
    fi
    
    echo
}

# ===== 主逻辑 =====
echo "=========================================="
echo "===== DMSC 训练任务 ====="
echo "=========================================="
echo

# 1. 等待GPU空闲
wait_for_gpu

# 2. 发送GPU空闲通知
echo
echo "发送GPU空闲通知..."
send_gpu_available_notification

# 3. 运行训练
echo
run_training "DMSC" 29500 \
    train_distributed_MovieReview.py \
    --config config_DMSC.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --output_dir outputs/DMSC_history_0221_1 \
    --max_epochs 50 \
    --wandb_project Qwen3-DMSC \
    --wandb_run_name history_0221_1 \
    --prompt_style simple

echo "=========================================="
echo "===== 训练任务完成 ====="
echo "完成时间: $(date)"
echo "=========================================="
