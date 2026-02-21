#!/bin/bash
# 分批训练示例脚本
# 用于避免一次性加载所有数据导致内存不足的问题

# 设置 Triton 缓存目录（避免磁盘空间不足）
export TRITON_CACHE_DIR="/mnt/parallel/CompactSubset_experiement/.cache/triton"
export XDG_CACHE_HOME="/mnt/parallel/CompactSubset_experiement/.cache"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"

# 配置参数
BATCH_SIZE=10000  # 每个批次的数据量
TOTAL_BATCHES=5    # 总批次数（根据实际数据量调整）

# 方法1：第一次运行 - 处理所有数据并保存，然后训练第一个批次
echo "===== 第一次运行：处理数据并训练第一批次 ====="
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed_MovieLens.py \
    --config config_MovieLens.json \
    --deepspeed ds_config_zero2.json \
    --ablation_config history_only \
    --history_strategy random \
    --history_ratio 0.5 \
    --output_dir outputs/MovieLens_history_0221_0 \
    --max_epochs 50 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --val_ratio 0.1 \
    --wandb_project Qwen3-MovieLens \
    --wandb_run_name history_0221_0_batch0 \
    --prompt_style simple \
    --data_batch_size $BATCH_SIZE \
    --data_batch_index 0

# 方法2：后续批次 - 从保存的文件加载，只训练特定批次
# 取消下面的注释来运行后续批次

# echo "===== 训练第2批次 ====="
# torchrun \
#     --nproc_per_node=8 \
#     --master_port=29501 \
#     train_distributed_MovieLens.py \
#     --config config_MovieLens.json \
#     --deepspeed ds_config_zero2.json \
#     --ablation_config history_only \
#     --history_strategy random \
#     --history_ratio 0.5 \
#     --output_dir outputs/MovieLens_history_0221_0 \
#     --max_epochs 50 \
#     --early_stopping_patience 3 \
#     --early_stopping_threshold 0.001 \
#     --val_ratio 0.1 \
#     --wandb_project Qwen3-MovieLens \
#     --wandb_run_name history_0221_0_batch1 \
#     --prompt_style simple \
#     --load_from_saved \
#     --data_batch_size $BATCH_SIZE \
#     --data_batch_index 1

# 方法3：循环训练所有批次（可选）
# for i in $(seq 0 $((TOTAL_BATCHES - 1))); do
#     echo "===== 训练第 $((i+1))/$TOTAL_BATCHES 批次 ====="
#     torchrun \
#         --nproc_per_node=8 \
#         --master_port=$((29500 + i)) \
#         train_distributed_MovieLens.py \
#         --config config_MovieLens.json \
#         --deepspeed ds_config_zero2.json \
#         --ablation_config history_only \
#         --history_strategy random \
#         --history_ratio 0.5 \
#         --output_dir outputs/MovieLens_history_0221_0 \
#         --max_epochs 50 \
#         --early_stopping_patience 3 \
#         --early_stopping_threshold 0.001 \
#         --val_ratio 0.1 \
#         --wandb_project Qwen3-MovieLens \
#         --wandb_run_name history_0221_0_batch$i \
#         --prompt_style simple \
#         --load_from_saved \
#         --data_batch_size $BATCH_SIZE \
#         --data_batch_index $i
#     
#     # 等待上一个任务完成
#     wait
# done

echo "===== 分批训练完成 ====="
