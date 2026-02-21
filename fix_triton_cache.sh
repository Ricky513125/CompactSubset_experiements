#!/bin/bash
# 修复 DeepSpeed Triton 缓存目录空间不足的问题
# 将缓存目录重定向到 /mnt/parallel（有充足空间）

# 设置 Triton 缓存目录到有空间的位置
export TRITON_CACHE_DIR="/mnt/parallel/CompactSubset_experiement/.cache/triton"
export XDG_CACHE_HOME="/mnt/parallel/CompactSubset_experiement/.cache"

# 确保缓存目录存在
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"

echo "✓ Triton 缓存目录已设置为: $TRITON_CACHE_DIR"
echo "✓ XDG 缓存目录已设置为: $XDG_CACHE_HOME"
echo ""
echo "使用方法："
echo "  source fix_triton_cache.sh"
echo "  然后运行您的训练命令"
echo ""
echo "或者直接在训练命令前添加环境变量："
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR XDG_CACHE_HOME=$XDG_CACHE_HOME torchrun ..."
