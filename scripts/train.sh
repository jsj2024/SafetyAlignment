#!/bin/bash
set -e
echo "=== HGA训练启动 ==="
echo "启动时间: $(date)"

if nvidia-smi > /dev/null 2>&1; then
    echo "✓ CUDA可用"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "检测到 $GPU_COUNT 个GPU"
else
    echo "⚠️ CUDA不可用，将使用CPU"
    GPU_COUNT=0
fi

# 环境变量设置
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5
# 根据GPU数量选择训练策略
if [ $GPU_COUNT -ge 2 ]; then
    echo "使用分布式训练 (2个GPU)"
    export CUDA_VISIBLE_DEVICES=4,5

    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    
    # 分布式训练启动
    torchrun \
        --nproc_per_node=2 \
        --master_port=12355 \
        main.py \
        --mode train \
        --config configs/llama3_8b_instruct.yaml \
        --output_dir ./hga_output_0711
        
elif [ $GPU_COUNT -eq 1 ]; then
    echo "使用单GPU训练"
    export CUDA_VISIBLE_DEVICES=4
    
    python main.py \
        --mode train \
        --config configs/llama3_8b_instruct.yaml \
        --output_dir ./hga_output_0711
        
else
    echo "使用CPU训练"
    export CUDA_VISIBLE_DEVICES=""
    
    python main.py \
        --mode train \
        --config configs/llama3_8b_instruct.yaml \
        --output_dir ./hga_output_0711
fi

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ 训练完成!"
    echo "结果保存在: ./hga_output_0711"
    ls -la ./hga_output_0711
else
    echo "❌ 训练失败!"
    exit 1
fi

echo "训练结束时间: $(date)"