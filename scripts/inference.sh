#!/bin/bash

# HGA模型推理脚本
# 使用方法: ./inference_hga.sh [model_path]

set -e

echo "=== HGA Interactive Inference ==="
echo "Starting inference at $(date)"

# 参数设置
MODEL_PATH=${1:-"./hga_output/final_model"}
CONFIG_FILE="config.yaml"

echo "Model path: $MODEL_PATH"
echo "Config file: $CONFIG_FILE"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path $MODEL_PATH not found!"
    echo "Available models:"
    if [ -d "./hga_output" ]; then
        ls -la ./hga_output/
    else
        echo "No trained models found. Please train a model first."
    fi
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "Starting interactive inference..."
echo "Available commands:"
echo "  - Type your message to chat with the model"
echo "  - Type 'switch' to change between attacker/defender roles"
echo "  - Type 'quit' to exit"
echo "=========================================="

# 运行推理
python main.py \
    --mode inference \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --show_analysis