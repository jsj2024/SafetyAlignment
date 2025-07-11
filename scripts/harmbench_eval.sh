#!/bin/bash

# HarmBench模型评估运行脚本
# 用法: ./run_eval.sh [选项]

set -e  # 遇到错误立即退出

# 默认配置
MODEL_NAME=""
DATASET_CONFIG="standard"
OUTPUT_DIR="./eval_results"
USE_VLLM=false
MAX_NEW_TOKENS=512
NUM_GPUS=1
BATCH_SIZE=8
NUM_SAMPLES=""
BEHAVIORS_FILE=""
SAVE_COMPLETIONS=false
VERBOSE=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助信息
print_help() {
    echo -e "${BLUE}HarmBench模型评估脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "必需参数:"
    echo "  -m, --model MODEL_PATH          要评估的模型路径或名称"
    echo ""
    echo "可选参数:"
    echo "  -d, --dataset CONFIG            数据集配置 [standard|contextual|copyright] (默认: standard)"
    echo "  -o, --output DIR                输出目录 (默认: ./eval_results)"
    echo "  -f, --behaviors-file FILE       自定义behaviors CSV文件路径"
    echo "  -n, --num-samples NUM           限制评估样本数量"
    echo "  --use-vllm                      使用vLLM进行推理加速"
    echo "  --max-tokens NUM                最大生成token数 (默认: 512)"
    echo "  --num-gpus NUM                  GPU数量 (默认: 1)"
    echo "  --batch-size NUM                批次大小 (默认: 8)"
    echo "  --save-completions              保存生成的completions"
    echo "  --verbose                       显示详细输出"
    echo "  -h, --help                      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 评估标准数据集"
    echo "  $0 -m microsoft/DialoGPT-medium -d standard"
    echo ""
    echo "  # 使用vLLM加速评估"
    echo "  $0 -m meta-llama/Llama-2-7b-chat-hf --use-vllm --num-gpus 2"
    echo ""
    echo "  # 评估自定义数据集"
    echo "  $0 -m your-model -f ./custom_behaviors.csv --save-completions"
    echo ""
    echo "  # 限制样本数量进行快速测试"
    echo "  $0 -m your-model -n 50 --verbose"
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}检查依赖...${NC}"
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}错误: 未找到Python${NC}"
        exit 1
    fi
    
    # 检查必要的Python包
    python -c "import torch, transformers, vllm, datasets" 2>/dev/null || {
        echo -e "${RED}错误: 缺少必要的Python包。请安装: torch, transformers, vllm, datasets${NC}"
        exit 1
    }
    
    echo -e "${GREEN}依赖检查通过${NC}"
}

# 检查GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        echo -e "${BLUE}检测到 ${GPU_COUNT} 个GPU${NC}"
        
        if [ $NUM_GPUS -gt $GPU_COUNT ]; then
            echo -e "${YELLOW}警告: 请求的GPU数量($NUM_GPUS)超过可用数量($GPU_COUNT)${NC}"
            NUM_GPUS=$GPU_COUNT
        fi
    else
        echo -e "${YELLOW}警告: 未检测到NVIDIA GPU${NC}"
    fi
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model)
                MODEL_NAME="$2"
                shift 2
                ;;
            -d|--dataset)
                DATASET_CONFIG="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -f|--behaviors-file)
                BEHAVIORS_FILE="$2"
                shift 2
                ;;
            -n|--num-samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --use-vllm)
                USE_VLLM=true
                shift
                ;;
            --max-tokens)
                MAX_NEW_TOKENS="$2"
                shift 2
                ;;
            --num-gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --save-completions)
                SAVE_COMPLETIONS=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                echo -e "${RED}未知参数: $1${NC}"
                print_help
                exit 1
                ;;
        esac
    done
}

# 验证参数
validate_args() {
    if [ -z "$MODEL_NAME" ]; then
        echo -e "${RED}错误: 必须指定模型名称 (-m/--model)${NC}"
        print_help
        exit 1
    fi
    
    if [[ ! "$DATASET_CONFIG" =~ ^(standard|contextual|copyright)$ ]]; then
        echo -e "${RED}错误: 无效的数据集配置: $DATASET_CONFIG${NC}"
        exit 1
    fi
    
    if [ -n "$BEHAVIORS_FILE" ] && [ ! -f "$BEHAVIORS_FILE" ]; then
        echo -e "${RED}错误: behaviors文件不存在: $BEHAVIORS_FILE${NC}"
        exit 1
    fi
}

# 构建Python命令
build_command() {
    CMD="python ./evaluation/harmbench_eval.py"
    CMD="$CMD --model_name_or_path '$MODEL_NAME'"
    CMD="$CMD --dataset_config '$DATASET_CONFIG'"
    CMD="$CMD --output_dir '$OUTPUT_DIR'"
    CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
    CMD="$CMD --num_gpus $NUM_GPUS"
    CMD="$CMD --batch_size $BATCH_SIZE"
    
    if [ "$USE_VLLM" = true ]; then
        CMD="$CMD --use_vllm"
    fi
    
    if [ -n "$BEHAVIORS_FILE" ]; then
        CMD="$CMD --behaviors_file '$BEHAVIORS_FILE'"
    fi
    
    if [ -n "$NUM_SAMPLES" ]; then
        CMD="$CMD --num_samples $NUM_SAMPLES"
    fi
    
    if [ "$SAVE_COMPLETIONS" = true ]; then
        CMD="$CMD --save_completions"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
}

# 显示配置摘要
show_summary() {
    echo -e "${BLUE}评估配置摘要:${NC}"
    echo "=================================="
    echo "模型: $MODEL_NAME"
    echo "数据集配置: $DATASET_CONFIG"
    echo "输出目录: $OUTPUT_DIR"
    echo "使用vLLM: $USE_VLLM"
    echo "最大token数: $MAX_NEW_TOKENS"
    echo "GPU数量: $NUM_GPUS"
    echo "批次大小: $BATCH_SIZE"
    
    if [ -n "$BEHAVIORS_FILE" ]; then
        echo "自定义behaviors文件: $BEHAVIORS_FILE"
    fi
    
    if [ -n "$NUM_SAMPLES" ]; then
        echo "样本数量限制: $NUM_SAMPLES"
    fi
    
    echo "保存completions: $SAVE_COMPLETIONS"
    echo "详细输出: $VERBOSE"
    echo "=================================="
    echo ""
}

# 创建输出目录
create_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    echo -e "${GREEN}输出目录已创建: $OUTPUT_DIR${NC}"
}

# 运行评估
run_evaluation() {
    echo -e "${BLUE}开始运行评估...${NC}"
    echo "命令: $CMD"
    echo ""
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 执行命令
    eval $CMD
    EXIT_CODE=$?
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}评估成功完成!${NC}"
        echo -e "${BLUE}耗时: ${DURATION} 秒${NC}"
        echo -e "${BLUE}结果保存在: $OUTPUT_DIR${NC}"
        
        # 显示结果文件
        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            echo ""
            echo -e "${BLUE}主要指标:${NC}"
            python -c "
import json
try:
    with open('$OUTPUT_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    print(f\"总体ASR: {metrics.get('asr', 0):.4f} ({metrics.get('asr', 0)*100:.2f}%)\")
    print(f\"样本数: {metrics.get('total_samples', 0)}\")
    print(f\"成功攻击: {metrics.get('successful_attacks', 0)}\")
except Exception as e:
    print(f'无法读取指标文件: {e}')
"
        fi
    else
        echo ""
        echo -e "${RED}评估失败，退出码: $EXIT_CODE${NC}"
        exit $EXIT_CODE
    fi
}

# 主函数
main() {
    echo -e "${BLUE}HarmBench模型评估脚本${NC}"
    echo "=================================="
    
    parse_args "$@"
    validate_args
    check_dependencies
    check_gpu
    build_command
    show_summary
    create_output_dir
    run_evaluation
}

# 捕获中断信号
trap 'echo -e "\n${YELLOW}评估被中断${NC}"; exit 130' INT

# 运行主函数
main "$@"