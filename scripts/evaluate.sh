#!/bin/bash

# HGA评估脚本 - 使用独特数据集评估方法 + vLLM加速
# 使用方法: ./scripts/evaluate.sh [dataset_name] [options]
# 支持的数据集: harmbench, advbench, actorattack, gsm8k, mmlu, redqueen, xstest, all
# 例如: bash scripts/evaluate.sh harmbench --max_samples 100

set -e  # 遇到错误时退出
export CUDA_VISIBLE_DEVICES=4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1


DATASET_NAME=${1:-"all"}

# --- 模型路径 ---
ATTACKER_MERGED_MODEL_PATH="./vllm_merged_models/attacker_merged_model"
DEFENDER_MERGED_MODEL_PATH="./vllm_merged_models/defender_merged_model"
# DEFENDER_MERGED_MODEL_PATH="/home/models/Meta-Llama-3.1-8B-Instruct"
HGA_LORA_MODEL_PATH="./hga_output/final_model"

# --- 配置文件 ---
CONFIG_FILE="/home/jsj/HGA/configs/llama3_8b_instruct.yaml"

# --- 目录和工具 ---
OUTPUT_DIR="./evaluation_results/dataset_specific_vllm"
DATASETS_DIR="./local_datasets"
# JUDGE_MODEL="/home/models/Qwen3-32B"
JUDGE_MODEL="cais/HarmBench-Llama-2-13b-cls"
TENSOR_PARALLEL_SIZE=2

# --- 解析额外参数 ---
MAX_SAMPLES=""
OUTPUT_PATH=""
JUDGE_MODE="external"
USE_STRING_MATCHING=false
ROLE="defender"
K_SHOT=""


# 解析命令行参数
shift  # 跳过第一个参数（数据集名称）
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --judge_mode)
            JUDGE_MODE="$2"
            shift 2
            ;;
        --string_matching)
            USE_STRING_MATCHING=true
            shift
            ;;
        --tensor_parallel_size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --role)
            ROLE="$2"
            shift 2
            ;;
        --k_shot)
            K_SHOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==========================================================
# --- 脚本主逻辑 ---
# ==========================================================

echo "==================================================="
echo "HGA Dataset-Specific vLLM Evaluation"
echo "==================================================="
echo "Dataset: $DATASET_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "Max Samples: ${MAX_SAMPLES:-'All'}"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Judge Model: $JUDGE_MODEL"
echo "==================================================="

# 检查所需模型是否存在
if [[ ! -d "$DEFENDER_MERGED_MODEL_PATH" ]]; then
    echo "Error: Merged models not found. Run 'scripts/merge_hga_model.py' first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ==========================================================
# --- 数据集特定评估函数 ---
# ==========================================================

# HarmBench评估 - 使用独特的HarmBench评估方法
evaluate_harmbench() {
    echo "=== Evaluating HarmBench (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/harmbench_results.json}"
    
    python -m evaluation.datasets.harmbench_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --judge_model "$JUDGE_MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        --subset standard \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
    
    echo "HarmBench evaluation completed. Results saved to: $output_file"
}

# AdvBench评估 - 使用独特的AdvBench评估方法
evaluate_advbench() {
    echo "=== Evaluating AdvBench (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/advbench_results.json}"
    
    python -m evaluation.datasets.advbench_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --judge_model "$JUDGE_MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        ${USE_STRING_MATCHING:+--use_string_matching}
    
    echo "AdvBench evaluation completed. Results saved to: $output_file"
}

# ActorAttack评估 - 使用独特的ActorAttack评估方法
evaluate_actorattack() {
    echo "=== Evaluating ActorAttack (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/actorattack_results.json}"
    
    python -m evaluation.datasets.actorattack_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --judge_model "$JUDGE_MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        --split test \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
    
    echo "ActorAttack evaluation completed. Results saved to: $output_file"
}

# GSM8K评估 - 使用独特的GSM8K评估方法
evaluate_gsm8k() {
    echo "=== Evaluating GSM8K (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/gsm8k_results.json}"
    local k_shot_param=${K_SHOT:-8}
    
    python -m evaluation.datasets.gsm8k_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        --split test \
        --k_shot "$k_shot_param" \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
    
    echo "GSM8K evaluation completed. Results saved to: $output_file"
}

# MMLU评估 - 使用独特的MMLU评估方法
evaluate_mmlu() {
    echo "=== Evaluating MMLU (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/mmlu_results.json}"
    local k_shot_param=${K_SHOT:-5}
    
    python -m evaluation.datasets.mmlu_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        --k_shot "$k_shot_param" \
        ${MAX_SAMPLES:+--max_samples_per_subject $MAX_SAMPLES}
    
    echo "MMLU evaluation completed. Results saved to: $output_file"
}

# RedQueen评估 - 使用独特的RedQueen评估方法
evaluate_redqueen() {
    echo "=== Evaluating RedQueen (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/redqueen_results.json}"
    
    python -m evaluation.datasets.redqueen_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --judge_model "$JUDGE_MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        --split test \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
    
    echo "RedQueen evaluation completed. Results saved to: $output_file"
}

# XSTest评估 - 使用独特的XSTest评估方法
evaluate_xstest() {
    echo "=== Evaluating XSTest (Dataset-Specific vLLM) ==="
    local output_file="${OUTPUT_PATH:-$OUTPUT_DIR/xstest_results.json}"
    local judge_mode_param=${JUDGE_MODE:-"keyword"}
    
    python -m evaluation.datasets.xstest_eval \
        --model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output_path "$output_file" \
        --judge_mode "$judge_mode_param" \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
    
    echo "XSTest evaluation completed. Results saved to: $output_file"
}

# 生成数据集特定的综合报告
generate_dataset_specific_report() {
    echo "=== Generating Dataset-Specific vLLM Report ==="
    
    python -c "
import json
import os
import glob
from datetime import datetime

# 配置
output_dir = '$OUTPUT_DIR'
report_path = os.path.join(output_dir, 'dataset_specific_vllm_summary.json')

def load_json_safe(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Warning: Could not load {path}: {e}')
        return None

def extract_key_metrics(data, dataset_name):
    '''提取每个数据集的关键指标'''
    metrics = {}
    
    if dataset_name == 'harmbench':
        metrics['overall_asr'] = data.get('overall_asr', 0)
        metrics['successful_attacks'] = data.get('successful_attacks', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
        if 'category_metrics' in data:
            metrics['category_breakdown'] = data['category_metrics']
    
    elif dataset_name == 'advbench':
        metrics['overall_asr'] = data.get('overall_asr', 0)
        metrics['successful_attacks'] = data.get('successful_attacks', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
        metrics['evaluation_method'] = data.get('evaluation_method', 'unknown')
    
    elif dataset_name == 'actorattack':
        metrics['judge_attack_success_rate'] = data.get('judge_attack_success_rate', 0)
        metrics['successful_attacks'] = data.get('successful_attacks', 0)
        metrics['avg_dialogue_length'] = data.get('avg_dialogue_length', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
    
    elif dataset_name == 'gsm8k':
        metrics['accuracy'] = data.get('accuracy', 0)
        metrics['correct_answers'] = data.get('correct_answers', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
        metrics['k_shot'] = data.get('k_shot', 0)
    
    elif dataset_name == 'mmlu':
        metrics['overall_accuracy'] = data.get('overall_accuracy', 0)
        metrics['total_correct'] = data.get('total_correct', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
        metrics['k_shot'] = data.get('k_shot', 0)
        if 'category_accuracies' in data:
            metrics['category_accuracies'] = data['category_accuracies']
    
    elif dataset_name == 'redqueen':
        metrics['judge_attack_success_rate'] = data.get('judge_attack_success_rate', 0)
        metrics['successful_attacks'] = data.get('successful_attacks', 0)
        metrics['avg_dialogue_length'] = data.get('avg_dialogue_length', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
    
    elif dataset_name == 'xstest':
        if 'metrics' in data:
            metrics['over_refusal_rate'] = data['metrics'].get('overall_over_refusal_rate', 0)
            metrics['safe_over_refusal_rate'] = data['metrics'].get('safe_over_refusal_rate', 0)
            metrics['unsafe_refusal_rate'] = data['metrics'].get('unsafe_refusal_rate', 0)
        metrics['total_samples'] = data.get('total_samples', 0)
        metrics['judge_mode'] = data.get('judge_mode', 'unknown')
    
    return metrics

# 收集所有结果文件
result_files = glob.glob(os.path.join(output_dir, '*_results.json'))

summary = {
    'evaluation_time': datetime.now().isoformat(),
    'evaluation_type': 'dataset_specific_vllm',
    'model_path': '$DEFENDER_MERGED_MODEL_PATH',
    'judge_model': '$JUDGE_MODEL',
    'tensor_parallel_size': $TENSOR_PARALLEL_SIZE,
    'datasets_evaluated': [],
    'results': {}
}

# 数据集分类
safety_datasets = ['harmbench', 'advbench', 'actorattack', 'redqueen']
capability_datasets = ['gsm8k', 'mmlu']
overrefusal_datasets = ['xstest']

total_samples = 0
safety_results = {}
capability_results = {}
overrefusal_results = {}

for file_path in result_files:
    dataset_name = os.path.basename(file_path).replace('_results.json', '')
    data = load_json_safe(file_path)
    
    if data:
        summary['datasets_evaluated'].append(dataset_name)
        
        # 提取数据集特定的关键指标
        key_metrics = extract_key_metrics(data, dataset_name)
        summary['results'][dataset_name] = key_metrics
        
        # 累计总样本数
        total_samples += key_metrics.get('total_samples', 0)
        
        # 按类别分组
        if dataset_name in safety_datasets:
            safety_results[dataset_name] = key_metrics
        elif dataset_name in capability_datasets:
            capability_results[dataset_name] = key_metrics
        elif dataset_name in overrefusal_datasets:
            overrefusal_results[dataset_name] = key_metrics

# 添加分类汇总
summary['category_summary'] = {
    'safety_evaluation': safety_results,
    'capability_evaluation': capability_results,
    'overrefusal_evaluation': overrefusal_results
}

# 添加统计信息
summary['evaluation_statistics'] = {
    'total_samples_evaluated': total_samples,
    'safety_datasets_count': len(safety_results),
    'capability_datasets_count': len(capability_results),
    'overrefusal_datasets_count': len(overrefusal_results),
    'total_datasets_count': len(summary['datasets_evaluated'])
}

# 保存摘要
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\\nDataset-Specific vLLM Summary report generated: {report_path}')

# 打印详细统计
print('\\n=== Dataset-Specific vLLM Evaluation Summary ===')
print(f'Model: {summary[\"model_path\"]}')
print(f'Judge Model: {summary[\"judge_model\"]}')
print(f'Tensor Parallel Size: {summary[\"tensor_parallel_size\"]}')
print(f'Total Samples Evaluated: {total_samples}')
print(f'Total Datasets: {len(summary[\"datasets_evaluated\"])}')
print()

# 安全评估结果
if safety_results:
    print('=== Safety Evaluation Results ===')
    for dataset, metrics in safety_results.items():
        print(f'{dataset.upper()}:')
        if 'overall_asr' in metrics:
            print(f'  Attack Success Rate: {metrics[\"overall_asr\"]:.1%}')
        if 'judge_attack_success_rate' in metrics:
            print(f'  Judge Attack Success Rate: {metrics[\"judge_attack_success_rate\"]:.1%}')
        if 'successful_attacks' in metrics:
            print(f'  Successful Attacks: {metrics[\"successful_attacks\"]}')
        if 'avg_dialogue_length' in metrics:
            print(f'  Avg Dialogue Length: {metrics[\"avg_dialogue_length\"]:.1f}')
        print(f'  Total Samples: {metrics.get(\"total_samples\", 0)}')
        print()

# 能力评估结果
if capability_results:
    print('=== Capability Evaluation Results ===')
    for dataset, metrics in capability_results.items():
        print(f'{dataset.upper()}:')
        if 'accuracy' in metrics:
            print(f'  Accuracy: {metrics[\"accuracy\"]:.1%}')
        if 'overall_accuracy' in metrics:
            print(f'  Overall Accuracy: {metrics[\"overall_accuracy\"]:.1%}')
        if 'correct_answers' in metrics:
            print(f'  Correct Answers: {metrics[\"correct_answers\"]}')
        if 'total_correct' in metrics:
            print(f'  Total Correct: {metrics[\"total_correct\"]}')
        if 'k_shot' in metrics:
            print(f'  K-shot: {metrics[\"k_shot\"]}')
        print(f'  Total Samples: {metrics.get(\"total_samples\", 0)}')
        print()

# 过度拒绝评估结果
if overrefusal_results:
    print('=== Over-refusal Evaluation Results ===')
    for dataset, metrics in overrefusal_results.items():
        print(f'{dataset.upper()}:')
        if 'over_refusal_rate' in metrics:
            print(f'  Over-refusal Rate: {metrics[\"over_refusal_rate\"]:.1%}')
        if 'safe_over_refusal_rate' in metrics:
            print(f'  Safe Over-refusal Rate: {metrics[\"safe_over_refusal_rate\"]:.1%}')
        if 'unsafe_refusal_rate' in metrics:
            print(f'  Unsafe Refusal Rate: {metrics[\"unsafe_refusal_rate\"]:.1%}')
        if 'judge_mode' in metrics:
            print(f'  Judge Mode: {metrics[\"judge_mode\"]}')
        print(f'  Total Samples: {metrics.get(\"total_samples\", 0)}')
        print()

print('=== Evaluation Complete ===')
"
}

# 检查各数据集评估模块是否存在
# check_evaluation_modules() {
#     echo "=== Checking Dataset-Specific Evaluation Modules ==="
    
#     local modules=(
#         "evaluation.datasets.harmbench_eval"
#         "evaluation.datasets.advbench_eval"
#         "evaluation.datasets.actorattack_eval"
#         "evaluation.datasets.gsm8k_eval"
#         "evaluation.datasets.mmlu_eval"
#         "evaluation.datasets.redqueen_eval"
#         "evaluation.datasets.xstest_eval"
#     )
    
#     for module in "${modules[@]}"; do
#         if python -c "import $module" 2>/dev/null; then
#             echo "✓ $module - Available"
#         else
#             echo "✗ $module - Missing"
#             return 1
#         fi
#     done
    
#     echo "All evaluation modules are available!"
# }

# 检查vLLM和数据集环境
check_environment() {
    echo "=== Checking vLLM and Dataset Environment ==="
    
    # 检查vLLM
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
        echo "Error: vLLM not installed or not accessible"
        exit 1
    }
    
    # 检查CUDA
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" || {
        echo "Warning: CUDA check failed"
    }
    
    # 检查模型文件
    if [[ ! -f "$DEFENDER_MERGED_MODEL_PATH/config.json" ]]; then
        echo "Error: Model config not found. Please check model path."
        exit 1
    fi
    
    # 检查数据集目录
    local datasets=("harmbench.json" "advbench.json" "xstest.json")
    local dataset_dirs=("ActorAttack" "RedQueen")
    
    for dataset in "${datasets[@]}"; do
        if [[ ! -f "$DATASETS_DIR/$dataset" ]]; then
            echo "Warning: Dataset $dataset not found at $DATASETS_DIR"
        fi
    done
    
    for dataset_dir in "${dataset_dirs[@]}"; do
        if [[ ! -d "$DATASETS_DIR/$dataset_dir" ]]; then
            echo "Warning: Dataset directory $dataset_dir not found at $DATASETS_DIR"
        fi
    done
    
    echo "Environment check completed!"
}

# ==========================================================
# --- 主评估逻辑 ---
# ==========================================================

# 先检查环境和模块
check_environment
# check_evaluation_modules

case $DATASET_NAME in
    "harmbench")
        evaluate_harmbench
        ;;
    "advbench")
        evaluate_advbench
        ;;
    "actorattack")
        evaluate_actorattack
        ;;
    "gsm8k")
        evaluate_gsm8k
        ;;
    "mmlu")
        evaluate_mmlu
        ;;
    "redqueen")
        evaluate_redqueen
        ;;
    "xstest")
        evaluate_xstest
        ;;
    "all")
        echo "Running all dataset-specific vLLM evaluations..."
        evaluate_harmbench
        evaluate_advbench
        evaluate_actorattack
        evaluate_gsm8k
        evaluate_mmlu
        evaluate_redqueen
        evaluate_xstest
        generate_dataset_specific_report
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET_NAME'"
        echo "Supported datasets: harmbench, advbench, actorattack, gsm8k, mmlu, redqueen, xstest, all"
        echo ""
        echo "Usage examples:"
        echo "  bash scripts/evaluate.sh harmbench --max_samples 100"
        echo "  bash scripts/evaluate.sh gsm8k --max_samples 50 --k_shot 8"
        echo "  bash scripts/evaluate.sh mmlu --max_samples 100 --k_shot 5"
        echo "  bash scripts/evaluate.sh all --tensor_parallel_size 4"
        echo "  bash scripts/evaluate.sh xstest --judge_mode keyword"
        echo "  bash scripts/evaluate.sh advbench --string_matching"
        exit 1
        ;;
esac

echo ""
echo "==================================================="
echo "Dataset-Specific vLLM Evaluation completed successfully!"
echo "Results saved in: $OUTPUT_DIR"
echo "Using dataset-specific evaluation methods with vLLM acceleration"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "==================================================="