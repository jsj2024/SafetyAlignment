#!/bin/bash

# HGA评估脚本 - vLLM 加速最终版 (区分单/多轮)
# 使用方法: ./scripts/evaluate.sh [eval_type]
# 例如: bash scripts/evaluate.sh full

set -e  # 遇到错误时退出

# ==========================================================
# --- 核心参数配置 ---
# ==========================================================
EVAL_TYPE=${1:-"full"}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# --- 模型路径 ---
# 合并后的模型，用于vLLM和非LoRA脚本
ATTACKER_MERGED_MODEL_PATH="./vllm_merged_models/attacker_merged_model"
# DEFENDER_MERGED_MODEL_PATH="./vllm_merged_models/defender_merged_model"
DEFENDER_MERGED_MODEL_PATH="/home/models/Meta-Llama-3.1-8B-Instruct"
# 原始HGA LoRA模型，用于需要动态切换角色的脚本 (如 multi_jailbreak_eval)
HGA_LORA_MODEL_PATH="./hga_output/final_model"

# --- 配置文件 ---
CONFIG_FILE="/home/jsj/HGA/configs/llama3_8b_instruct.yaml"

# --- 目录和工具 ---
OUTPUT_DIR="./evaluation_results/vllm_final_differentiated"
DATASETS_DIR="./local_datasets"
# JUDGE_MODEL="/home/models/Qwen3-32B"
JUDGE_MODEL="cais/HarmBench-Llama-2-13b-cls"
TENSOR_PARALLEL_SIZE=2

# ==========================================================
# --- 脚本主逻辑 ---
# ==========================================================

echo "==================================================="
echo "HGA Model Evaluation (Full vLLM Acceleration)"
echo "==================================================="
echo "Evaluation Type: $EVAL_TYPE"
echo "Output Directory:  $OUTPUT_DIR"
echo "==================================================="

# 检查所需模型是否存在
if [[ ! -d "$DEFENDER_MERGED_MODEL_PATH" ]]; then
    echo "Error: Merged models not found. Run 'scripts/merge_hga_model.py' first."
    exit 1
fi
if [[ ! -d "$HGA_LORA_MODEL_PATH" ]]; then
    echo "Error: HGA LoRA model not found at $HGA_LORA_MODEL_PATH."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ==========================================================
# --- 辅助函数：封装评估流程 ---
# ==========================================================

# 函数1: 用于单轮攻击的 vLLM 生成 + 评判
run_vllm_single_turn_eval() {
    local BENCHMARK_NAME=$1
    local BENCHMARK_PATH=$2
    local ROLE=$3
    local MAX_SAMPLES=$4
    
    local SUB_DIR="$OUTPUT_DIR/single_turn_asr_results"
    mkdir -p "$SUB_DIR"
    
    local GENERATION_OUTPUT="$SUB_DIR/${BENCHMARK_NAME}_${ROLE}_generations.json"
    local JUDGED_OUTPUT="$SUB_DIR/${BENCHMARK_NAME}_${ROLE}_judged.json"

    echo -e "\n---> [Single-Turn ASR] Starting vLLM evaluation for '$BENCHMARK_NAME' ($ROLE)"
    
    # vLLM 生成
    python evaluation/harmbench_eval.py \
        --attacker_merged_model_path "$ATTACKER_MERGED_MODEL_PATH" \
        --defender_merged_model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --benchmark_path "$BENCHMARK_PATH" \
        --output_path "$GENERATION_OUTPUT" \
        --role "$ROLE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        ${MAX_SAMPLES:+--limit $MAX_SAMPLES}

    echo "--- [Single-Turn ASR] Judging results for '$BENCHMARK_NAME'..."
    python evaluation/judge.py \
        --file_path "$GENERATION_OUTPUT" \
        --save_path "$JUDGED_OUTPUT" \
        --judge_model "$JUDGE_MODEL" \
        --judge_type external \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
}

# 函数2: 用于多轮攻击的 vLLM 加速版本
run_multi_turn_eval() {
    local BENCHMARK_NAME=$1
    local BENCHMARK_PATH=$2
    local MAX_SAMPLES=$3
    
    local SUB_DIR="$OUTPUT_DIR/multi_turn_asr_results"
    mkdir -p "$SUB_DIR"

    echo -e "\n---> [Multi-Turn ASR] Starting vLLM evaluation for '$BENCHMARK_NAME'"
    
    local OUTPUT_FILE="$SUB_DIR/${BENCHMARK_NAME}_multi_turn_results.json"
    local JUDGED_OUTPUT="$SUB_DIR/${BENCHMARK_NAME}_multi_turn_judged.json"
    
    # 使用vLLM加速的多轮评估
    python evaluation/multi_jailbreak_eval.py \
        --attacker_merged_model_path "$ATTACKER_MERGED_MODEL_PATH" \
        --defender_merged_model_path "$DEFENDER_MERGED_MODEL_PATH" \
        --benchmark_path "$BENCHMARK_PATH" \
        --output_path "$OUTPUT_FILE" \
        --role "defender" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        ${MAX_SAMPLES:+--limit $MAX_SAMPLES}
    
    echo "--- [Multi-Turn ASR] Judging multi-turn results for '$BENCHMARK_NAME' (vLLM)..."
    # 使用vLLM judge进行多轮评估
    python evaluation/judge.py \
        --file_path "$OUTPUT_FILE" \
        --save_path "$JUDGED_OUTPUT" \
        --judge_model "$JUDGE_MODEL" \
        --judge_type external \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" 

    
    echo "--- Multi-turn evaluation completed for '$BENCHMARK_NAME'"
}

# ==========================================================
# --- 主流程：根据EVAL_TYPE选择执行任务 ---
# ==========================================================
case $EVAL_TYPE in
    "asr")
        echo "Running ASR evaluation suite (Differentiated)..."
        # 单轮
        run_vllm_single_turn_eval "harmbench" "$DATASETS_DIR/harmbench.json" "defender" 500
        # run_vllm_single_turn_eval "advbench" "$DATASETS_DIR/advbench.json" "defender" 500
        # 多轮
        # run_multi_turn_eval "actorattack" "$DATASETS_DIR/ActorAttack/Attack_test_600.json" 600
        ;;
    
    "general")
        echo "Running General Capability evaluation (vLLM)..."
        python evaluation/run_general_eval.py \
            --model_path "$DEFENDER_MERGED_MODEL_PATH" \
            --benchmarks mmlu gsm8k \
            --output_dir "$OUTPUT_DIR/general_results" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        ;;

    "overrefusal")
        echo "Running Overrefusal evaluation (vLLM)..."
        python evaluation/overrefusal_eval.py \
            --model_path "$DEFENDER_MERGED_MODEL_PATH" \
            --file_path "$DATASETS_DIR/xstest.json" \
            --judge_mode strmatch \
            --output_dir "$OUTPUT_DIR/overrefusal_results" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        ;;

    "quick")
        echo "Running QUICK evaluation suite (Differentiated)..."
        echo "--- Quick Single-Turn ASR ---"
        run_vllm_single_turn_eval "harmbench_quick" "$DATASETS_DIR/harmbench.json" "defender" 20
        echo -e "\n--- Quick Multi-Turn ASR ---"
        run_multi_turn_eval "actorattack_quick" "$DATASETS_DIR/ActorAttack/Attack_test_600.json" 20
        echo -e "\n--- Quick General Capability ---"
        python evaluation/run_general_eval.py \
            --model_path "$DEFENDER_MERGED_MODEL_PATH" \
            --benchmarks gsm8k \
            --max_samples 50 \
            --output_dir "$OUTPUT_DIR/general_results" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        ;;
        "overrefusal")
        echo "\n--- Quick Overrefusal Eval---"
        python evaluation/overrefusal_eval.py \
            --model_path "$DEFENDER_MERGED_MODEL_PATH" \
            --data_path "$DATASETS_DIR/xstest.json" \
            --judge_mode strmatch \
            --max_samples 50 \
            --output_dir "$OUTPUT_DIR/overrefusal_results" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        ;;
    
    "full")
        echo "Running FULL evaluation suite (Differentiated)..."
        
        # Step 1: 单轮 ASR 评估 (vLLM 加速)
        echo -e "\n========== Step 1/4: Single-Turn ASR Evaluation (vLLM) =========="
        run_vllm_single_turn_eval "harmbench" "$DATASETS_DIR/harmbench.json" "defender"
        run_vllm_single_turn_eval "advbench" "$DATASETS_DIR/advbench.json" "defender"
        
        # Step 2: 多轮 ASR 评估
        echo -e "\n========== Step 2/4: Multi-Turn ASR Evaluation =========="
        run_multi_turn_eval "actorattack" "$DATASETS_DIR/ActorAttack/Attack_test_600.json"
        
        # Step 3: 通用能力评估 (vLLM)
        echo -e "\n========== Step 3/4: General Capability (vLLM) =========="
        python evaluation/run_general_eval.py \
            --model_path "$DEFENDER_MERGED_MODEL_PATH" \
            --benchmarks mmlu gsm8k \
            --output_dir "$OUTPUT_DIR/general_results" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"

        # Step 4: 过度拒绝评估 (vLLM)
        echo -e "\n========== Step 4/4: Overrefusal (vLLM) =========="
        python evaluation/overrefusal_eval.py \
            --model_path "$DEFENDER_MERGED_MODEL_PATH" \
            --file_path "$DATASETS_DIR/xstest.json" \
            --judge_mode strmatch \
            --output_dir "$OUTPUT_DIR/overrefusal_results" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        ;;
    
    *)
        echo "Error: Unknown evaluation type '$EVAL_TYPE'"
        echo "Supported types: full, asr, general, overrefusal, quick"
        exit 1
        ;;
esac

# ==========================================================
# --- 最终报告生成模块 (已更新以适应新的目录结构) ---
# ==========================================================
if [[ $? -eq 0 ]]; then
    echo -e "\n\n=========================================="
    echo "All tasks completed. Generating final report..."
    echo "=========================================="

    python -c "
import json
import os
import glob
from datetime import datetime

# --- 配置 ---
output_dir = '$OUTPUT_DIR'
eval_type = '$EVAL_TYPE'
hga_lora_model_path = '$HGA_LORA_MODEL_PATH'
defender_merged_model_path = '$DEFENDER_MERGED_MODEL_PATH'
final_report_path = os.path.join(output_dir, 'final_evaluation_report.json')

# --- 辅助函数 ---
def load_json_safe(path):
    if not os.path.exists(path): return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Warning: Could not load or parse {path}: {e}')
        return None

# --- 主逻辑 ---
def generate_report():
    print(f'Collecting results from: {output_dir}')
    
    report = {
        'hga_lora_model_path': hga_lora_model_path,
        'defender_merged_model_path': defender_merged_model_path,
        'evaluation_type': eval_type,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }

    # 1. 收集单轮 ASR 结果
    single_turn_asr = {}
    judged_files = glob.glob(os.path.join(output_dir, 'single_turn_asr_results', '*_judged.json'))
    for f_path in judged_files:
        judged_data = load_json_safe(f_path)
        if judged_data and 'score' in judged_data:
            benchmark_name = os.path.basename(f_path).replace('_judged.json', '')
            single_turn_asr[benchmark_name] = {'ASR_score': judged_data['score']}
    if single_turn_asr:
        report['results']['Single_Turn_ASR'] = single_turn_asr

    # 2. 收集多轮 ASR 结果
    multi_turn_asr = {}
    multi_turn_files = glob.glob(os.path.join(output_dir, 'multi_turn_asr_results', 'results_*.json'))
    if multi_turn_files:
        # 假设只运行了一次，取最新的那个文件
        latest_file = max(multi_turn_files, key=os.path.getmtime)
        multi_data = load_json_safe(latest_file)
        if multi_data and 'statistics' in multi_data:
            stats = multi_data['statistics']
            multi_turn_asr['actorattack'] = {
                'judge_attack_success_rate': stats.get('judge_attack_success_rate'),
                'avg_dialogue_length': stats.get('avg_dialogue_length')
            }
    if multi_turn_asr:
        report['results']['Multi_Turn_ASR'] = multi_turn_asr

    # 3. 收集通用能力结果
    general_summary_path = os.path.join(output_dir, 'general_results', 'general_evaluation_summary.json')
    general_data = load_json_safe(general_summary_path)
    if general_data:
        report['results']['General_Capability'] = general_data
        
    # 4. 收集过度拒绝结果
    overrefusal_stats_path = os.path.join(output_dir, 'overrefusal_results', 'overrefusal_statistics.json')
    overrefusal_data = load_json_safe(overrefusal_stats_path)
    if overrefusal_data:
        report['results']['Overrefusal'] = overrefusal_data

    # 保存最终报告
    with open(final_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f'\\n✅ Final evaluation report generated successfully!')
    print(f'==> {final_report_path}')

if __name__ == '__main__':
    generate_report()
"
else
    echo -e "\n\n=========================================="
    echo "Evaluation failed with exit code $?"
    echo "Check the logs above for error details."
    echo "=========================================="
    exit 1
fi