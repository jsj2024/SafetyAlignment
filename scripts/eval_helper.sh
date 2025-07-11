#!/bin/bash

# HGA评估助手脚本 - 数据集特定vLLM加速版
# 使用每个数据集独特的评估方法

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的文本
print_colored() {
    echo -e "${1}${2}${NC}"
}

# 显示帮助信息
show_help() {
    print_colored $BLUE "=============================================="
    print_colored $BLUE "HGA Dataset-Specific vLLM Evaluation Helper"
    print_colored $BLUE "=============================================="
    echo ""
    print_colored $YELLOW "Available Datasets (with unique evaluation methods):"
    print_colored $CYAN "Security Datasets:"
    echo "  harmbench    - 安全行为评估 (多类别ASR分析)"
    echo "  advbench     - 对抗性行为评估 (字符串匹配可选)"
    echo "  actorattack  - 多轮对话攻击评估 (格式化查询)"
    echo "  redqueen     - 多轮对话安全评估 (难度分析)"
    print_colored $CYAN "Capability Datasets:"
    echo "  gsm8k        - 数学推理能力评估 (8-shot, 答案提取)"
    echo "  mmlu         - 多学科知识评估 (5-shot, 概率分析)"
    print_colored $CYAN "Safety Datasets:"
    echo "  xstest       - 过度拒绝评估 (关键词/LLM判断)"
    echo "  all          - 运行所有数据集特定评估"
    echo ""
    print_colored $YELLOW "Usage:"
    echo "  $0 [dataset] [options]"
    echo ""
    print_colored $YELLOW "Options:"
    echo "  --max_samples N        限制评估样本数"
    echo "  --output_path PATH     指定输出文件路径"
    echo "  --judge_mode MODE      Judge模式 (external/keyword)"
    echo "  --string_matching      使用字符串匹配判断 (仅advbench)"
    echo "  --tensor_parallel_size N  设置tensor并行数 (默认2)"
    echo "  --k_shot N             设置few-shot数量 (gsm8k/mmlu)"
    echo "  --role ROLE            设置角色 (attacker/defender)"
    echo ""
    print_colored $PURPLE "Dataset-Specific Features:"
    echo "  每个数据集使用独特的评估逻辑"
    echo "  保持原始论文的评估方法"
    echo "  vLLM加速所有推理过程"
    echo "  详细的数据集特定指标"
    echo ""
    print_colored $YELLOW "Examples:"
    echo "  $0 harmbench --max_samples 100 --tensor_parallel_size 4"
    echo "  $0 gsm8k --max_samples 50 --k_shot 8"
    echo "  $0 mmlu --max_samples 100 --k_shot 5"
    echo "  $0 xstest --judge_mode keyword"
    echo "  $0 advbench --string_matching"
    echo "  $0 actorattack --max_samples 50"
    echo "  $0 all --tensor_parallel_size 2"
    echo ""
}

# 快速评估命令 - 数据集特定vLLM加速版
quick_eval() {
    local dataset=$1
    print_colored $GREEN "Running dataset-specific quick vLLM evaluation for $dataset..."
    
    case $dataset in
        "harmbench")
            bash scripts/evaluate.sh harmbench --max_samples 50 --tensor_parallel_size 2
            ;;
        "advbench")
            bash scripts/evaluate.sh advbench --max_samples 50 --string_matching --tensor_parallel_size 2
            ;;
        "actorattack")
            bash scripts/evaluate.sh actorattack --max_samples 20 --tensor_parallel_size 2
            ;;
        "gsm8k")
            bash scripts/evaluate.sh gsm8k --max_samples 100 --k_shot 8 --tensor_parallel_size 2
            ;;
        "mmlu")
            bash scripts/evaluate.sh mmlu --max_samples 50 --k_shot 5 --tensor_parallel_size 2
            ;;
        "redqueen")
            bash scripts/evaluate.sh redqueen --max_samples 20 --tensor_parallel_size 2
            ;;
        "xstest")
            bash scripts/evaluate.sh xstest --max_samples 100 --judge_mode keyword --tensor_parallel_size 2
            ;;
        *)
            print_colored $RED "Unknown dataset: $dataset"
            return 1
            ;;
    esac
}

# 完整评估命令
full_eval() {
    local dataset=$1
    print_colored $GREEN "Running full evaluation for $dataset..."
    
    bash scripts/evaluate.sh $dataset
}

# 批量快速评估 - vLLM加速版
batch_quick_eval() {
    local datasets=("harmbench" "advbench" "gsm8k" "xstest")
    
    print_colored $GREEN "Running batch quick vLLM evaluation..."
    print_colored $BLUE "Using tensor_parallel_size=2 for optimal performance"
    
    for dataset in "${datasets[@]}"; do
        print_colored $BLUE "Starting vLLM $dataset evaluation..."
        quick_eval $dataset
        print_colored $GREEN "vLLM $dataset evaluation completed"
        echo ""
    done
    
    print_colored $GREEN "All batch vLLM evaluations completed!"
    print_colored $BLUE "Generating summary report..."
    bash scripts/evaluate.sh all > /dev/null 2>&1  # 只生成报告
}

# 性能测试函数
performance_test() {
    local dataset=${1:-"harmbench"}
    local samples=${2:-10}
    
    print_colored $GREEN "Running vLLM performance test for $dataset..."
    print_colored $BLUE "Samples: $samples"
    
    # 测试不同的tensor parallel size
    for tp_size in 1 2 4; do
        print_colored $YELLOW "Testing tensor_parallel_size=$tp_size"
        
        local start_time=$(date +%s)
        bash scripts/evaluate.sh $dataset --max_samples $samples --tensor_parallel_size $tp_size >/dev/null 2>&1
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_colored $GREEN "  Time taken: ${duration}s"
    done
}

# 检查数据集特定的vLLM环境
check_environment() {
    print_colored $YELLOW "Checking dataset-specific vLLM environment..."
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        print_colored $RED "Python not found!"
        return 1
    fi
    
    # 检查vLLM
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null || {
        print_colored $RED "vLLM not found! Please install vLLM"
        return 1
    }
    
    # 检查数据集特定的评估模块
    print_colored $BLUE "Checking dataset-specific evaluation modules..."
    local modules=(
        "evaluation.datasets.harmbench_eval"
        "evaluation.datasets.advbench_eval"
        "evaluation.datasets.actorattack_eval"
        "evaluation.datasets.gsm8k_eval"
        "evaluation.datasets.mmlu_eval"
        "evaluation.datasets.redqueen_eval"
        "evaluation.datasets.xstest_eval"
    )
    
    for module in "${modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            print_colored $GREEN "  ✓ $module"
        else
            print_colored $RED "  ✗ $module - Missing or has errors"
            return 1
        fi
    done
    
    # 检查CUDA
    print_colored $BLUE "Checking CUDA..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || {
        print_colored $YELLOW "CUDA check failed. vLLM may not work properly."
    }
    
    # 检查必要的库
    python -c "import transformers, datasets" 2>/dev/null || {
        print_colored $RED "Required libraries not found! Please install transformers, datasets"
        return 1
    }
    
    # 检查模型路径
    if [[ ! -d "./vllm_merged_models/defender_merged_model" ]]; then
        print_colored $RED "Merged model not found! Please run merge script first."
        return 1
    fi
    
    # 检查模型配置
    if [[ ! -f "./vllm_merged_models/defender_merged_model/config.json" ]]; then
        print_colored $RED "Model config not found! Please check model integrity."
        return 1
    fi
    
    # 检查数据集文件
    print_colored $BLUE "Checking dataset files..."
    local datasets=("harmbench.json" "advbench.json" "xstest.json")
    local dataset_dirs=("ActorAttack" "RedQueen")
    
    for dataset in "${datasets[@]}"; do
        if [[ -f "./local_datasets/$dataset" ]]; then
            print_colored $GREEN "  ✓ $dataset"
        else
            print_colored $YELLOW "  ○ $dataset - Not found (may cause evaluation failure)"
        fi
    done
    
    for dataset_dir in "${dataset_dirs[@]}"; do
        if [[ -d "./local_datasets/$dataset_dir" ]]; then
            print_colored $GREEN "  ✓ $dataset_dir/"
        else
            print_colored $YELLOW "  ○ $dataset_dir/ - Not found (may cause evaluation failure)"
        fi
    done
    
    # 检查GPU内存
    print_colored $BLUE "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || {
        print_colored $YELLOW "nvidia-smi not available. Cannot check GPU memory."
    }
    
    print_colored $GREEN "Dataset-specific vLLM environment check completed!"
}

# 显示数据集特定的vLLM评估状态
show_status() {
    local output_dir="./evaluation_results/dataset_specific_vllm"
    
    print_colored $BLUE "Dataset-Specific vLLM Evaluation Status:"
    print_colored $BLUE "========================================"
    
    if [[ -d "$output_dir" ]]; then
        # 安全评估数据集
        print_colored $CYAN "Security Evaluation:"
        for dataset in harmbench advbench actorattack redqueen; do
            local result_file="$output_dir/${dataset}_results.json"
            
            if [[ -f "$result_file" ]]; then
                local timestamp=$(stat -c %Y "$result_file" 2>/dev/null || stat -f %m "$result_file" 2>/dev/null)
                local readable_time=$(date -d @$timestamp 2>/dev/null || date -r $timestamp 2>/dev/null)
                
                # 提取关键指标
                local key_metric=$(python -c "
import json
try:
    with open('$result_file', 'r') as f:
        data = json.load(f)
    if 'overall_asr' in data:
        print(f'ASR: {data[\"overall_asr\"]:.1%}')
    elif 'judge_attack_success_rate' in data:
        print(f'ASR: {data[\"judge_attack_success_rate\"]:.1%}')
    else:
        print('Completed')
except:
    print('N/A')
" 2>/dev/null)
                
                print_colored $GREEN "  ✓ $dataset - $key_metric ($readable_time)"
            else
                print_colored $YELLOW "  ○ $dataset - not evaluated"
            fi
        done
        
        # 能力评估数据集
        print_colored $CYAN "Capability Evaluation:"
        for dataset in gsm8k mmlu; do
            local result_file="$output_dir/${dataset}_results.json"
            
            if [[ -f "$result_file" ]]; then
                local timestamp=$(stat -c %Y "$result_file" 2>/dev/null || stat -f %m "$result_file" 2>/dev/null)
                local readable_time=$(date -d @$timestamp 2>/dev/null || date -r $timestamp 2>/dev/null)
                
                # 提取关键指标
                local key_metric=$(python -c "
import json
try:
    with open('$result_file', 'r') as f:
        data = json.load(f)
    if 'accuracy' in data:
        print(f'Acc: {data[\"accuracy\"]:.1%}')
    elif 'overall_accuracy' in data:
        print(f'Acc: {data[\"overall_accuracy\"]:.1%}')
    else:
        print('Completed')
except:
    print('N/A')
" 2>/dev/null)
                
                print_colored $GREEN "  ✓ $dataset - $key_metric ($readable_time)"
            else
                print_colored $YELLOW "  ○ $dataset - not evaluated"
            fi
        done
        
        # 过度拒绝评估数据集
        print_colored $CYAN "Over-refusal Evaluation:"
        for dataset in xstest; do
            local result_file="$output_dir/${dataset}_results.json"
            
            if [[ -f "$result_file" ]]; then
                local timestamp=$(stat -c %Y "$result_file" 2>/dev/null || stat -f %m "$result_file" 2>/dev/null)
                local readable_time=$(date -d @$timestamp 2>/dev/null || date -r $timestamp 2>/dev/null)
                
                # 提取关键指标
                local key_metric=$(python -c "
import json
try:
    with open('$result_file', 'r') as f:
        data = json.load(f)
    if 'metrics' in data and 'overall_over_refusal_rate' in data['metrics']:
        print(f'Over-refusal: {data[\"metrics\"][\"overall_over_refusal_rate\"]:.1%}')
    else:
        print('Completed')
except:
    print('N/A')
" 2>/dev/null)
                
                print_colored $GREEN "  ✓ $dataset - $key_metric ($readable_time)"
            else
                print_colored $YELLOW "  ○ $dataset - not evaluated"
            fi
        done
        
        # 显示汇总报告状态
        if [[ -f "$output_dir/dataset_specific_vllm_summary.json" ]]; then
            print_colored $GREEN "✓ Dataset-specific vLLM Summary report available"
        else
            print_colored $YELLOW "○ Dataset-specific vLLM Summary report not generated"
        fi
        
        # 显示文件统计
        local result_files=$(find "$output_dir" -name "*_results.json" | wc -l)
        
        print_colored $BLUE "\nFile Statistics:"
        print_colored $BLUE "  Result files: $result_files"
        
        # 显示磁盘使用情况
        if command -v du &> /dev/null; then
            local disk_usage=$(du -sh "$output_dir" 2>/dev/null | cut -f1)
            print_colored $BLUE "  Disk usage: $disk_usage"
        fi
    else
        print_colored $YELLOW "No dataset-specific vLLM evaluation results found."
    fi
}

# 清理vLLM结果
clean_results() {
    local output_dir="./evaluation_results/vllm_individual_datasets"
    
    if [[ -d "$output_dir" ]]; then
        print_colored $YELLOW "Cleaning vLLM evaluation results..."
        
        # 显示清理前的统计
        local gen_files=$(find "$output_dir" -name "*_generations.json" | wc -l)
        local result_files=$(find "$output_dir" -name "*_results.json" | wc -l)
        
        print_colored $BLUE "Files to be cleaned:"
        print_colored $BLUE "  Generation files: $gen_files"
        print_colored $BLUE "  Result files: $result_files"
        
        # 询问确认
        read -p "Are you sure you want to clean all vLLM results? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$output_dir"
            print_colored $GREEN "vLLM results cleaned!"
        else
            print_colored $YELLOW "Cleaning cancelled."
        fi
    else
        print_colored $YELLOW "No vLLM results to clean."
    fi
}

# 主逻辑 - 数据集特定vLLM加速版
case ${1:-"help"} in
    "help"|"-h"|"--help")
        show_help
        ;;
    "quick")
        if [[ -n "$2" ]]; then
            quick_eval "$2"
        else
            batch_quick_eval
        fi
        ;;
    "full")
        if [[ -n "$2" ]]; then
            full_eval "$2"
        else
            full_eval "all"
        fi
        ;;
    "check")
        check_environment
        ;;
    "status")
        show_status
        ;;
    "clean")
        clean_results
        ;;
    "perf")
        performance_test "$2" "$3"
        ;;
    "gpu")
        # 显示GPU状态
        print_colored $BLUE "GPU Status:"
        nvidia-smi 2>/dev/null || print_colored $YELLOW "nvidia-smi not available"
        ;;
    "info")
        show_dataset_info "$2"
        ;;
    "modules")
        # 检查评估模块状态
        print_colored $BLUE "Dataset-Specific Evaluation Modules:"
        check_environment | grep -E "(✓|✗)" | grep "evaluation\."
        ;;
    *)
        # 直接传递给评估脚本
        bash scripts/evaluate.sh "$@"
        ;;
esac