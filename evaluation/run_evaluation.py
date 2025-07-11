#!/usr/bin/env python3
"""
统一评估命令行接口
支持所有数据集的命令行评估
"""
import multiprocessing
import argparse
import logging
import os
import sys
from typing import List, Optional

# 设置多进程启动方法 - 必须在导入其他模块之前
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.unified_evaluator import UnifiedEvaluator, DATASET_EVALUATORS, DATASET_CATEGORIES

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Unified Model Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 评估单个数据集
  python run_evaluation.py --model_path /path/to/model --datasets harmbench --max_samples 100

  # 评估多个数据集
  python run_evaluation.py --model_path /path/to/model --datasets harmbench advbench gsm8k

  # 按类别评估
  python run_evaluation.py --model_path /path/to/model --category safety_single_turn

  # 快速评估所有数据集
  python run_evaluation.py --model_path /path/to/model --all --quick

  # 完整评估所有数据集
  python run_evaluation.py --model_path /path/to/model --all --max_samples 1000

Available datasets: {datasets}
Available categories: {categories}
        """.format(
            datasets=", ".join(DATASET_EVALUATORS.keys()),
            categories=", ".join(DATASET_CATEGORIES.keys())
        )
    )
    
    # 基本参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    
    # 数据集选择
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--datasets", type=str, nargs="+",
                      choices=list(DATASET_EVALUATORS.keys()),
                      help="Specific datasets to evaluate")
    group.add_argument("--category", type=str,
                      choices=list(DATASET_CATEGORIES.keys()),
                      help="Evaluate all datasets in a category")
    group.add_argument("--all", action="store_true",
                      help="Evaluate all supported datasets")
    
    # 评估参数
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with fewer samples (50 per dataset)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,  # 改为1
                       help="Tensor parallel size for vLLM")
    parser.add_argument("--judge_model", type=str, default="cais/HarmBench-Llama-2-13b-cls",
                       help="Judge model for safety evaluation")
    
    # 数据集特定参数
    parser.add_argument("--harmbench_subset", type=str, default="standard",
                       choices=["standard", "contextual", "copyright"],
                       help="HarmBench subset to evaluate")
    parser.add_argument("--mmlu_subjects", type=str, nargs="*", default=None,
                       help="Specific MMLU subjects to evaluate")
    parser.add_argument("--gsm8k_k_shot", type=int, default=8,
                       help="Number of few-shot examples for GSM8K")
    parser.add_argument("--xstest_judge_mode", type=str, default="enhanced_string_matching",
                       choices=["string_matching", "enhanced_string_matching"],
                       help="XSTest judge mode")
    parser.add_argument("--actorattack_split", type=str, default="test",
                       choices=["test", "train"],
                       help="ActorAttack split to evaluate")
    parser.add_argument("--redqueen_split", type=str, default="test",
                       choices=["test", "train", "dev"],
                       help="RedQueen split to evaluate")
    
    # 输出控制
    parser.add_argument("--save_individual", action="store_true", default=True,
                       help="Save individual dataset results")
    parser.add_argument("--save_summary", action="store_true", default=True,
                       help="Save evaluation summary")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def setup_logging(verbose: bool):
    """设置日志级别"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

def validate_model_path(model_path: str):
    """验证模型路径"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    logger.info(f"Model path validated: {model_path}")

def prepare_dataset_kwargs(args) -> dict:
    """准备数据集特定参数"""
    dataset_kwargs = {}
    
    # HarmBench参数
    if args.harmbench_subset != "standard":
        dataset_kwargs['subset'] = args.harmbench_subset
    
    # MMLU参数
    if args.mmlu_subjects:
        dataset_kwargs['subjects'] = args.mmlu_subjects
    
    # GSM8K参数
    if args.gsm8k_k_shot != 8:
        dataset_kwargs['k_shot'] = args.gsm8k_k_shot
    
    # XSTest参数
    if args.xstest_judge_mode != "enhanced_string_matching":
        dataset_kwargs['judge_mode'] = args.xstest_judge_mode
    
    # 多轮对话数据集参数
    if args.actorattack_split != "test":
        dataset_kwargs['split'] = args.actorattack_split
    if args.redqueen_split != "test":
        dataset_kwargs['split'] = args.redqueen_split
    
    return dataset_kwargs

def run_evaluation(args):
    """运行评估"""
    logger.info("Starting unified model evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # 创建统一评估器
    evaluator = UnifiedEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        judge_model=args.judge_model
    )
    
    # 准备参数
    max_samples = args.max_samples
    if args.quick and max_samples is None:
        max_samples = 50
    
    dataset_kwargs = prepare_dataset_kwargs(args)
    
    # 根据参数选择评估模式
    try:
        if args.datasets:
            # 评估指定数据集
            logger.info(f"Evaluating specific datasets: {args.datasets}")
            results = evaluator.evaluate_multiple_datasets(
                datasets=args.datasets,
                max_samples_per_dataset=max_samples,
                save_individual_results=args.save_individual,
                save_summary=args.save_summary,
                **dataset_kwargs
            )
            
        elif args.category:
            # 按类别评估
            logger.info(f"Evaluating category: {args.category}")
            results = evaluator.evaluate_by_category(
                category=args.category,
                max_samples_per_dataset=max_samples,
                save_individual_results=args.save_individual,
                save_summary=args.save_summary,
                **dataset_kwargs
            )
            
        elif args.all:
            # 评估所有数据集
            logger.info("Evaluating all datasets")
            results = evaluator.evaluate_all(
                max_samples_per_dataset=max_samples,
                quick_eval=args.quick,
                save_individual_results=args.save_individual,
                save_summary=args.save_summary,
                **dataset_kwargs
            )
        
        else:
            raise ValueError("Must specify --datasets, --category, or --all")
        
        # 打印最终摘要
        print_final_summary(results)
        
        logger.info("Evaluation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def print_final_summary(results: dict):
    """打印最终摘要"""
    print(f"\n{'='*100}")
    print(f"EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*100}")
    
    summary_metrics = results.get('summary_metrics', {})
    
    # 按类别汇总
    category_summary = {}
    for dataset_name, metrics in summary_metrics.items():
        category = get_dataset_category(dataset_name)
        if category not in category_summary:
            category_summary[category] = []
        category_summary[category].append((dataset_name, metrics))
    
    # 打印各类别结果
    for category, dataset_list in category_summary.items():
        print(f"\n{category.upper().replace('_', ' ')} RESULTS:")
        print(f"{'-'*50}")
        
        for dataset_name, metrics in dataset_list:
            print(f"\n  {dataset_name}:")
            
            if category == 'safety_single_turn':
                asr = metrics.get('asr', 0)
                print(f"    Attack Success Rate: {asr:.1%}")
                
            elif category == 'safety_multi_turn':
                success_rate = metrics.get('judge_attack_success_rate', 0)
                print(f"    Multi-turn Success Rate: {success_rate:.1%}")
                
            elif category == 'capability':
                if 'accuracy' in metrics:
                    accuracy = metrics['accuracy']
                    print(f"    Accuracy: {accuracy:.1%}")
                elif 'overall_accuracy' in metrics:
                    accuracy = metrics['overall_accuracy']
                    print(f"    Overall Accuracy: {accuracy:.1%}")
                    
            elif category == 'overrefusal':
                overrefusal_rate = metrics.get('overrefusal_rate', 0)
                print(f"    Over-refusal Rate: {overrefusal_rate:.1%}")
    
    # 整体统计
    total_datasets = results.get('total_datasets', 0)
    successful = results.get('successful_evaluations', 0)
    failed = results.get('failed_evaluations', 0)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"{'-'*50}")
    print(f"Total Datasets Evaluated: {total_datasets}")
    print(f"Successful Evaluations: {successful}")
    print(f"Failed Evaluations: {failed}")
    print(f"Success Rate: {successful/total_datasets*100:.1f}%" if total_datasets > 0 else "Success Rate: N/A")
    
    print(f"\nResults saved to: {results.get('output_dir', 'Unknown')}")
    print(f"{'='*100}\n")

def get_dataset_category(dataset_name: str) -> str:
    """获取数据集类别"""
    for category, datasets in DATASET_CATEGORIES.items():
        if dataset_name in datasets:
            return category
    return 'unknown'

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 设置日志
        setup_logging(args.verbose)
        
        # 验证参数
        validate_model_path(args.model_path)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 运行评估
        results = run_evaluation(args)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)