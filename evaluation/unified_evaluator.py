"""
统一评估管理器
支持所有数据集的统一评估接口
"""
import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib

from evaluation.utils.metrics import MetricsCalculator
from evaluation.utils.data_loader import DatasetRegistry

logger = logging.getLogger(__name__)

# 数据集评估器映射
DATASET_EVALUATORS = {
    'harmbench': 'evaluation.datasets.harmbench_eval.HarmBenchEvaluator',
    'advbench': 'evaluation.datasets.advbench_eval.AdvBenchEvaluator',
    'actorattack': 'evaluation.datasets.actorattack_eval.ActorAttackEvaluator',
    'redqueen': 'evaluation.datasets.redqueen_eval.RedQueenEvaluator',
    'gsm8k': 'evaluation.datasets.gsm8k_eval.GSM8KEvaluator',
    'mmlu': 'evaluation.datasets.mmlu_eval.MMLUEvaluator',
    'xstest': 'evaluation.datasets.xstest_eval.XSTestEvaluator'
}

# 数据集类型分类
DATASET_CATEGORIES = {
    'safety_single_turn': ['harmbench', 'advbench'],
    'safety_multi_turn': ['actorattack', 'redqueen'],
    'capability': ['gsm8k', 'mmlu'],
    'overrefusal': ['xstest']
}

class UnifiedEvaluator:
    """统一评估管理器"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./evaluation_results",
        tensor_parallel_size: int = 2,
        judge_model: str = "cais/HarmBench-Llama-2-13b-cls"
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.judge_model = judge_model
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"UnifiedEvaluator initialized for model: {model_path}")
    
    def _get_evaluator_class(self, dataset_name: str):
        """动态加载评估器类"""
        if dataset_name not in DATASET_EVALUATORS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                           f"Supported datasets: {list(DATASET_EVALUATORS.keys())}")
        
        module_path = DATASET_EVALUATORS[dataset_name]
        module_name, class_name = module_path.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
            evaluator_class = getattr(module, class_name)
            return evaluator_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load evaluator for {dataset_name}: {e}")
    
    def _create_evaluator(self, dataset_name: str, **kwargs):
        """创建数据集评估器实例"""
        evaluator_class = self._get_evaluator_class(dataset_name)
        
        # 合并通用参数
        common_params = {
            'model_path': self.model_path,
            'tensor_parallel_size': self.tensor_parallel_size
        }
        
        # 添加数据集特定参数
        if dataset_name in ['harmbench', 'advbench', 'actorattack', 'redqueen']:
            common_params['judge_model'] = self.judge_model
        
        common_params.update(kwargs)
        
        return evaluator_class(**common_params)
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        **dataset_kwargs
    ) -> Dict[str, Any]:
        """
        评估单个数据集
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大样本数
            save_results: 是否保存结果
            **dataset_kwargs: 数据集特定参数
            
        Returns:
            评估结果字典
        """
        logger.info(f"Starting evaluation on {dataset_name}")
        
        try:
            # 创建评估器
            evaluator = self._create_evaluator(dataset_name, **dataset_kwargs)
            
            # 准备评估参数
            eval_params = {'max_samples': max_samples}
            
            # 添加数据集特定参数
            if dataset_name == 'harmbench':
                eval_params['subset'] = dataset_kwargs.get('subset', 'standard')
            elif dataset_name in ['actorattack', 'redqueen']:
                eval_params['split'] = dataset_kwargs.get('split', 'test')
            elif dataset_name in ['gsm8k', 'mmlu']:
                eval_params['split'] = dataset_kwargs.get('split', 'test')
            elif dataset_name == 'mmlu':
                eval_params['subjects'] = dataset_kwargs.get('subjects', None)
            elif dataset_name == 'gsm8k':
                eval_params['k_shot'] = dataset_kwargs.get('k_shot', 8)
            elif dataset_name == 'xstest':
                eval_params['judge_mode'] = dataset_kwargs.get('judge_mode', 'enhanced_string_matching')
            
            # 设置输出路径
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{dataset_name}_results_{timestamp}.json"
                eval_params['output_path'] = os.path.join(self.output_dir, output_filename)
            
            # 运行评估
            results = evaluator.evaluate(**eval_params)
            
            # 计算综合指标
            if 'detailed_results' in results:
                dataset_category = self._get_dataset_category(dataset_name)
                comprehensive_metrics = self.metrics_calculator.calculate_all_metrics(
                    results['detailed_results'], 
                    dataset_category
                )
                results['comprehensive_metrics'] = comprehensive_metrics
            
            logger.info(f"Evaluation completed for {dataset_name}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            return {
                'dataset': dataset_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def evaluate_multiple_datasets(
        self,
        datasets: List[str],
        max_samples_per_dataset: Optional[int] = None,
        save_individual_results: bool = True,
        save_summary: bool = True,
        **common_kwargs
    ) -> Dict[str, Any]:
        """
        评估多个数据集
        
        Args:
            datasets: 数据集名称列表
            max_samples_per_dataset: 每个数据集的最大样本数
            save_individual_results: 是否保存单独结果
            save_summary: 是否保存汇总结果
            **common_kwargs: 通用参数
            
        Returns:
            汇总评估结果
        """
        logger.info(f"Starting multi-dataset evaluation: {datasets}")
        
        all_results = {}
        summary_metrics = {}
        
        for dataset_name in datasets:
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            # 获取数据集特定参数
            dataset_kwargs = common_kwargs.copy()
            
            try:
                # 运行评估
                result = self.evaluate_dataset(
                    dataset_name=dataset_name,
                    max_samples=max_samples_per_dataset,
                    save_results=save_individual_results,
                    **dataset_kwargs
                )
                
                all_results[dataset_name] = result
                
                # 提取关键指标用于汇总
                if 'error' not in result:
                    summary_metrics[dataset_name] = self._extract_key_metrics(result, dataset_name)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                all_results[dataset_name] = {
                    'dataset': dataset_name,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # 创建汇总结果
        summary_result = {
            'model_path': self.model_path,
            'evaluation_timestamp': datetime.now().isoformat(),
            'datasets_evaluated': datasets,
            'total_datasets': len(datasets),
            'successful_evaluations': len([r for r in all_results.values() if 'error' not in r]),
            'failed_evaluations': len([r for r in all_results.values() if 'error' in r]),
            'summary_metrics': summary_metrics,
            'detailed_results': all_results
        }
        
        # 保存汇总结果
        if save_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"evaluation_summary_{timestamp}.json"
            summary_path = os.path.join(self.output_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Summary results saved to {summary_path}")
        
        # 打印汇总
        self._print_evaluation_summary(summary_result)
        
        return summary_result
    
    def evaluate_by_category(
        self,
        category: str,
        max_samples_per_dataset: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        按类别评估数据集
        
        Args:
            category: 数据集类别 ('safety_single_turn', 'safety_multi_turn', 'capability', 'overrefusal')
            max_samples_per_dataset: 每个数据集的最大样本数
            **kwargs: 其他参数
            
        Returns:
            评估结果
        """
        if category not in DATASET_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. "
                           f"Available categories: {list(DATASET_CATEGORIES.keys())}")
        
        datasets = DATASET_CATEGORIES[category]
        logger.info(f"Evaluating {category} datasets: {datasets}")
        
        return self.evaluate_multiple_datasets(
            datasets=datasets,
            max_samples_per_dataset=max_samples_per_dataset,
            **kwargs
        )
    
    def evaluate_all(
        self,
        max_samples_per_dataset: Optional[int] = None,
        quick_eval: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        评估所有支持的数据集
        
        Args:
            max_samples_per_dataset: 每个数据集的最大样本数
            quick_eval: 是否快速评估（使用更少样本）
            **kwargs: 其他参数
            
        Returns:
            完整评估结果
        """
        if quick_eval and max_samples_per_dataset is None:
            max_samples_per_dataset = 50  # 快速评估使用50个样本
        
        all_datasets = list(DATASET_EVALUATORS.keys())
        logger.info(f"Starting comprehensive evaluation of all datasets: {all_datasets}")
        
        return self.evaluate_multiple_datasets(
            datasets=all_datasets,
            max_samples_per_dataset=max_samples_per_dataset,
            **kwargs
        )
    
    def _get_dataset_category(self, dataset_name: str) -> str:
        """获取数据集类别"""
        for category, datasets in DATASET_CATEGORIES.items():
            if dataset_name in datasets:
                return category
        return 'unknown'
    
    def _extract_key_metrics(self, result: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """提取关键指标用于汇总"""
        key_metrics = {'dataset': dataset_name}
        
        # 根据数据集类型提取不同的关键指标
        if dataset_name in ['harmbench', 'advbench']:
            # 安全性单轮指标
            key_metrics['asr'] = result.get('overall_asr', result.get('asr', 0))
            key_metrics['total_samples'] = result.get('total_samples', 0)
            key_metrics['successful_attacks'] = result.get('successful_attacks', 0)
            
        elif dataset_name in ['actorattack', 'redqueen']:
            # 安全性多轮指标
            key_metrics['judge_attack_success_rate'] = result.get('judge_attack_success_rate', 0)
            key_metrics['avg_dialogue_length'] = result.get('avg_dialogue_length', 0)
            key_metrics['total_samples'] = result.get('total_samples', 0)
            
        elif dataset_name == 'gsm8k':
            # 数学推理指标
            key_metrics['accuracy'] = result.get('accuracy', 0)
            key_metrics['k_shot'] = result.get('k_shot', 0)
            key_metrics['total_samples'] = result.get('total_samples', 0)
            
        elif dataset_name == 'mmlu':
            # 多学科指标
            key_metrics['overall_accuracy'] = result.get('overall_accuracy', 0)
            key_metrics['total_subjects'] = result.get('total_subjects', 0)
            key_metrics['total_samples'] = result.get('total_samples', 0)
            if 'category_accuracies' in result:
                key_metrics['category_accuracies'] = result['category_accuracies']
                
        elif dataset_name == 'xstest':
            # 过度拒绝指标
            metrics = result.get('metrics', {})
            key_metrics['overrefusal_rate'] = metrics.get('overrefusal_rate', 0)
            key_metrics['total_samples'] = metrics.get('total_samples', 0)
        
        return key_metrics
    
    def _print_evaluation_summary(self, summary_result: Dict[str, Any]):
        """打印评估汇总"""
        print(f"\n{'='*80}")
        print(f"UNIFIED EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {summary_result['model_path']}")
        print(f"Timestamp: {summary_result['evaluation_timestamp']}")
        print(f"Total Datasets: {summary_result['total_datasets']}")
        print(f"Successful: {summary_result['successful_evaluations']}")
        print(f"Failed: {summary_result['failed_evaluations']}")
        
        print(f"\nKey Metrics by Dataset:")
        print(f"{'-'*80}")
        
        for dataset_name, metrics in summary_result['summary_metrics'].items():
            print(f"\n{dataset_name.upper()}:")
            
            if dataset_name in ['harmbench', 'advbench']:
                asr = metrics.get('asr', 0)
                samples = metrics.get('total_samples', 0)
                attacks = metrics.get('successful_attacks', 0)
                print(f"  ASR: {asr:.1%} ({attacks}/{samples})")
                
            elif dataset_name in ['actorattack', 'redqueen']:
                success_rate = metrics.get('judge_attack_success_rate', 0)
                avg_length = metrics.get('avg_dialogue_length', 0)
                samples = metrics.get('total_samples', 0)
                print(f"  Multi-turn Success Rate: {success_rate:.1%}")
                print(f"  Avg Dialogue Length: {avg_length:.1f}")
                print(f"  Total Conversations: {samples}")
                
            elif dataset_name == 'gsm8k':
                accuracy = metrics.get('accuracy', 0)
                k_shot = metrics.get('k_shot', 0)
                samples = metrics.get('total_samples', 0)
                print(f"  Accuracy ({k_shot}-shot): {accuracy:.1%}")
                print(f"  Total Questions: {samples}")
                
            elif dataset_name == 'mmlu':
                accuracy = metrics.get('overall_accuracy', 0)
                subjects = metrics.get('total_subjects', 0)
                samples = metrics.get('total_samples', 0)
                print(f"  Overall Accuracy: {accuracy:.1%}")
                print(f"  Subjects: {subjects}")
                print(f"  Total Questions: {samples}")
                
                if 'category_accuracies' in metrics:
                    print(f"  Category Accuracies:")
                    for cat, acc in metrics['category_accuracies'].items():
                        print(f"    {cat}: {acc:.1%}")
                        
            elif dataset_name == 'xstest':
                overrefusal_rate = metrics.get('overrefusal_rate', 0)
                samples = metrics.get('total_samples', 0)
                print(f"  Over-refusal Rate: {overrefusal_rate:.1%}")
                print(f"  Total Samples: {samples}")
        
        print(f"\n{'='*80}\n")

def create_unified_evaluator(
    model_path: str,
    output_dir: str = "./evaluation_results",
    **kwargs
) -> UnifiedEvaluator:
    """创建统一评估器的便捷函数"""
    return UnifiedEvaluator(
        model_path=model_path,
        output_dir=output_dir,
        **kwargs
    )