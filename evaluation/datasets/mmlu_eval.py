"""
MMLU数据集评估模块 - 修复版本
基于官方评估方法实现：https://github.com/hendrycks/test
5-shot多学科多选题评估
修复了循环引用问题
"""
import json
import logging
import os
import re
import math
import copy
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np

from ..utils.data_loader import BaseDataLoader

logger = logging.getLogger(__name__)

# MMLU官方学科分类
MMLU_SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

MMLU_CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

class MMLUDataLoader(BaseDataLoader):
    """MMLU数据加载器"""
    
    def load_dataset(self, split: str = "test") -> Dict[str, List[Dict[str, Any]]]:
        """
        加载MMLU数据集
        
        Args:
            split: 数据集分割 ("test", "dev", "val")
            
        Returns:
            按学科组织的数据字典
        """
        try:
            # 从HuggingFace加载
            dataset = load_dataset("cais/mmlu", "all", split=split)
            
            # 按学科组织数据
            subjects_data = {}
            for item in dataset:
                subject = item['subject']
                if subject not in subjects_data:
                    subjects_data[subject] = []
                
                # 确保所有数据都是基本类型，避免循环引用
                processed_item = {
                    'question': str(item['question']),
                    'choices': [str(choice) for choice in item['choices']],
                    'answer': int(item['answer']),  # 数字索引 (0-3)
                    'answer_label': str(['A', 'B', 'C', 'D'][item['answer']]),
                    'subject': str(subject),
                    'source': 'mmlu'
                }
                subjects_data[subject].append(processed_item)
            
            logger.info(f"Loaded MMLU {split} data for {len(subjects_data)} subjects")
            return subjects_data
            
        except Exception as e:
            logger.error(f"Failed to load MMLU from HuggingFace: {e}")
            raise RuntimeError(f"Cannot load MMLU dataset: {e}")

class MMLUEvaluator:
    """MMLU评估器"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        max_model_len: int = 4096,
        k_shot: int = 5
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.k_shot = k_shot
        
        # 初始化数据加载器
        self.data_loader = MMLUDataLoader()
        
        # 初始化模型
        self.llm = self._load_model()
        self.tokenizer = self.llm.get_tokenizer()
        
        logger.info(f"MMLUEvaluator initialized with {k_shot}-shot")
    
    def _load_model(self) -> LLM:
        """加载目标模型"""
        return LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype='bfloat16',
            gpu_memory_utilization=0.5,
            max_model_len=self.max_model_len,
            enforce_eager=True,
            disable_custom_all_reduce=True
        )
    
    def _format_subject(self, subject: str) -> str:
        """格式化学科名称"""
        return " ".join(subject.split("_"))
    
    def _format_example(self, question: str, choices: List[str], answer: str = None, include_answer: bool = True) -> str:
        """格式化MMLU示例"""
        choice_labels = ["A", "B", "C", "D"]
        prompt = question
        for j, choice in enumerate(choices):
            prompt += f"\n{choice_labels[j]}. {choice}"
        prompt += "\nAnswer:"
        if include_answer and answer is not None:
            prompt += f" {answer}\n\n"
        return prompt
    
    def _build_prompt(self, dev_examples: List[Dict], test_item: Dict, subject: str) -> str:
        """构建MMLU prompt"""
        prompt = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}.\n\n"
        
        # 添加few-shot示例
        for example in dev_examples:
            prompt += self._format_example(
                example['question'],
                example['choices'],
                example['answer_label'],
                include_answer=True
            )
        
        # 添加测试问题
        prompt += self._format_example(
            test_item['question'],
            test_item['choices'],
            include_answer=False
        )
        
        return prompt
    
    def _check_prompt_length(self, prompt: str) -> bool:
        """检查prompt长度"""
        try:
            tokens = self.tokenizer.encode(prompt)
            return len(tokens) <= self.max_model_len - 10  # 留少量空间给答案
        except:
            return len(prompt.split()) * 1.3 <= self.max_model_len - 10
    
    def _get_stop_token_ids(self) -> List[int]:
        """获取停止token IDs"""
        stop_tokens = []
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.append(self.tokenizer.eos_token_id)
        
        for token in ["<|eot_id|>", "<|end_of_turn|>", "<|im_end|>", "</s>"]:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id and token_id not in stop_tokens:
                    stop_tokens.append(token_id)
            except:
                continue
        
        return stop_tokens
    
    def _extract_choice_probabilities(self, output, choice_labels: List[str] = ["A", "B", "C", "D"]) -> List[float]:
        """从输出中提取选择概率，确保返回基本类型"""
        if not output.outputs[0].logprobs:
            return [0.25, 0.25, 0.25, 0.25]  # 均匀分布，确保是基本类型
        
        logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
        
        choice_logprobs = []
        for label in choice_labels:
            prob = None
            # 尝试不同格式
            for token_format in [label, f" {label}", f"{label})", f" {label})"]:
                if token_format in logprobs:
                    prob = float(logprobs[token_format].logprob)  # 确保转换为基本float类型
                    break
            
            if prob is None:
                prob = float('-inf')
            
            choice_logprobs.append(prob)
        
        # 转换为概率
        choice_logprobs = np.array(choice_logprobs)
        choice_logprobs = choice_logprobs - np.max(choice_logprobs)
        choice_probs = np.exp(choice_logprobs)
        choice_probs = choice_probs / np.sum(choice_probs)
        
        # 确保返回基本的Python float列表
        return [float(p) for p in choice_probs.tolist()]
    
    def _parse_answer(self, generated_text: str, choice_probs: List[float]) -> int:
        """解析生成的答案"""
        generated_text = generated_text.strip().upper()
        choice_labels = ['A', 'B', 'C', 'D']
        
        # 直接匹配
        if generated_text in choice_labels:
            return choice_labels.index(generated_text)
        
        # 查找第一个出现的选择标签
        for i, label in enumerate(choice_labels):
            if label in generated_text:
                return i
        
        # 使用概率最高的选择
        return int(np.argmax(choice_probs))
    
    def _clean_item_for_serialization(self, item: Dict) -> Dict:
        """清理项目数据，确保可以JSON序列化"""
        cleaned_item = {}
        for key, value in item.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                cleaned_item[key] = value
            elif isinstance(value, list):
                # 确保列表中的元素也是基本类型
                cleaned_item[key] = [
                    str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                    for v in value
                ]
            else:
                # 转换为字符串
                cleaned_item[key] = str(value)
        return cleaned_item
    
    def _evaluate_subject(self, subject: str, test_data: List[Dict], dev_data: List[Dict]) -> Dict[str, Any]:
        """评估单个学科"""
        logger.info(f"Evaluating MMLU subject: {subject}")
        
        prompts = []
        items = []
        
        # 限制dev示例数量
        dev_examples = dev_data[:self.k_shot] if len(dev_data) >= self.k_shot else dev_data
        
        for item in test_data:
            # 动态调整few-shot数量
            current_k = len(dev_examples)
            prompt = None
            
            while current_k >= 0:
                current_examples = dev_examples[:current_k]
                temp_prompt = self._build_prompt(current_examples, item, subject)
                
                if self._check_prompt_length(temp_prompt):
                    prompt = temp_prompt
                    break
                else:
                    current_k -= 1
            
            if prompt is None:
                # 最简prompt
                prompt = f"Question: {item['question']}\nChoices:\n"
                for j, choice in enumerate(item['choices']):
                    prompt += f"{['A', 'B', 'C', 'D'][j]}. {choice}\n"
                prompt += "Answer:"
            
            prompts.append(prompt)
            items.append(item)
        
        # 生成回答
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=5,
            stop_token_ids=self._get_stop_token_ids(),
            stop=["\n", "B.", "C.", "D.", "\n\n"],
            skip_special_tokens=True,
            logprobs=10
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 处理结果
        correct_count = 0
        total_count = len(items)
        detailed_results = []
        
        for i, output in enumerate(outputs):
            item = items[i]
            generated_text = str(output.outputs[0].text.strip())  # 确保是字符串
            
            # 获取选择概率
            choice_probs = self._extract_choice_probabilities(output)
            
            # 解析答案
            predicted_choice = self._parse_answer(generated_text, choice_probs)
            predicted_label = ['A', 'B', 'C', 'D'][predicted_choice]
            
            # 检查正确性
            is_correct = (predicted_choice == item['answer'])
            if is_correct:
                correct_count += 1
            
            # 清理并构建结果项，确保所有数据都是可序列化的基本类型
            result_item = {
                'question': str(item['question']),
                'choices': [str(choice) for choice in item['choices']],
                'correct_answer': int(item['answer']),
                'correct_label': str(item['answer_label']),
                'predicted_choice': int(predicted_choice),
                'predicted_label': str(predicted_label),
                'choice_probs': choice_probs,  # 已经是基本float列表
                'generated_text': str(generated_text),
                'is_correct': bool(is_correct)
            }
            
            detailed_results.append(result_item)
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            'subject': str(subject),
            'accuracy': float(accuracy),
            'correct_count': int(correct_count),
            'total_count': int(total_count),
            'detailed_results': detailed_results
        }
    
    def evaluate(
        self,
        subjects: Optional[List[str]] = None,
        max_samples_per_subject: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行MMLU评估
        
        Args:
            subjects: 要评估的学科列表，None表示全部
            max_samples_per_subject: 每个学科最大样本数
            output_path: 结果保存路径
            
        Returns:
            评估结果
        """
        logger.info(f"Starting MMLU evaluation ({self.k_shot}-shot)")
        
        # 加载数据
        test_data_dict = self.data_loader.load_dataset(split="test")
        dev_data_dict = self.data_loader.load_dataset(split="dev")
        
        # 确定要评估的学科
        if subjects is None:
            subjects = list(test_data_dict.keys())
        
        # 评估每个学科
        subject_results = {}
        all_correct = 0
        all_total = 0
        
        for subject in tqdm(subjects, desc="Evaluating MMLU subjects"):
            if subject not in test_data_dict:
                logger.warning(f"Subject {subject} not found in test data")
                continue
            
            test_data = test_data_dict[subject]
            dev_data = dev_data_dict.get(subject, [])
            
            # 限制样本数
            if max_samples_per_subject and len(test_data) > max_samples_per_subject:
                test_data = test_data[:max_samples_per_subject]
            
            # 评估学科
            result = self._evaluate_subject(subject, test_data, dev_data)
            subject_results[subject] = result
            
            all_correct += result['correct_count']
            all_total += result['total_count']
            
            logger.info(f"{subject}: {result['accuracy']:.1%}")
        
        # 计算分类准确率
        category_results = self._calculate_category_accuracies(subject_results)
        
        # 计算校准指标
        calibration_metrics = self._calculate_calibration_metrics(subject_results)
        
        # 整理最终结果 - 确保所有数据都是可序列化的
        overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
        
        final_results = {
            'dataset': 'mmlu',
            'model_path': str(self.model_path),
            'k_shot': int(self.k_shot),
            'total_subjects': int(len(subject_results)),
            'total_samples': int(all_total),
            'total_correct': int(all_correct),
            'overall_accuracy': float(overall_accuracy),
            'subject_accuracies': {k: float(v['accuracy']) for k, v in subject_results.items()},
            'category_accuracies': {k: float(v) for k, v in category_results.items()},
            'calibration_metrics': {k: float(v) for k, v in calibration_metrics.items()},
            'detailed_results': subject_results
        }
        
        # 保存结果前再次清理，确保没有循环引用
        try:
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 创建一个深拷贝并清理所有数据
                safe_results = self._deep_clean_for_json(final_results)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(safe_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # 尝试保存简化版本
            simplified_results = {
                'dataset': final_results['dataset'],
                'model_path': final_results['model_path'],
                'k_shot': final_results['k_shot'],
                'overall_accuracy': final_results['overall_accuracy'],
                'subject_accuracies': final_results['subject_accuracies'],
                'category_accuracies': final_results['category_accuracies']
            }
            if output_path:
                simplified_path = output_path.replace('.json', '_simplified.json')
                with open(simplified_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Simplified results saved to {simplified_path}")
        
        # 打印摘要
        self._print_summary(final_results)
        
        return final_results
    
    def _deep_clean_for_json(self, obj):
        """深度清理对象，确保JSON可序列化"""
        if isinstance(obj, dict):
            return {k: self._deep_clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_clean_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._deep_clean_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # 对于其他类型，转换为字符串
            return str(obj)
    
    def _calculate_category_accuracies(self, subject_results: Dict[str, Dict]) -> Dict[str, float]:
        """计算各类别准确率"""
        category_stats = {cat: {'correct': 0, 'total': 0} for cat in MMLU_CATEGORIES}
        
        for subject, result in subject_results.items():
            subcats = MMLU_SUBCATEGORIES.get(subject, ["other"])
            
            for subcat in subcats:
                for cat_name, cat_subcats in MMLU_CATEGORIES.items():
                    if subcat in cat_subcats:
                        category_stats[cat_name]['correct'] += result['correct_count']
                        category_stats[cat_name]['total'] += result['total_count']
                        break
        
        return {
            cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            for cat, stats in category_stats.items()
        }
    
    def _calculate_calibration_metrics(self, subject_results: Dict[str, Dict]) -> Dict[str, float]:
        """计算校准指标"""
        all_probs = []
        all_correct = []
        
        for result in subject_results.values():
            for item in result['detailed_results']:
                max_prob = max(item['choice_probs'])
                is_correct = item['is_correct']
                all_probs.append(max_prob)
                all_correct.append(is_correct)
        
        if not all_probs:
            return {}
        
        all_probs = np.array(all_probs)
        all_correct = np.array(all_correct)
        
        # 计算ECE
        ece = self._compute_ece(all_probs, all_correct)
        
        return {
            'avg_confidence': float(np.mean(all_probs)),
            'accuracy': float(np.mean(all_correct)),
            'ece': float(ece),
            'confidence_accuracy_gap': float(abs(np.mean(all_probs) - np.mean(all_correct)))
        }
    
    def _compute_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
        """计算Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*80}")
        print(f"MMLU Evaluation Results ({results['k_shot']}-shot)")
        print(f"{'='*80}")
        print(f"Model: {results['model_path']}")
        print(f"Total Subjects: {results['total_subjects']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
        
        print(f"\nCategory Accuracies:")
        for category, accuracy in results['category_accuracies'].items():
            print(f"  {category}: {accuracy:.1%}")
        
        if results['calibration_metrics']:
            cal_metrics = results['calibration_metrics']
            print(f"\nCalibration Metrics:")
            print(f"  Average Confidence: {cal_metrics['avg_confidence']:.3f}")
            print(f"  Expected Calibration Error: {cal_metrics['ece']:.3f}")
            print(f"  Confidence-Accuracy Gap: {cal_metrics['confidence_accuracy_gap']:.3f}")
        
        print(f"\nTop 5 Subject Accuracies:")
        sorted_subjects = sorted(results['subject_accuracies'].items(), key=lambda x: x[1], reverse=True)
        for subject, accuracy in sorted_subjects[:5]:
            print(f"  {subject}: {accuracy:.1%}")
        
        print(f"\nBottom 5 Subject Accuracies:")
        for subject, accuracy in sorted_subjects[-5:]:
            print(f"  {subject}: {accuracy:.1%}")
        
        print(f"{'='*80}\n")

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MMLU Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--subjects", type=str, nargs="*", default=None,
                       help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--k_shot", type=int, default=5,
                       help="Number of few-shot examples")
    parser.add_argument("--max_samples_per_subject", type=int, default=None,
                       help="Maximum samples per subject")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Tensor parallel size for vLLM")
    
    args = parser.parse_args()
    
    # 运行评估
    evaluator = MMLUEvaluator(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        k_shot=args.k_shot
    )
    
    results = evaluator.evaluate(
        subjects=args.subjects,
        max_samples_per_subject=args.max_samples_per_subject,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()