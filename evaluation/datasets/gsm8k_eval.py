"""
GSM8K数据集评估模块
基于官方评估方法实现：https://github.com/openai/grade-school-math
8-shot数学推理评估
"""
import json
import logging
import os
import re
import math
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np

from ..utils.data_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class GSM8KDataLoader(BaseDataLoader):
    """GSM8K数据加载器"""
    
    def load_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        加载GSM8K数据集
        
        Args:
            split: 数据集分割 ("train" or "test")
            
        Returns:
            数据列表
        """
        try:
            # 从HuggingFace加载
            dataset = load_dataset("gsm8k", "main", split=split)
            
            data_list = []
            for i, item in enumerate(dataset):
                processed_item = {
                    'id': f"gsm8k_{split}_{i}",
                    'question': item['question'],
                    'answer': item['answer'],
                    'ground_truth_value': self._extract_answer_value(item['answer']),
                    'source': 'gsm8k'
                }
                data_list.append(processed_item)
                
            logger.info(f"Loaded {len(data_list)} GSM8K {split} samples")
            return data_list
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K from HuggingFace: {e}")
            raise RuntimeError(f"Cannot load GSM8K dataset: {e}")
    
    def _extract_answer_value(self, answer_text: str) -> float:
        """从GSM8K答案中提取数值（官方方法）"""
        # GSM8K使用 "#### 数值" 格式标记最终答案
        pattern = r'####\s*([+-]?[0-9,]+(?:\.[0-9]+)?)'
        match = re.search(pattern, answer_text)
        if match:
            number_str = match.group(1).replace(',', '')
            try:
                return float(number_str)
            except ValueError:
                pass
        
        # 如果找不到####格式，尝试提取最后一个数字
        numbers = re.findall(r'[+-]?[0-9,]+(?:\.[0-9]+)?', answer_text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                pass
        
        return 0.0

class GSM8KEvaluator:
    """GSM8K评估器"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        max_model_len: int = 4096,
        k_shot: int = 8
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.k_shot = k_shot
        
        # 初始化数据加载器
        self.data_loader = GSM8KDataLoader()
        
        # 初始化模型
        self.llm = self._load_model()
        self.tokenizer = self.llm.get_tokenizer()
        
        # 加载few-shot示例
        self.few_shot_examples = self._load_few_shot_examples()
        
        logger.info(f"GSM8KEvaluator initialized with {k_shot}-shot")
    
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
    
    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        """加载few-shot示例"""
        train_data = self.data_loader.load_dataset(split="train")
        
        # 选择多样化的示例
        selected_examples = []
        for i, item in enumerate(train_data):
            if len(selected_examples) >= self.k_shot:
                break
            
            # 确保答案格式良好
            if "####" in item['answer']:
                selected_examples.append({
                    'question': item['question'],
                    'answer': item['answer']
                })
        
        return selected_examples
    
    def _build_prompt(self, test_question: str) -> str:
        """构建GSM8K的few-shot prompt"""
        prompt = ""
        
        # 添加few-shot示例
        for i, example in enumerate(self.few_shot_examples):
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n"
            if i < len(self.few_shot_examples) - 1:
                prompt += "\n"
        
        # 添加测试问题
        if self.few_shot_examples:
            prompt += "\n"
        prompt += f"Question: {test_question}\n"
        prompt += "Answer:"
        
        return prompt
    
    def _check_prompt_length(self, prompt: str) -> bool:
        """检查prompt长度是否超限"""
        try:
            tokens = self.tokenizer.encode(prompt)
            return len(tokens) <= self.max_model_len - 1024  # 留1024个tokens给回答
        except:
            # 简单估计
            return len(prompt.split()) * 1.3 <= self.max_model_len - 1024
    
    def _generate_responses(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量生成模型回复"""
        prompts = []
        valid_items = []
        
        # 构建prompts
        for item in data_list:
            current_k = self.k_shot
            prompt = None
            
            # 动态调整few-shot数量以适应长度限制
            while current_k >= 0:
                current_examples = self.few_shot_examples[:current_k]
                temp_prompt = self._build_prompt_with_examples(item['question'], current_examples)
                
                if self._check_prompt_length(temp_prompt):
                    prompt = temp_prompt
                    break
                else:
                    current_k -= 1
            
            if prompt is None:
                # 最简prompt
                prompt = f"Question: {item['question']}\nAnswer:"
            
            prompts.append(prompt)
            valid_items.append(item)
        
        # vLLM生成参数
        sampling_params = SamplingParams(
            temperature=0.0,  # 确定性生成用于数学
            top_p=1.0,
            max_tokens=1024,  # 数学解答可能较长
            stop_token_ids=self._get_stop_token_ids(),
            stop=["\n\nQuestion:", "\nQuestion:", "Problem:", "\n\n\n"],
            skip_special_tokens=True
        )
        
        logger.info(f"Generating answers for {len(prompts)} GSM8K questions...")
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 处理结果
        results = []
        for i, output in enumerate(outputs):
            result = valid_items[i].copy()
            result['prompt'] = prompts[i]
            result['generated_answer'] = output.outputs[0].text.strip()
            result['predicted_value'] = self._extract_predicted_value(result['generated_answer'])
            results.append(result)
        
        return results
    
    def _build_prompt_with_examples(self, question: str, examples: List[Dict[str, str]]) -> str:
        """使用指定示例构建prompt"""
        prompt = ""
        
        for i, example in enumerate(examples):
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n"
            if i < len(examples) - 1:
                prompt += "\n"
        
        if examples:
            prompt += "\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        
        return prompt
    
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
    
    def _extract_predicted_value(self, generated_answer: str) -> Optional[float]:
        """从生成的答案中提取预测值（增强版）"""
        if not generated_answer:
            return None
        
        # 策略1: 查找 #### number 格式
        pattern = r'####\s*([+-]?(?:\d+\.?\d*|\.\d+))'
        match = re.search(pattern, generated_answer)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 策略2: 查找 boxed{number} 格式
        boxed_patterns = [
            r'\\boxed\{([+-]?(?:\d+\.?\d*|\.\d+))\}',
            r'\$\\boxed\{([+-]?(?:\d+\.?\d*|\.\d+))\}\$',
            r'boxed\{([+-]?(?:\d+\.?\d*|\.\d+))\}'
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, generated_answer)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 策略3: 查找答案关键词
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[：:]?\s*([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:therefore|so)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:total|sum)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, generated_answer, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 策略4: 查找最后几行的数字
        lines = generated_answer.strip().split('\n')
        for i in range(min(3, len(lines))):
            line = lines[-(i+1)].strip()
            if not line:
                continue
            
            # 检查整行是否为数字
            if re.match(r'^[+-]?(?:\d+\.?\d*|\.\d+)$', line):
                try:
                    return float(line)
                except ValueError:
                    continue
            
            # 查找行末数字
            end_match = re.search(r'([+-]?(?:\d+\.?\d*|\.\d+))\s*$', line)
            if end_match:
                try:
                    return float(end_match.group(1))
                except ValueError:
                    continue
        
        # 策略5: 最后备选 - 查找所有数字并选择合理的
        all_numbers = re.findall(r'([+-]?(?:\d+\.?\d*|\.\d+))', generated_answer)
        if all_numbers:
            try:
                # 从后往前找第一个合理的数字
                for num_str in reversed(all_numbers):
                    num = float(num_str)
                    # GSM8K答案通常在合理范围内
                    if -1e8 <= num <= 1e8 and num != 0.0:
                        return num
            except ValueError:
                pass
        
        return None
    
    def _is_correct(self, predicted: Optional[float], correct: float) -> bool:
        """判断答案是否正确（官方方法）"""
        if predicted is None:
            return False
        
        # 使用相对和绝对误差容忍度
        if abs(correct) < 1e-6:
            return abs(predicted) < 1e-6
        else:
            relative_error = abs(predicted - correct) / abs(correct)
            absolute_error = abs(predicted - correct)
            return relative_error < 1e-6 or absolute_error < 1e-6
    
    def evaluate(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行GSM8K评估
        
        Args:
            split: 数据集分割
            max_samples: 最大样本数
            output_path: 结果保存路径
            
        Returns:
            评估结果
        """
        logger.info(f"Starting GSM8K evaluation ({self.k_shot}-shot)")
        
        # 加载数据
        data_list = self.data_loader.load_dataset(split=split)
        if max_samples:
            data_list = data_list[:max_samples]
        
        # 生成回答
        results = self._generate_responses(data_list)
        
        # 评估准确率
        correct_count = 0
        total_count = len(results)
        
        for result in results:
            is_correct = self._is_correct(
                result['predicted_value'],
                result['ground_truth_value']
            )
            result['is_correct'] = is_correct
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # 分析错误模式
        error_analysis = self._analyze_errors(results)
        
        # 整理最终结果
        final_results = {
            'dataset': 'gsm8k',
            'split': split,
            'model_path': self.model_path,
            'k_shot': self.k_shot,
            'total_samples': total_count,
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'error_analysis': error_analysis,
            'detailed_results': results
        }
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False, default=self._json_serializable)
            logger.info(f"Results saved to {output_path}")
        
        # 打印摘要
        self._print_summary(final_results)
        
        return final_results
    
    def _analyze_errors(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析错误模式"""
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['is_correct'])
        
        # 分析无法解析答案的情况
        no_prediction = sum(1 for r in results if r['predicted_value'] is None)
        
        # 分析数值错误的情况
        numerical_errors = []
        for result in results:
            if not result['is_correct'] and result['predicted_value'] is not None:
                error = abs(result['predicted_value'] - result['ground_truth_value'])
                numerical_errors.append(error)
        
        return {
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'accuracy': correct_samples / total_samples if total_samples > 0 else 0,
            'no_prediction_count': no_prediction,
            'no_prediction_rate': no_prediction / total_samples if total_samples > 0 else 0,
            'numerical_error_count': len(numerical_errors),
            'avg_numerical_error': sum(numerical_errors) / len(numerical_errors) if numerical_errors else 0,
            'max_numerical_error': max(numerical_errors) if numerical_errors else 0
        }
    
    def _json_serializable(self, obj):
        """JSON序列化处理"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"GSM8K Evaluation Results ({results['k_shot']}-shot)")
        print(f"{'='*60}")
        print(f"Model: {results['model_path']}")
        print(f"Split: {results['split']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.1%}")
        
        error_analysis = results['error_analysis']
        print(f"\nError Analysis:")
        print(f"  No Prediction: {error_analysis['no_prediction_count']} ({error_analysis['no_prediction_rate']:.1%})")
        print(f"  Numerical Errors: {error_analysis['numerical_error_count']}")
        if error_analysis['avg_numerical_error'] > 0:
            print(f"  Avg Numerical Error: {error_analysis['avg_numerical_error']:.2f}")
            print(f"  Max Numerical Error: {error_analysis['max_numerical_error']:.2f}")
        
        print(f"{'='*60}\n")

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GSM8K Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "train"],
                       help="Dataset split to evaluate")
    parser.add_argument("--k_shot", type=int, default=8,
                       help="Number of few-shot examples")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Tensor parallel size for vLLM")
    
    args = parser.parse_args()
    
    # 运行评估
    evaluator = GSM8KEvaluator(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        k_shot=args.k_shot
    )
    
    results = evaluator.evaluate(
        split=args.split,
        max_samples=args.max_samples,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()