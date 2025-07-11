# evaluation/run_general_eval_vllm_official.py

"""
通用能力评估模块 - vLLM加速版（基于官方代码优化）
评估模型在MMLU、GSM8K等标准基准上的表现
"""
import argparse
import json
import logging
import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
import re
import math

from vllm import LLM, SamplingParams

# JSON序列化辅助函数
def json_serializable(obj):
    """将numpy类型转换为Python原生类型以支持JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    else:
        return obj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 官方MMLU分类系统
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

class GeneralCapabilityEvaluatorVLLM:
    """通用能力评估器 - vLLM加速版（基于官方代码优化）"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_model_len: int = 4096
    ):
        self.model_path = model_path
        self.device = device
        self.max_model_len = max_model_len
        
        logger.info(f"Loading vLLM model from: {model_path}")
        
        # 初始化vLLM引擎
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype='bfloat16',
            gpu_memory_utilization=0.5,
            max_model_len=max_model_len,
            enforce_eager=True,
            disable_custom_all_reduce=True
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        
        # 获取停止标记
        self.stop_token_ids = self._get_safe_stop_tokens()
        
        logger.info("GeneralCapabilityEvaluatorVLLM initialized successfully")
    
    def _get_safe_stop_tokens(self) -> List[int]:
        """获取安全的停止标记"""
        stop_tokens = []
        
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.append(self.tokenizer.eos_token_id)
        
        common_stop_tokens = [
            "<|eot_id|>", "<|end_of_turn|>", "<|im_end|>", 
            "</s>", "<|endoftext|>"
        ]
        
        for token in common_stop_tokens:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if (token_id is not None and 
                    token_id != self.tokenizer.unk_token_id and 
                    token_id > 0 and 
                    token_id not in stop_tokens):
                    stop_tokens.append(token_id)
            except:
                continue
        
        return stop_tokens
    
    def _check_prompt_length(self, prompt: str) -> bool:
        """检查prompt长度是否在限制内"""
        try:
            tokens = self.tokenizer.encode(prompt)
            return len(tokens) <= self.max_model_len - 100  # 留100个tokens的buffer
        except:
            # 如果tokenizer出错，简单估计
            return len(prompt.split()) * 1.3 <= self.max_model_len - 100
    
    def _format_subject(self, subject: str) -> str:
        """格式化学科名称（官方格式）"""
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()
    
    def _format_mmlu_example(self, question: str, choices: List[str], answer: str = None, include_answer: bool = True) -> str:
        """格式化MMLU示例（官方格式）"""
        choice_labels = ["A", "B", "C", "D"]
        prompt = question
        for j, choice in enumerate(choices):
            prompt += f"\n{choice_labels[j]}. {choice}"
        prompt += "\nAnswer:"
        if include_answer and answer is not None:
            prompt += f" {answer}\n\n"
        return prompt
    
    def _gen_mmlu_prompt(self, train_examples: List[Dict], subject: str, k: int = -1) -> str:
        """生成MMLU prompt（官方格式）"""
        prompt = f"The following are multiple choice questions (with answers) about{self._format_subject(subject)}.\n\n"
        
        if k == -1:
            k = len(train_examples)
        
        for i in range(min(k, len(train_examples))):
            example = train_examples[i]
            prompt += self._format_mmlu_example(
                example['question'], 
                example['choices'], 
                example['answer'], 
                include_answer=True
            )
        
        return prompt
    
    def evaluate_mmlu(
        self,
        subjects: Optional[List[str]] = None,
        max_samples_per_subject: Optional[int] = None,
        k_shot: int = 5,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        在MMLU数据集上评估模型（使用官方格式）
        """
        logger.info("Starting MMLU evaluation with vLLM (official format)")
        
        # 加载MMLU数据集
        try:
            dataset = load_dataset("cais/mmlu", "all")
        except:
            try:
                dataset = load_dataset("hendrycks/test", "all")
            except:
                raise RuntimeError("无法加载MMLU数据集，请检查网络连接或数据集名称")
        
        # 获取所有学科
        if subjects is None:
            subjects = list(set(dataset['test']['subject']))
        
        all_results = {}
        subject_scores = {}
        category_results = {cat: {'correct': 0, 'total': 0} for cat in MMLU_CATEGORIES}
        all_probs = []
        all_labels = []
        
        for subject in tqdm(subjects, desc="Evaluating MMLU subjects"):
            try:
                # 过滤当前学科的数据
                dev_data = dataset['dev'].filter(lambda x: x['subject'] == subject)
                test_data = dataset['test'].filter(lambda x: x['subject'] == subject)
                
                if max_samples_per_subject and len(test_data) > max_samples_per_subject:
                    indices = np.random.choice(len(test_data), max_samples_per_subject, replace=False)
                    test_data = test_data.select(indices)
                
                # 评估当前学科
                subject_result = self._evaluate_mmlu_subject_vllm(
                    dev_data, test_data, subject, k_shot
                )
                all_results[subject] = subject_result
                subject_scores[subject] = subject_result['accuracy']
                
                # 收集概率和标签用于校准分析
                for result in subject_result['detailed_results']:
                    all_probs.append(result['choice_probs'])
                    all_labels.append(result['correct_answer'])
                
                # 更新分类统计
                subcats = MMLU_SUBCATEGORIES.get(subject, ["other"])
                for subcat in subcats:
                    for cat_name, cat_subcats in MMLU_CATEGORIES.items():
                        if subcat in cat_subcats:
                            category_results[cat_name]['correct'] += subject_result['correct_answers']
                            category_results[cat_name]['total'] += subject_result['total_questions']
                            break
                
                logger.info(f"{subject}: {subject_result['accuracy']:.1%}")
                
            except Exception as e:
                logger.warning(f"Error evaluating subject {subject}: {e}")
                continue
        
        # 计算分类准确率
        category_accuracies = {}
        for cat_name, cat_data in category_results.items():
            if cat_data['total'] > 0:
                category_accuracies[cat_name] = float(cat_data['correct'] / cat_data['total'])
            else:
                category_accuracies[cat_name] = 0.0
        
        # 计算校准指标
        calibration_metrics = self._compute_calibration_metrics(all_probs, all_labels)
        
        # 计算整体统计
        overall_stats = {
            'overall_accuracy': float(np.mean(list(subject_scores.values()))),
            'subject_scores': {k: float(v) for k, v in subject_scores.items()},
            'category_accuracies': category_accuracies,
            'calibration_metrics': json_serializable(calibration_metrics),
            'num_subjects': int(len(subject_scores)),
            'total_questions': int(sum(r['total_questions'] for r in all_results.values())),
            'total_correct': int(sum(r['correct_answers'] for r in all_results.values()))
        }
        
        # 保存结果
        if save_results:
            self._save_mmlu_results(all_results, overall_stats, output_dir)
        
        return {
            'type': 'mmlu',
            'results': all_results,
            'statistics': overall_stats
        }
    
    def _evaluate_mmlu_subject_vllm(self, dev_data, test_data, subject_name: str, k_shot: int) -> Dict[str, Any]:
        """使用vLLM评估MMLU单个学科（官方格式）"""
        total_questions = len(test_data)
        choice_labels = ["A", "B", "C", "D"]
        
        # 准备few-shot示例
        dev_examples = [
            {
                'question': item['question'],
                'choices': item['choices'],
                'answer': choice_labels[item['answer']]
            } for item in dev_data
        ]
        
        # 构建所有prompts
        prompts = []
        items = []
        
        for item in test_data:
            # 动态调整few-shot数量以适应长度限制
            current_k = k_shot
            full_prompt = None
            
            while current_k >= 0:
                try:
                    # 生成基础prompt
                    train_prompt = self._gen_mmlu_prompt(dev_examples, subject_name, current_k)
                    test_example = self._format_mmlu_example(
                        item['question'], 
                        item['choices'], 
                        include_answer=False
                    )
                    full_prompt = train_prompt + test_example
                    
                    # 检查长度
                    if self._check_prompt_length(full_prompt):
                        break
                    else:
                        current_k -= 1
                        if current_k < 0:
                            # 如果连0-shot都太长，截断question
                            max_question_len = 1000  # 限制question长度
                            if len(item['question']) > max_question_len:
                                truncated_question = item['question'][:max_question_len] + "..."
                                test_example = self._format_mmlu_example(
                                    truncated_question, 
                                    item['choices'], 
                                    include_answer=False
                                )
                                full_prompt = f"The following are multiple choice questions about{self._format_subject(subject_name)}.\n\n" + test_example
                            break
                except Exception as e:
                    logger.warning(f"Error building prompt for {subject_name}: {e}")
                    current_k -= 1
                    if current_k < 0:
                        break
            
            if full_prompt is None:
                logger.warning(f"Could not create valid prompt for question in {subject_name}")
                continue
                
            prompts.append(full_prompt)
            items.append(item)
        
        if not prompts:
            logger.error(f"No valid prompts generated for {subject_name}")
            return {
                'subject': subject_name,
                'accuracy': 0.0,
                'correct_answers': 0,
                'total_questions': 0,
                'detailed_results': []
            }
        
        # vLLM批量生成
        sampling_params = SamplingParams(
            temperature=0.0,  # 确定性采样
            top_p=1.0,
            max_tokens=5,  # 给稍微多一点空间，但仍然很短
            stop_token_ids=self.stop_token_ids,
            stop=["\n", "B.", "C.", "D.", "\n\n"],  # 添加停止条件
            skip_special_tokens=True,
            logprobs=10  # 获取更多logprobs用于概率分析
        )
        
        logger.info(f"Generating answers for {len(prompts)} {subject_name} questions...")
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 处理结果
        correct_answers = 0
        detailed_results = []
        
        for i, output in enumerate(outputs):
            item = items[i]
            generated_text = output.outputs[0].text.strip()
            
            question = item['question']
            choices = item['choices']
            correct_answer = item['answer']  # 数字索引
            correct_label = choice_labels[correct_answer]
            
            # 获取选择概率
            choice_probs = self._extract_choice_probabilities(output, choice_labels)
            
            # 解析答案
            predicted_choice = self._parse_mmlu_answer_official(generated_text, choice_probs)
            predicted_label = choice_labels[predicted_choice]
            
            # 检查是否正确
            is_correct = (predicted_choice == correct_answer)
            if is_correct:
                correct_answers += 1
            
            detailed_results.append({
                'question': question,
                'choices': choices,
                'correct_answer': int(correct_answer),  # 确保是Python int
                'correct_label': correct_label,
                'predicted_choice': int(predicted_choice),  # 确保是Python int
                'predicted_label': predicted_label,
                'choice_probs': [float(p) for p in choice_probs],  # 确保是Python float
                'generated_text': generated_text,
                'is_correct': bool(is_correct)  # 确保是Python bool
            })
        
        accuracy = correct_answers / len(items) if len(items) > 0 else 0.0
        
        return {
            'subject': subject_name,
            'accuracy': float(accuracy),  # 确保是Python float
            'correct_answers': int(correct_answers),  # 确保是Python int
            'total_questions': int(len(items)),  # 确保是Python int
            'detailed_results': detailed_results
        }
    
    def _extract_choice_probabilities(self, output, choice_labels: List[str]) -> List[float]:
        """从输出中提取选择概率"""
        if not output.outputs[0].logprobs:
            # 如果没有logprobs，返回均匀分布
            return [0.25] * 4
        
        logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
        
        # 获取每个选择的log概率
        choice_logprobs = []
        for label in choice_labels:
            # 尝试不同的token格式
            prob = None
            for token_format in [label, f" {label}", f"{label})", f" {label})"]:
                if token_format in logprobs:
                    prob = logprobs[token_format].logprob
                    break
            
            if prob is None:
                prob = float('-inf')  # 如果找不到，设为极小值
            
            choice_logprobs.append(prob)
        
        # 转换为概率
        choice_logprobs = np.array(choice_logprobs)
        choice_logprobs = choice_logprobs - np.max(choice_logprobs)  # 数值稳定性
        choice_probs = np.exp(choice_logprobs)
        choice_probs = choice_probs / np.sum(choice_probs)  # 归一化
        
        return choice_probs.tolist()
    
    def _parse_mmlu_answer_official(self, generated_text: str, choice_probs: List[float]) -> int:
        """解析MMLU答案（官方方法）"""
        generated_text = generated_text.strip().upper()
        choice_labels = ['A', 'B', 'C', 'D']
        
        # 方法1: 直接匹配选择标签
        if generated_text in choice_labels:
            return choice_labels.index(generated_text)
        
        # 方法2: 查找第一个出现的选择标签
        for i, label in enumerate(choice_labels):
            if label in generated_text:
                return i
        
        # 方法3: 使用概率最高的选择
        return np.argmax(choice_probs)
    
    def evaluate_gsm8k(
        self,
        max_samples: Optional[int] = None,
        k_shot: int = 8,  # GSM8K推荐使用8-shot
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        在GSM8K数学推理数据集上评估模型（优化的官方格式，支持8-shot）
        """
        logger.info("Starting GSM8K evaluation with vLLM (optimized official format, 8-shot)")
        
        # 加载GSM8K数据集
        try:
            dataset = load_dataset("gsm8k", "main")
            train_dataset = dataset['train'] 
            test_dataset = dataset['test']
        except:
            try:
                dataset = load_dataset("openai/gsm8k", "main")
                train_dataset = dataset['train']
                test_dataset = dataset['test']
            except:
                raise RuntimeError("无法加载GSM8K数据集，请检查网络连接或数据集名称")
        
        if max_samples and len(test_dataset) > max_samples:
            indices = np.random.choice(len(test_dataset), max_samples, replace=False)
            test_dataset = test_dataset.select(indices)
        
        # 选择最优的few-shot示例
        train_examples = self._select_best_gsm8k_examples(train_dataset, k_shot)
        
        # 构建所有prompts
        prompts = []
        items = []
        
        for item in test_dataset:
            # 动态调整few-shot数量以适应长度限制
            current_k = k_shot
            full_prompt = None
            
            while current_k >= 0:
                try:
                    # 构建优化的few-shot prompt
                    current_examples = train_examples[:current_k]
                    prompt = self._build_gsm8k_prompt(current_examples, item['question'])
                    
                    # 检查长度
                    if self._check_prompt_length(prompt):
                        full_prompt = prompt
                        break
                    else:
                        current_k -= 1
                except Exception as e:
                    logger.warning(f"Error building GSM8K prompt: {e}")
                    current_k -= 1
            
            if full_prompt is None:
                # 如果连0-shot都太长，使用最简化prompt
                full_prompt = f"Question: {item['question']}\nAnswer:"
            
            prompts.append(full_prompt)
            items.append(item)
        
        # vLLM批量生成 - 使用优化的参数
        sampling_params = SamplingParams(
            temperature=0.0,  # 完全确定性
            top_p=1.0,
            max_tokens=1024,  # 进一步增加max_tokens以容纳更长的推理
            stop_token_ids=self.stop_token_ids,
            stop=["\n\nQuestion:", "\nQuestion:", "Problem:", "\n\n\n"],  # 更严格的停止条件
            skip_special_tokens=True
        )
        
        logger.info(f"Generating answers for {len(prompts)} GSM8K questions with {k_shot}-shot...")
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 处理结果
        correct_answers = 0
        total_questions = len(items)
        detailed_results = []
        
        for i, output in enumerate(outputs):
            item = items[i]
            generated_answer = output.outputs[0].text.strip()
            
            question = item['question']
            correct_answer = self._extract_gsm8k_answer_official(item['answer'])
            
            # 使用改进的答案解析
            predicted_answer = self._parse_gsm8k_answer_improved(generated_answer)
            
            # 检查是否正确
            is_correct = self._is_gsm8k_correct(predicted_answer, correct_answer)
            if is_correct:
                correct_answers += 1
            
            detailed_results.append({
                'question': question,
                'correct_answer': float(correct_answer),
                'predicted_answer': float(predicted_answer) if predicted_answer is not None else None,
                'generated_answer': generated_answer,
                'ground_truth': item['answer'],
                'is_correct': bool(is_correct)
            })
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # 保存结果
        result = {
            'type': 'gsm8k',
            'k_shot': int(k_shot),
            'accuracy': float(accuracy),
            'correct_answers': int(correct_answers),
            'total_questions': int(total_questions),
            'detailed_results': detailed_results
        }
        
        if save_results:
            self._save_gsm8k_results(result, output_dir)
        
        logger.info(f"GSM8K accuracy ({k_shot}-shot): {accuracy:.1%}")
        
        return result
    
    def _build_gsm8k_prompt(self, train_examples: List[Dict], test_question: str) -> str:
        """构建GSM8K的few-shot prompt（优化的官方格式）"""
        prompt = ""
        
        # 添加few-shot示例 - 使用更标准的格式
        for i, example in enumerate(train_examples):
            prompt += f"Question: {example['question']}\n"
            # 确保答案包含完整的推理过程和最终答案
            prompt += f"Answer: {example['answer']}\n"
            if i < len(train_examples) - 1:
                prompt += "\n"
        
        # 添加测试问题
        if train_examples:
            prompt += "\n"
        prompt += f"Question: {test_question}\n"
        prompt += "Answer:"
        
        return prompt
    
    def _select_best_gsm8k_examples(self, train_dataset, k_shot: int = 8) -> List[Dict]:
        """选择最优的GSM8K few-shot示例"""
        selected_examples = []
        
        # 尝试选择不同类型的问题以增加多样性
        for i, item in enumerate(train_dataset):
            if len(selected_examples) >= k_shot:
                break
            
            # 确保答案格式良好
            if "####" in item['answer']:
                selected_examples.append({
                    'question': item['question'],
                    'answer': item['answer']
                })
        
        return selected_examples
    
    def _extract_gsm8k_answer_official(self, answer_text: str) -> float:
        """从GSM8K答案文本中提取数值（官方方法）"""
        # GSM8K官方格式：使用 "#### 数值" 
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
    
    def _parse_gsm8k_answer_improved(self, generated_answer: str) -> Optional[float]:
        """高度优化的GSM8K答案解析函数"""
        if not generated_answer:
            return None
        
        # 清理文本 - 移除常见的货币符号和格式字符
        text = generated_answer.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
        
        # 策略1: 优先查找 #### number 格式 (GSM8K官方标准格式)
        pattern = r'####\s*([+-]?(?:\d+\.?\d*|\.\d+))'
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 策略2: 查找 $\boxed{number} 格式 (LaTeX数学格式)
        boxed_patterns = [
            r'\\boxed\{([+-]?(?:\d+\.?\d*|\.\d+))\}',  # \boxed{123}
            r'\$\\boxed\{([+-]?(?:\d+\.?\d*|\.\d+))\}\$',  # $\boxed{123}$
            r'boxed\{([+-]?(?:\d+\.?\d*|\.\d+))\}',  # 没有反斜杠的情况
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 策略3: 查找明确的答案表述模式
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[：:]?\s*([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:答案是|答案为|答案)\s*[：:]?\s*([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:total|总计|共|sum)\s*[：=]\s*([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:therefore|所以|因此)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:result|结果)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 策略4: 查找最后几行中的数字 - 更精确的处理
        lines = text.strip().split('\n')
        
        # 从最后一行开始，向前查找最多3行
        for i in range(min(3, len(lines))):
            line = lines[-(i+1)].strip()
            if not line:
                continue
            
            # 4a: 检查是否整行就是一个数字（可能是最终答案）
            if re.match(r'^[+-]?(?:\d+\.?\d*|\.\d+)$', line):
                try:
                    return float(line)
                except ValueError:
                    continue
            
            # 4b: 查找以句号或答案标识结尾的数字
            end_patterns = [
                r'([+-]?(?:\d+\.?\d*|\.\d+))\.?\s*$',  # 行末数字
                r'=\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*$',  # 等号后的数字
                r'is\s+([+-]?(?:\d+\.?\d*|\.\d+))\s*$',  # "is" 后的数字
            ]
            
            for pattern in end_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
        
        # 策略5: 查找带有上下文的数字（更智能的识别）
        context_patterns = [
            r'(?:costs?|pays?|makes?|earns?|saves?|spends?)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:needs?|requires?|takes?)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:total|altogether|in total)[^\d]*?([+-]?(?:\d+\.?\d*|\.\d+))',
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # 取最后一个匹配，通常是最终答案
                    return float(matches[-1])
                except ValueError:
                    continue
        
        # 策略6: 最后的备选方案 - 更智能的数字选择
        all_numbers = re.findall(r'([+-]?(?:\d+\.?\d*|\.\d+))', text)
        if all_numbers:
            # 过滤掉明显不合理的数字
            valid_numbers = []
            for num_str in reversed(all_numbers):  # 从后往前检查
                try:
                    num = float(num_str)
                    # GSM8K答案的合理范围检查
                    if -1e8 <= num <= 1e8 and num != 0.0:  # 排除0（通常不是答案）
                        valid_numbers.append(num)
                except ValueError:
                    continue
            
            # 返回最后一个有效数字
            if valid_numbers:
                return valid_numbers[0]  # 第一个有效数字（因为我们reverse了）
        
        return None
    
    def _is_gsm8k_correct(self, predicted: Optional[float], correct: float) -> bool:
        """判断GSM8K答案是否正确（官方方法）"""
        if predicted is None:
            return False
        
        # 使用相对和绝对误差容忍度
        if abs(correct) < 1e-6:
            return abs(predicted) < 1e-6
        else:
            relative_error = abs(predicted - correct) / abs(correct)
            absolute_error = abs(predicted - correct)
            return relative_error < 1e-6 or absolute_error < 1e-6
    
    def _compute_calibration_metrics(self, all_probs: List[List[float]], all_labels: List[int]) -> Dict[str, float]:
        """计算校准指标"""
        if not all_probs or not all_labels:
            return {}
        
        # 转换为numpy数组
        probs_array = np.array(all_probs)
        labels_array = np.array(all_labels)
        
        # 获取最大概率和预测
        max_probs = np.max(probs_array, axis=1)
        predictions = np.argmax(probs_array, axis=1)
        
        # 计算准确率
        accuracy = float(np.mean(predictions == labels_array))
        
        # 计算平均置信度
        avg_confidence = float(np.mean(max_probs))
        
        # 计算Expected Calibration Error (ECE)
        ece = float(self._compute_ece(max_probs, predictions == labels_array))
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'ece': ece,
            'confidence_accuracy_gap': float(abs(avg_confidence - accuracy))
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
        
        return float(ece)
    
    def _save_mmlu_results(
        self,
        results: Dict[str, Any],
        stats: Dict[str, Any],
        output_dir: Optional[str] = None
    ):
        """保存MMLU结果"""
        if output_dir is None:
            output_dir = "./general_evaluation_results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        with open(os.path.join(output_dir, "mmlu_detailed_results.json"), 'w') as f:
            json.dump(json_serializable(results), f, indent=2, ensure_ascii=False)
        
        # 保存统计摘要
        with open(os.path.join(output_dir, "mmlu_statistics.json"), 'w') as f:
            json.dump(json_serializable(stats), f, indent=2, ensure_ascii=False)
        
        logger.info(f"MMLU results saved to {output_dir}")
    
    def _save_gsm8k_results(self, result: Dict[str, Any], output_dir: Optional[str] = None):
        """保存GSM8K结果"""
        if output_dir is None:
            output_dir = "./general_evaluation_results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "gsm8k_results.json"), 'w') as f:
            json.dump(json_serializable(result), f, indent=2, ensure_ascii=False)
        
        logger.info(f"GSM8K results saved to {output_dir}")

def run_general_evaluation_vllm_official(
    model_path: str,
    benchmarks: List[str],
    max_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    tensor_parallel_size: int = 4,
    max_model_len: int = 4096,  # 增加默认长度
    k_shot_mmlu: int = 5,
    k_shot_gsm8k: int = 8,  # GSM8K推荐8-shot
    seed: int = 42  # 添加随机种子以确保可重复性
) -> Dict[str, Any]:
    """
    运行通用能力评估的便捷函数 - vLLM版本（官方格式）
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 初始化评估器
    evaluator = GeneralCapabilityEvaluatorVLLM(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len
    )
    
    all_results = {}
    
    # 评估每个基准
    for benchmark in benchmarks:
        logger.info(f"Starting evaluation on {benchmark}")
        
        try:
            if benchmark.lower() == 'mmlu':
                results = evaluator.evaluate_mmlu(
                    max_samples_per_subject=max_samples,
                    k_shot=k_shot_mmlu,
                    output_dir=output_dir
                )
            elif benchmark.lower() == 'gsm8k':
                results = evaluator.evaluate_gsm8k(
                    max_samples=max_samples,
                    k_shot=k_shot_gsm8k,  # 使用8-shot
                    output_dir=output_dir
                )
            else:
                logger.warning(f"Unknown benchmark: {benchmark}")
                continue
            
            all_results[benchmark] = results
            
        except Exception as e:
            logger.error(f"Failed to evaluate {benchmark}: {e}")
            all_results[benchmark] = {'error': str(e)}
    
    # 保存汇总结果
    if output_dir:
        summary_file = os.path.join(output_dir, "general_evaluation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            # 提取关键统计信息
            summary = {}
            for benchmark, results in all_results.items():
                if 'error' not in results:
                    if benchmark == 'mmlu':
                        summary[benchmark] = results['statistics']
                    elif benchmark == 'gsm8k':
                        summary[benchmark] = {
                            'accuracy': results['accuracy'],
                            'total_questions': results['total_questions'],
                            'k_shot': results.get('k_shot', 8)
                        }
                else:
                    summary[benchmark] = results
            json.dump(json_serializable(summary), f, indent=2, ensure_ascii=False)
    
    return all_results

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="Run general capability evaluation using vLLM (official format)")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the merged model")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       default=["mmlu", "gsm8k"],
                       help="Benchmarks to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per benchmark")
    parser.add_argument("--output_dir", type=str, default="./general_eval_results_official",
                       help="Output directory for results")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model sequence length")
    parser.add_argument("--k_shot_mmlu", type=int, default=5,
                       help="Number of few-shot examples for MMLU")
    parser.add_argument("--k_shot_gsm8k", type=int, default=8,
                       help="Number of few-shot examples for GSM8K")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 运行评估
    results = run_general_evaluation_vllm_official(
        model_path=args.model_path,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        k_shot_mmlu=args.k_shot_mmlu,
        k_shot_gsm8k=args.k_shot_gsm8k,
        seed=args.seed
    )
    
    # 打印汇总
    print("\n" + "="*80)
    print("GENERAL CAPABILITY EVALUATION SUMMARY (Official Format)")
    print("="*80)
    
    for benchmark, result in results.items():
        if 'error' not in result:
            if benchmark == 'mmlu':
                stats = result['statistics']
                print(f"{benchmark.upper()}:")
                print(f"  Overall Accuracy: {stats['overall_accuracy']:.1%}")
                print(f"  Subjects: {stats['num_subjects']}")
                
                # 打印分类准确率
                for category, accuracy in stats['category_accuracies'].items():
                    print(f"  {category}: {accuracy:.1%}")
                
                # 打印校准指标
                if 'calibration_metrics' in stats:
                    cal_metrics = stats['calibration_metrics']
                    print(f"  ECE: {cal_metrics.get('ece', 0):.3f}")
                    print(f"  Avg Confidence: {cal_metrics.get('avg_confidence', 0):.3f}")
                
            elif benchmark == 'gsm8k':
                k_shot_info = f" ({result.get('k_shot', 'unknown')}-shot)" if 'k_shot' in result else ""
                print(f"{benchmark.upper()}{k_shot_info}:")
                print(f"  Accuracy: {result['accuracy']:.1%}")
                print(f"  Questions: {result['total_questions']}")
        else:
            print(f"{benchmark.upper()}: ERROR - {result['error']}")
    
    print("="*80)

if __name__ == "__main__":
    main()