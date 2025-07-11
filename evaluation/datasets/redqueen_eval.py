"""
RedQueen数据集评估模块
基于官方评估方法实现：https://github.com/redqueen-ai/redqueen
多轮对话安全评估
"""
import json
import logging
import os
import copy
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ..judges.safety_judge import SafetyJudge
from ..utils.data_loader import BaseDataLoader
from ..utils.metrics import calculate_multi_turn_asr

logger = logging.getLogger(__name__)

class RedQueenDataLoader(BaseDataLoader):
    """RedQueen数据加载器"""
    
    def load_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        加载RedQueen数据集
        
        Args:
            split: 数据集分割
            
        Returns:
            数据列表
        """
        # 尝试从本地文件加载
        local_files = {
            "test": "./local_datasets/RedQueen/redqueen_test.json",
            "train": "./local_datasets/RedQueen/redqueen_train.json",
            "dev": "./local_datasets/RedQueen/redqueen_dev.json"
        }
        
        if split in local_files and os.path.exists(local_files[split]):
            return self.load_from_local(local_files[split])
        else:
            raise RuntimeError(f"Cannot find RedQueen {split} dataset. Please ensure data is available at {local_files.get(split, 'unknown path')}")
    
    def load_from_local(self, file_path: str) -> List[Dict[str, Any]]:
        """从本地文件加载数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的数据格式
        if isinstance(data, list):
            data_list = data
        elif 'data' in data:
            data_list = data['data']
        elif 'samples' in data:
            data_list = data['samples']
        else:
            data_list = [data]
        
        # 标准化格式
        processed_data = []
        for i, item in enumerate(data_list):
            processed_item = {
                'id': item.get('id', f"redqueen_{i}"),
                'goal': item.get('goal', item.get('target', '')),
                'turns': item.get('turns', item.get('conversation_turns', [])),
                'category': item.get('category', 'multi_turn_attack'),
                'difficulty': item.get('difficulty', 'medium'),
                'attack_type': item.get('attack_type', 'unknown'),
                'source': 'redqueen'
            }
            processed_data.append(processed_item)
        
        logger.info(f"Loaded {len(processed_data)} RedQueen samples from local file")
        return processed_data

class RedQueenEvaluator:
    """RedQueen评估器"""
    
    def __init__(
        self,
        model_path: str,
        judge_model: str = "cais/HarmBench-Llama-2-13b-cls",
        tensor_parallel_size: int = 2,
        max_model_len: int = 4096
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        
        # 初始化数据加载器
        self.data_loader = RedQueenDataLoader()
        
        # 初始化模型
        self.llm = self._load_model()
        self.tokenizer = self.llm.get_tokenizer()
        
        # 初始化judge
        self.judge = SafetyJudge(
            model_name=judge_model,
            tensor_parallel_size=1,
            max_model_len=max_model_len
        )
        
        logger.info("RedQueenEvaluator initialized")
    
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
    
    def _conduct_conversation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        进行多轮对话
        
        Args:
            item: 包含对话轮次的数据项
            
        Returns:
            对话结果
        """
        goal = item['goal']
        turns = item['turns']
        
        # 初始化对话历史
        conversation_history = []
        stop_token_ids = self._get_stop_token_ids()
        
        # 逐轮进行对话
        for turn_idx, turn in enumerate(turns):
            # RedQueen格式可能有不同的结构
            if isinstance(turn, dict):
                user_message = turn.get('user', turn.get('human', turn.get('prompt', '')))
            else:
                user_message = str(turn)
            
            # 添加用户消息
            conversation_history.append({"role": "user", "content": user_message})
            
            # 构建prompt
            try:
                prompt = self.tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")
                # 备用格式
                prompt_parts = []
                for msg in conversation_history:
                    role = msg['role']
                    content = msg['content']
                    if role == 'user':
                        prompt_parts.append(f"Human: {content}")
                    else:
                        prompt_parts.append(f"Assistant: {content}")
                prompt_parts.append("Assistant:")
                prompt = "\n\n".join(prompt_parts)
            
            # 生成回复
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop_token_ids=stop_token_ids,
                stop=["Human:", "### Human:", "\n\nHuman:"],
                skip_special_tokens=True
            )
            
            outputs = self.llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 清理回复
            unwanted_endings = ["]]", "] ]", "[", "}", ")]", ")", "]"]
            for ending in unwanted_endings:
                if response.endswith(ending):
                    response = response[:-len(ending)].strip()
            
            # 添加助手回复
            conversation_history.append({"role": "assistant", "content": response})
        
        return {
            'goal': goal,
            'conversations': conversation_history,
            'dialogue_length': len(conversation_history) // 2,
            'attack_type': item.get('attack_type', 'unknown'),
            'difficulty': item.get('difficulty', 'medium')
        }
    
    def _batch_conduct_conversations(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量进行多轮对话"""
        results = []
        
        for item in tqdm(data_list, desc="Conducting RedQueen conversations"):
            try:
                result = self._conduct_conversation(item)
                # 保留原始元数据
                result.update({
                    'id': item['id'],
                    'category': item['category'],
                    'source': item['source']
                })
                results.append(result)
            except Exception as e:
                logger.error(f"Error in conversation for {item.get('id', 'unknown')}: {e}")
                # 创建空结果
                results.append({
                    'goal': item['goal'],
                    'conversations': [],
                    'dialogue_length': 0,
                    'attack_type': item.get('attack_type', 'unknown'),
                    'difficulty': item.get('difficulty', 'medium'),
                    'id': item['id'],
                    'category': item['category'],
                    'source': item['source'],
                    'error': str(e)
                })
        
        return results
    
    def evaluate(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行RedQueen评估
        
        Args:
            split: 数据集分割
            max_samples: 最大样本数
            output_path: 结果保存路径
            
        Returns:
            评估结果
        """
        logger.info(f"Starting RedQueen evaluation (split: {split})")
        
        # 加载数据
        data_list = self.data_loader.load_dataset(split=split)
        if max_samples:
            data_list = data_list[:max_samples]
        
        # 进行多轮对话
        conversation_results = self._batch_conduct_conversations(data_list)
        
        # 使用judge评估
        logger.info("Evaluating conversations with judge...")
        judged_results = self.judge.evaluate_multi_turn_batch(conversation_results)
        
        # 计算指标
        metrics = calculate_multi_turn_asr(judged_results)
        
        # 按类别和难度分析
        category_analysis = self._analyze_by_category(judged_results)
        difficulty_analysis = self._analyze_by_difficulty(judged_results)
        attack_type_analysis = self._analyze_by_attack_type(judged_results)
        
        # 整理最终结果
        final_results = {
            'dataset': 'redqueen',
            'split': split,
            'model_path': self.model_path,
            'total_samples': len(judged_results),
            'judge_attack_success_rate': metrics['judge_attack_success_rate'],
            'successful_attacks': metrics['successful_attacks'],
            'avg_dialogue_length': metrics['avg_dialogue_length'],
            'category_analysis': category_analysis,
            'difficulty_analysis': difficulty_analysis,
            'attack_type_analysis': attack_type_analysis,
            'detailed_results': judged_results
        }
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        # 打印摘要
        self._print_summary(final_results)
        
        return final_results
    
    def _analyze_by_category(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """按类别分析结果"""
        category_stats = {}
        
        for result in results:
            category = result.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'successful': 0}
            
            category_stats[category]['total'] += 1
            if result.get('attack_success', False):
                category_stats[category]['successful'] += 1
        
        # 计算成功率
        category_analysis = {}
        for category, stats in category_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            category_analysis[category] = {
                'total_samples': stats['total'],
                'successful_attacks': stats['successful'],
                'success_rate': success_rate
            }
        
        return category_analysis
    
    def _analyze_by_difficulty(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """按难度分析结果"""
        difficulty_stats = {}
        
        for result in results:
            difficulty = result.get('difficulty', 'unknown')
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'total': 0, 'successful': 0}
            
            difficulty_stats[difficulty]['total'] += 1
            if result.get('attack_success', False):
                difficulty_stats[difficulty]['successful'] += 1
        
        # 计算成功率
        difficulty_analysis = {}
        for difficulty, stats in difficulty_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            difficulty_analysis[difficulty] = {
                'total_samples': stats['total'],
                'successful_attacks': stats['successful'],
                'success_rate': success_rate
            }
        
        return difficulty_analysis
    
    def _analyze_by_attack_type(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """按攻击类型分析结果"""
        attack_type_stats = {}
        
        for result in results:
            attack_type = result.get('attack_type', 'unknown')
            if attack_type not in attack_type_stats:
                attack_type_stats[attack_type] = {'total': 0, 'successful': 0}
            
            attack_type_stats[attack_type]['total'] += 1
            if result.get('attack_success', False):
                attack_type_stats[attack_type]['successful'] += 1
        
        # 计算成功率
        attack_type_analysis = {}
        for attack_type, stats in attack_type_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            attack_type_analysis[attack_type] = {
                'total_samples': stats['total'],
                'successful_attacks': stats['successful'],
                'success_rate': success_rate
            }
        
        return attack_type_analysis
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"RedQueen Evaluation Results ({results['split']})")
        print(f"{'='*60}")
        print(f"Model: {results['model_path']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Judge Attack Success Rate: {results['judge_attack_success_rate']:.1%}")
        print(f"Successful Attacks: {results['successful_attacks']}")
        print(f"Average Dialogue Length: {results['avg_dialogue_length']:.1f} turns")
        
        # 按类别显示结果
        if results['category_analysis']:
            print(f"\nResults by Category:")
            for category, analysis in results['category_analysis'].items():
                print(f"  {category}: {analysis['success_rate']:.1%} "
                      f"({analysis['successful_attacks']}/{analysis['total_samples']})")
        
        # 按难度显示结果
        if results['difficulty_analysis']:
            print(f"\nResults by Difficulty:")
            for difficulty, analysis in results['difficulty_analysis'].items():
                print(f"  {difficulty}: {analysis['success_rate']:.1%} "
                      f"({analysis['successful_attacks']}/{analysis['total_samples']})")
        
        # 按攻击类型显示结果
        if results['attack_type_analysis']:
            print(f"\nResults by Attack Type:")
            for attack_type, analysis in results['attack_type_analysis'].items():
                print(f"  {attack_type}: {analysis['success_rate']:.1%} "
                      f"({analysis['successful_attacks']}/{analysis['total_samples']})")
        
        print(f"{'='*60}\n")

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RedQueen Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "train", "dev"],
                       help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results")
    parser.add_argument("--judge_model", type=str, default="cais/HarmBench-Llama-2-13b-cls",
                       help="Judge model for evaluation")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Tensor parallel size for vLLM")
    
    args = parser.parse_args()
    
    # 运行评估
    evaluator = RedQueenEvaluator(
        model_path=args.model_path,
        judge_model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    results = evaluator.evaluate(
        split=args.split,
        max_samples=args.max_samples,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()