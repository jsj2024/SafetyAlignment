"""
主评估脚本
整合所有HGA评估模块，提供统一的评估接口
"""
import argparse
import gc
import json
import logging
import numpy as np
import os
import torch
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import set_seed

from api import EvalInstance
from models.hga_llm import HGA_LLM, HGATokenizer
from models.utility_model import UtilityModel
from engine.dialogue_game import DialogueGame
from judge import create_judge
from utils import (
    load_hga_model_and_tokenizer,
    generate_with_hga,
    multi_turn_generate_with_hga,
    calculate_utility_scores,
    handle_non_serializable
)
from overrefusal_eval import OverrefusalEvaluator, overrefusal_judge, overrefusal_analysis
from multi_jailbreak_eval import MultiJailbreakEvaluator, MultiEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HGAEvaluator:
    """HGA统一评估器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 加载模型和相关组件
        self.model, self.tokenizer = load_hga_model_and_tokenizer(
            model_path, config_path, device
        )
        self.utility_model = UtilityModel()
        
        # 初始化对话博弈
        self.dialogue_game = DialogueGame(
            hga_model=self.model,
            tokenizer=self.tokenizer,
            utility_model=self.utility_model,
            max_turns=self.config.get('max_turns', 5),
            mcts_simulations=self.config.get('mcts_simulations', 10)
        )
        
        logger.info("HGAEvaluator initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'max_turns': 5,
                'mcts_simulations': 10,
                'max_new_tokens': 512,
                'temperature': 0.7,
                'batch_size': 8
            }
    
    def load_dataset(self, benchmark_path: str, start: int = 0, limit: Optional[int] = None) -> List[EvalInstance]:
        """加载数据集"""
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
        
        instances = []
        
        # 处理不同的数据格式
        if "data" in benchmark:
            data_list = benchmark["data"]
        elif "samples" in benchmark:
            data_list = benchmark["samples"]
        else:
            data_list = benchmark
        
        for d in data_list:
            instance = EvalInstance(**d)
            
            # 处理上下文
            if instance.context is not None and hasattr(instance, 'messages') and instance.messages:
                # 将上下文添加到最后一条用户消息
                last_user_msg = None
                for i, msg in enumerate(instance.messages):
                    if msg.get('role') == 'user':
                        last_user_msg = i
                
                if last_user_msg is not None:
                    original_content = instance.messages[last_user_msg]["content"]
                    instance.messages[last_user_msg]["content"] = f"{instance.context}\n\n---\n\n{original_content}"
            
            instances.append(instance)
        
        # 应用范围限制
        if limit is not None:
            instances = instances[start:start + limit]
        else:
            instances = instances[start:]
        
        logger.info(f"Loaded {len(instances)} instances from {benchmark_path}")
        return instances
    
    def evaluate_standard(
        self,
        instances: List[EvalInstance],
        role: str = "defender",
        use_hga_features: bool = True,
        gen_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[EvalInstance]:
        """标准评估（单轮或多轮对话）"""
        logger.info(f"Running standard evaluation with role: {role}")
        
        if gen_kwargs is None:
            gen_kwargs = {
                'max_new_tokens': self.config.get('max_new_tokens', 512),
                'temperature': self.config.get('temperature', 0.7),
                'batch_size': self.config.get('batch_size', 8),
                'role': role
            }
        else:
            gen_kwargs['role'] = role
        
        # 检查是否为多轮对话
        is_multi_turn = any(
            hasattr(instance, 'queries') and instance.queries 
            for instance in instances
        )
        
        if is_multi_turn:
            logger.info("Detected multi-turn conversation format")
            multi_turn_generate_with_hga(self.model, self.tokenizer, instances, gen_kwargs)
        else:
            logger.info("Using standard single-turn generation")
            generate_with_hga(self.model, self.tokenizer, instances, gen_kwargs)
        
        # 如果启用HGA特性，计算效用分数
        if use_hga_features:
            logger.info("Calculating utility scores...")
            calculate_utility_scores(instances, self.utility_model)
        
        return instances
    
    def evaluate_self_play(
        self,
        instances: List[EvalInstance],
        num_dialogues_per_instruction: int = 3,
        use_mcts: bool = True
    ) -> List[EvalInstance]:
        """自我博弈评估"""
        logger.info(f"Running self-play evaluation with {num_dialogues_per_instruction} dialogues per instruction")
        
        from tqdm import tqdm
        
        for instance in tqdm(instances, desc="Self-play evaluation"):
            try:
                # 提取指令
                instruction = instance.behavior or instance.default_target or ""
                if hasattr(instance, 'messages') and instance.messages:
                    # 从消息中提取用户指令
                    for msg in instance.messages:
                        if msg.get('role') == 'user':
                            instruction = msg.get('content', '')
                            break
                
                # 运行多次对话
                best_trajectory = None
                best_utility = float('-inf')
                instance.dialogue_trajectories = []
                
                for dialogue_idx in range(num_dialogues_per_instruction):
                    try:
                        trajectory, final_utility = self.dialogue_game.run_self_play(
                            initial_instruction=instruction,
                            use_mcts=use_mcts
                        )
                        
                        instance.dialogue_trajectories.append({
                            'dialogue_id': dialogue_idx,
                            'trajectory': trajectory,
                            'final_utility': final_utility
                        })
                        
                        # 保存最佳轨迹
                        if final_utility > best_utility:
                            best_utility = final_utility
                            best_trajectory = trajectory
                    
                    except Exception as e:
                        logger.warning(f"Error in dialogue {dialogue_idx}: {e}")
                        continue
                
                # 从最佳轨迹中提取最终结果
                if best_trajectory:
                    instance.trajectory = best_trajectory
                    instance.messages = []
                    
                    for turn in best_trajectory:
                        role = turn.get('role', '')
                        content = turn.get('content', '')
                        
                        if role in ['user', 'assistant']:
                            instance.messages.append({
                                'role': role,
                                'content': content
                            })
                    
                    # 设置最终生成
                    if instance.messages and instance.messages[-1]['role'] == 'assistant':
                        instance.generation = instance.messages[-1]['content']
                    
                    # 检测攻击成功
                    instance.attack_success = self.utility_model._detect_jailbreak_success(
                        instance.generation or ""
                    )
                
            except Exception as e:
                logger.error(f"Error in self-play for instance: {e}")
                instance.generation = ""
                instance.attack_success = False
        
        return instances
    
    def evaluate_with_judge(
        self,
        instances: List[EvalInstance],
        judge_config: Dict[str, Any],
        evaluation_type: str = "standard"
    ) -> float:
        """使用判断器评估"""
        logger.info(f"Evaluating with judge: {judge_config.get('model_name_or_path', 'HGA judge')}")
        
        # 创建判断器
        judge = create_judge(judge_config, self.utility_model)
        
        # 根据评估类型选择评估方法
        if evaluation_type == "multi_turn":
            score = judge.multi_turn_eval(instances)
        else:
            score = judge.evaluate(instances)
        
        logger.info(f"Judge evaluation score: {score:.4f}")
        return score

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="HGA Model Evaluation")
    
    # 基本参数
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True,
                       help="Path to the trained HGA model")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--benchmark", type=str, required=True,
                       help="Path to benchmark dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    
    # 数据参数
    parser.add_argument("--start", type=int, default=0,
                       help="Start index for evaluation")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of instances")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # 评估模式
    parser.add_argument("--eval_mode", type=str, default="auto",
                       choices=["auto", "standard", "self_play", "overrefusal", "multi_jailbreak"],
                       help="Evaluation mode")
    parser.add_argument("--role", type=str, default="defender",
                       choices=["defender", "attacker"],
                       help="Role to use for generation")
    parser.add_argument("--use_hga_features", action="store_true",
                       help="Use HGA-specific features (utility scores, etc.)")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--do_sample", type=bool, default=True,
                       help="Whether to use sampling")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation")
    parser.add_argument("--prefill", type=bool, default=False,
                       help="Use prefill attack")
    parser.add_argument("--use_template", type=bool, default=True,
                       help="Use chat template")
    
    # 自我博弈参数
    parser.add_argument("--num_dialogues", type=int, default=3,
                       help="Number of dialogues per instruction for self-play")
    parser.add_argument("--use_mcts", action="store_true",
                       help="Use MCTS in self-play")
    
    # 判断器参数
    parser.add_argument("-j", "--judge_name_or_path", type=str, default="cais/HarmBench-Llama-2-13b-cls",
                       help="Judge model name or path")
    parser.add_argument("--use_hga_judge", action="store_true",
                       help="Use HGA utility model as judge")
    parser.add_argument("--judge_temperature", type=float, default=0.0,
                       help="Judge temperature")
    parser.add_argument("--judge_max_new_tokens", type=int, default=128,
                       help="Judge max new tokens")
    
    # 过度拒绝评估参数
    parser.add_argument("--judge_overrefusal_mode", type=str, default="hga_enhanced",
                       choices=["strmatch", "hga", "hga_enhanced"],
                       help="Overrefusal evaluation mode")
    
    # 兼容性参数
    parser.add_argument("--save_norms", action="store_true",
                       help="Save activation norms")
    parser.add_argument("--vlm_acc", type=bool, default=False,
                       help="VLM acceleration (not supported with HGA)")
    
    return parser.parse_args()

def evaluate(
    model_name_or_path: str,
    benchmark_path: str,
    start: int,
    limit: Optional[int],
    seed: int,
    overrefusal_judge_mode: str,
    gen_kwargs: Dict[str, Any],
    judge_config: Dict[str, Any],
    eval_mode: str = "auto",
    config_path: Optional[str] = None,
    use_hga_features: bool = True,
    num_dialogues: int = 3,
    use_mcts: bool = False
):
    """主评估函数"""
    if seed is not None:
        set_seed(seed)
    
    # 自动检测评估模式
    if eval_mode == "auto":
        benchmark_name = os.path.basename(benchmark_path).lower()
        if "overrefusal" in benchmark_name or "xstest" in benchmark_name:
            eval_mode = "overrefusal"
        elif "multi_turn" in benchmark_name or "actor" in benchmark_name:
            eval_mode = "multi_jailbreak"
        elif "harmbench" in benchmark_name or "advbench" in benchmark_name:
            eval_mode = "self_play"  # 使用自我博弈进行攻击评估
        else:
            eval_mode = "standard"
    
    logger.info(f"Using evaluation mode: {eval_mode}")
    
    # 运行评估
    try:
        if eval_mode == "overrefusal":
            # 过度拒绝评估
            overrefusal_evaluator = OverrefusalEvaluator(
                model_path=model_name_or_path,
                config_path=config_path
            )
            
            # 初始化标准评估器来加载数据
            evaluator = HGAEvaluator(
                model_path=model_name_or_path,
                config_path=config_path
            )
            instances = evaluator.load_dataset(benchmark_path, start, limit)
            
            results = overrefusal_evaluator.evaluate_dataset(
                instances=instances,
                judge_mode=overrefusal_judge_mode
            )
            
            score = results['statistics']['overrefusal_rate']
            final_instances = results['instances']
            
            # 释放模型资源
            del evaluator.model
            del overrefusal_evaluator.model
        
        elif eval_mode == "multi_jailbreak":
            # 多轮越狱评估
            multi_evaluator = MultiJailbreakEvaluator(
                model_path=model_name_or_path,
                config_path=config_path,
                judge_config=judge_config
            )
            
            results = multi_evaluator.evaluate_dataset(
                data_path=benchmark_path,
                start_idx=start,
                end_idx=start + limit if limit else -1
            )
            
            score = results['statistics']['judge_attack_success_rate']
            final_instances = results['data']
            
            # 释放模型资源
            del multi_evaluator.model
        
        else:
            # 标准评估或自我博弈评估
            evaluator = HGAEvaluator(
                model_path=model_name_or_path,
                config_path=config_path
            )
            
            instances = evaluator.load_dataset(benchmark_path, start, limit)
            
            if eval_mode == "self_play":
                # 自我博弈评估
                final_instances = evaluator.evaluate_self_play(
                    instances=instances,
                    num_dialogues_per_instruction=num_dialogues,
                    use_mcts=use_mcts
                )
            else:
                # 标准评估
                final_instances = evaluator.evaluate_standard(
                    instances=instances,
                    role=gen_kwargs.get('role', 'defender'),
                    use_hga_features=use_hga_features,
                    gen_kwargs=gen_kwargs
                )
            
            # 使用判断器评估
            evaluation_type = "multi_turn" if "multi_turn" in benchmark_path.lower() else "standard"
            score = evaluator.evaluate_with_judge(final_instances, judge_config, evaluation_type)
            
            # 释放模型资源
            del evaluator.model
        
        # 清理GPU内存
        gc.collect()
        torch.cuda.empty_cache()
        
        return score, final_instances
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 打印参数
    logger.info("Evaluation arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # 生成参数
    gen_kwargs = {
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "prefill": args.prefill,
        "use_template": args.use_template,
        "role": args.role
    }
    
    # 判断器配置
    judge_config = {
        "model_name_or_path": args.judge_name_or_path,
        "temperature": args.judge_temperature,
        "max_new_tokens": args.judge_max_new_tokens,
        "seed": args.seed,
        "use_hga_judge": args.use_hga_judge
    }
    
    # 运行评估
    score, instances = evaluate(
        model_name_or_path=args.model_name_or_path,
        benchmark_path=args.benchmark,
        start=args.start,
        limit=args.limit,
        seed=args.seed,
        overrefusal_judge_mode=args.judge_overrefusal_mode,
        gen_kwargs=gen_kwargs,
        judge_config=judge_config,
        eval_mode=args.eval_mode,
        config_path=args.config,
        use_hga_features=args.use_hga_features,
        num_dialogues=args.num_dialogues,
        use_mcts=args.use_mcts
    )
    
    logger.info(f"Final evaluation score: {score:.4f}")
    
    # 保存结果
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备结果数据
        args_dict = vars(args)
        result_data = {
            "args": args_dict,
            "timestamp": datetime.now().isoformat(),
            "evaluation_mode": args.eval_mode,
            "score": score,
            "model_path": args.model_name_or_path,
            "benchmark": args.benchmark,
            "samples": []
        }
        
        # 转换实例为字典格式
        for instance in instances:
            if hasattr(instance, '__dict__'):
                instance_dict = instance.__dict__.copy()
            else:
                instance_dict = asdict(instance)
            
            # 移除大型字段以节省空间
            for field in ['activation_norms', 'tokens', 'trajectory', 'dialogue_trajectories']:
                if field in instance_dict:
                    del instance_dict[field]
            
            result_data["samples"].append(instance_dict)
        
        # 保存到文件
        file_name = os.path.basename(args.benchmark).split('.')[0]
        output_file = output_dir / f"{file_name}_{args.eval_mode}_{args.start}.json"
        
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=handle_non_serializable)
        
        logger.info(f"Results saved to {output_file}")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Mode: {args.eval_mode}")
    print(f"Score: {score:.4f}")

if __name__ == "__main__":
    main()