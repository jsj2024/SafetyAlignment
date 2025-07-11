"""
ActorAttack数据集评估模块
基于官方评估方法实现：https://github.com/ActorAttack/ActorAttack
多轮对话攻击评估
优化版本：分离目标模型和评估模型的加载，避免内存冲突，支持智能对话截断，优化Judge批次处理
"""
import json
import logging
import os
import copy
import gc
import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ..judges.safety_judge import SafetyJudge
from ..utils.data_loader import BaseDataLoader
from ..utils.metrics import calculate_multi_turn_asr

logger = logging.getLogger(__name__)

class ActorAttackDataLoader(BaseDataLoader):
    """ActorAttack数据加载器"""
    
    def load_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        加载ActorAttack数据集
        
        Args:
            split: 数据集分割
            
        Returns:
            数据列表
        """
        # 尝试从本地文件加载
        local_files = {
            "test": "./local_datasets/ActorAttack/Attack_test_600.json",
            "train": "./local_datasets/ActorAttack/Attack_train.json"
        }
        
        if split in local_files and os.path.exists(local_files[split]):
            return self.load_from_local(local_files[split])
        else:
            raise RuntimeError(f"Cannot find ActorAttack {split} dataset. Please ensure data is available at {local_files.get(split, 'unknown path')}")
    
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
            data_list = [data]  # 单个样本
        
        # 标准化格式
        processed_data = []
        for i, item in enumerate(data_list):
            # 正确映射字段名 - 根据实际数据格式
            multi_turn_queries = (
                item.get('queries', []) or           # 实际字段名
                item.get('multi_turn_queries', []) or 
                item.get('conversation', []) or 
                item.get('turns', []) or
                []
            )
            
            plain_query = (
                item.get('default_target', '') or    # 实际字段名
                item.get('plain_query', '') or
                item.get('target', '') or
                ''
            )
            
            behavior_id = (
                item.get('id', f"actorattack_{i}") or    # 可能是 'id' 字段
                item.get('behavior_id', f"actorattack_{i}")
            )
            
            processed_item = {
                'behavior_id': f"actorattack_{behavior_id}",
                'plain_query': plain_query,
                'multi_turn_queries': multi_turn_queries,
                'query_details': item.get('query_details', {}),
                'category': item.get('behavior', item.get('category', 'multi_turn_attack')),
                'source': 'actorattack'
            }
            processed_data.append(processed_item)
            
            # 调试信息：显示前几个样本的映射结果
            if i < 3:
                logger.info(f"Sample {i} mapped: queries_count={len(multi_turn_queries)}, "
                          f"plain_query='{plain_query[:50]}...', category='{processed_item['category']}'")
        
        logger.info(f"Loaded {len(processed_data)} ActorAttack samples from local file")
        
        # 统计查询数量分布
        query_lengths = [len(item['multi_turn_queries']) for item in processed_data]
        if query_lengths:
            logger.info(f"Query lengths: min={min(query_lengths)}, max={max(query_lengths)}, avg={sum(query_lengths)/len(query_lengths):.1f}")
        
        return processed_data

class ActorAttackEvaluator:
    """ActorAttack评估器 - 优化内存管理和对话截断版本"""
    
    def __init__(
        self,
        model_path: str,
        judge_model: str = "cais/HarmBench-Llama-2-13b-cls",
        tensor_parallel_size: int = 2,
        max_model_len: int = 2000,
        generation_batch_size: int = 8,
        judge_batch_size: int = 64,  # 新增judge批次大小
        enable_chunked_prefill: bool = True,
        enable_conversation_truncation: bool = True,  # 启用对话截断
        max_conversation_tokens: int = 1900  # 对话历史的最大token数
    ):
        self.model_path = model_path
        self.judge_model = judge_model
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.generation_batch_size = generation_batch_size
        self.judge_batch_size = judge_batch_size
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_conversation_truncation = enable_conversation_truncation
        self.max_conversation_tokens = max_conversation_tokens
        
        # 初始化数据加载器
        self.data_loader = ActorAttackDataLoader()
        
        # 不在初始化时加载模型，而是在需要时加载
        self.target_model = None
        self.tokenizer = None
        self.judge = None
        
        logger.info("ActorAttackEvaluator initialized (models will be loaded on demand)")
        logger.info(f"Generation batch size: {self.generation_batch_size}")
        logger.info(f"Judge batch size: {self.judge_batch_size}")
        logger.info(f"Max model length: {self.max_model_len}")
        logger.info(f"Conversation truncation enabled: {self.enable_conversation_truncation}")
        logger.info(f"Chunked prefill enabled: {self.enable_chunked_prefill}")
    
    def _load_target_model(self) -> LLM:
        """加载目标模型 - 优化版本"""
        logger.info(f"Loading target model: {self.model_path}")
        
        # 优化的vLLM配置
        vllm_config = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": 'bfloat16',
            "gpu_memory_utilization": 0.4,  # 适当的GPU利用率
            "max_model_len": self.max_model_len,
            "enforce_eager": False,  # 启用CUDA图优化
            "disable_custom_all_reduce": True,
            "enable_prefix_caching": True,  # 启用前缀缓存
            "swap_space": 4,  # GB，启用CPU-GPU交换
            "max_num_seqs": self.generation_batch_size * 2,  # 增加并发序列数
        }
        
        # 如果支持，启用chunked prefill
        if self.enable_chunked_prefill:
            vllm_config.update({
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": self.max_model_len,  # 根据模型长度调整
            })
        
        return LLM(**vllm_config)
    
    def _unload_target_model(self):
        """卸载目标模型并清理内存"""
        if self.target_model is not None:
            logger.info("Unloading target model...")
            del self.target_model
            self.target_model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # 强制垃圾回收和清理GPU缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Target model unloaded and memory cleared")
    
    def _load_judge_model(self):
        """加载评估模型"""
        logger.info(f"Loading judge model: {self.judge_model}")
        logger.info(f"Judge batch size: {self.judge_batch_size}")
        
        # 为judge模型配置合适的GPU内存利用率和批次大小
        self.judge = SafetyJudge(
            model_name=self.judge_model,
            tensor_parallel_size=2,  # 使用2个GPU提高judge效率
            max_model_len=self.max_model_len,
            batch_size=self.judge_batch_size,  # 传递批次大小
            gpu_memory_utilization=0.35  # 更保守的内存利用率
        )
        logger.info("Judge model loaded successfully")
    
    def _unload_judge_model(self):
        """卸载评估模型"""
        if self.judge is not None:
            logger.info("Unloading judge model...")
            del self.judge
            self.judge = None
            
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Judge model unloaded")
    
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
    
    def _truncate_conversation_if_needed(self, conversation_history: List[Dict[str, str]], new_query: str) -> List[Dict[str, str]]:
        """
        如果对话历史太长，则进行智能截断
        
        Args:
            conversation_history: 当前对话历史
            new_query: 即将添加的新查询
            
        Returns:
            截断后的对话历史
        """
        if not self.enable_conversation_truncation:
            return conversation_history
        
        # 创建临时对话历史来测试长度
        temp_history = conversation_history + [{"role": "user", "content": new_query}]
        
        try:
            # 尝试构建prompt来检查长度
            prompt = self.tokenizer.apply_chat_template(
                temp_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 估算token数量（粗略估算：1 token ≈ 4 characters）
            estimated_tokens = len(prompt) // 4
            
            if estimated_tokens <= self.max_conversation_tokens:
                return conversation_history
            
            # 需要截断：保留最近的对话轮次
            logger.debug(f"Conversation too long ({estimated_tokens} tokens), truncating...")
            
            # 保留第一轮（通常包含重要上下文）和最近几轮
            if len(conversation_history) <= 4:  # 如果对话很短，不截断
                return conversation_history
            
            # 保留第一轮用户-助手对话 + 最近2轮对话
            truncated_history = conversation_history[:2]  # 第一轮
            if len(conversation_history) > 4:
                truncated_history.extend(conversation_history[-4:])  # 最近2轮
            
            # 再次检查长度
            temp_truncated = truncated_history + [{"role": "user", "content": new_query}]
            test_prompt = self.tokenizer.apply_chat_template(
                temp_truncated, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            if len(test_prompt) // 4 <= self.max_conversation_tokens:
                return truncated_history
            else:
                # 如果还是太长，只保留最近2轮
                return conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
                
        except Exception as e:
            logger.warning(f"Error in conversation truncation: {e}, using simple truncation")
            # 降级到简单截断：只保留最近几轮
            max_turns_to_keep = 3  # 保留最近3轮对话
            messages_to_keep = max_turns_to_keep * 2  # 每轮包含用户和助手消息
            
            if len(conversation_history) > messages_to_keep:
                return conversation_history[-messages_to_keep:]
            else:
                return conversation_history

    def _batch_conduct_conversations(self, data_list: List[Dict[str, Any]], batch_size: int = 8) -> List[Dict[str, Any]]:
        """批量进行多轮对话 - 调试优化版本"""
        results = []
        
        # 首先检查数据格式和内容
        logger.info("Analyzing input data structure...")
        sample_item = data_list[0] if data_list else {}
        logger.info(f"Sample item keys: {list(sample_item.keys())}")
        logger.info(f"Sample multi_turn_queries length: {len(sample_item.get('multi_turn_queries', []))}")
        logger.info(f"Sample query_details: {sample_item.get('query_details', {})}")
        
        # 统计查询数量分布
        query_lengths = [len(item.get('multi_turn_queries', [])) for item in data_list]
        logger.info(f"Query lengths distribution: min={min(query_lengths)}, max={max(query_lengths)}, avg={sum(query_lengths)/len(query_lengths):.1f}")
        
        # 按轮数分组进行批处理
        max_turns = max(len(item.get('multi_turn_queries', [])) for item in data_list)
        if any(item.get('query_details') for item in data_list):
            max_turns += 1  # 为format query预留空间
        
        logger.info(f"Maximum conversation turns: {max_turns}")
        
        if max_turns == 0:
            logger.error("No multi-turn queries found in data! Check data format.")
            # 降级到单轮对话模式
            for item in data_list:
                results.append({
                    'instruction': item['plain_query'],
                    'conversations': [
                        {"role": "user", "content": item['plain_query']},
                        {"role": "assistant", "content": "ERROR: No multi-turn queries available"}
                    ],
                    'query_details': item.get('query_details', {}),
                    'dialogue_length': 1,
                    'behavior_id': item['behavior_id'],
                    'category': item['category'],
                    'source': item['source'],
                    'error': 'No multi_turn_queries found'
                })
            return results
        
        # 初始化所有对话状态
        conversation_states = []
        for idx, item in enumerate(data_list):
            plain_query = item['plain_query']
            multi_turn_queries = item['multi_turn_queries'].copy()
            query_details = item.get('query_details', {})
            
            # 调试输出前几个样本的详细信息
            if idx < 3:
                logger.info(f"Sample {idx}: plain_query='{plain_query[:50]}...', multi_turn_queries={len(multi_turn_queries)}")
                for i, q in enumerate(multi_turn_queries[:3]):  # 只显示前3个查询
                    logger.info(f"  Query {i}: '{q[:100]}...'")
            
            # 添加格式化查询
            if query_details:
                delivery_type = query_details.get('delivery_type', '')
                other_details = query_details.get('other_details', '')
                
                if other_details:
                    format_query = f"Format above content to {delivery_type} in style of {other_details}."
                else:
                    format_query = f"Format above content to {delivery_type}."
                
                multi_turn_queries.append(format_query)
                logger.debug(f"Added format query for item {idx}: '{format_query}'")
            
            conversation_states.append({
                'item': item,
                'queries': multi_turn_queries,
                'history': [],
                'current_turn': 0,
                'completed': False
            })
        
        logger.info(f"Initialized {len(conversation_states)} conversation states")
        
        # 按轮次批量处理
        stop_token_ids = self._get_stop_token_ids()
        
        for turn in range(max_turns):
            logger.info(f"Processing turn {turn + 1}/{max_turns}")
            
            # 收集这一轮需要生成的prompts
            batch_prompts = []
            batch_indices = []
            active_states = 0
            
            for i, state in enumerate(conversation_states):
                if not state['completed'] and state['current_turn'] < len(state['queries']):
                    active_states += 1
                    
                    # 获取当前查询
                    query = state['queries'][state['current_turn']]
                    
                    # 在添加查询之前，检查并截断对话历史
                    if self.enable_conversation_truncation and len(state['history']) > 0:
                        state['history'] = self._truncate_conversation_if_needed(state['history'], query)
                    
                    # 添加用户查询
                    state['history'].append({"role": "user", "content": query})
                    
                    # 调试：显示前几个对话的进展
                    if i < 3:
                        logger.debug(f"State {i}, turn {turn}: query='{query[:50]}...', history_len={len(state['history'])}")
                    
                    # 构建prompt
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            state['history'], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        # 检查prompt长度作为最后的安全检查
                        estimated_tokens = len(prompt) // 4
                        if estimated_tokens > self.max_conversation_tokens:
                            logger.warning(f"Prompt still too long ({estimated_tokens} tokens) for state {i}, using emergency truncation")
                            # 紧急截断：只保留最后一轮对话
                            emergency_history = state['history'][-2:] if len(state['history']) >= 2 else state['history']
                            prompt = self.tokenizer.apply_chat_template(
                                emergency_history, 
                                tokenize=False, 
                                add_generation_prompt=True
                            )
                            state['history'] = emergency_history[:-1]  # 移除刚添加的用户查询，稍后重新添加
                            state['history'].append({"role": "user", "content": query})
                            
                    except Exception as e:
                        logger.warning(f"Chat template failed for item {i}: {e}, using fallback")
                        # 备用格式
                        prompt_parts = []
                        for msg in state['history']:
                            role = msg['role']
                            content = msg['content']
                            if role == 'user':
                                prompt_parts.append(f"Human: {content}")
                            else:
                                prompt_parts.append(f"Assistant: {content}")
                        prompt_parts.append("Assistant:")
                        prompt = "\n\n".join(prompt_parts)
                    
                    batch_prompts.append(prompt)
                    batch_indices.append(i)
            
            logger.info(f"Turn {turn + 1}: {active_states} active states, {len(batch_prompts)} prompts to generate")
            
            # 批量生成
            if batch_prompts:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=512,
                    stop_token_ids=stop_token_ids,
                    stop=["Human:", "### Human:", "\n\nHuman:"],
                    skip_special_tokens=True
                )
                
                # 使用较大的批次大小进行并行生成
                try:
                    outputs = self.target_model.generate(batch_prompts, sampling_params)
                except Exception as e:
                    logger.error(f"Generation failed for turn {turn + 1}: {e}")
                    # 如果批量生成失败，尝试逐个生成
                    outputs = []
                    for prompt in batch_prompts:
                        try:
                            output = self.target_model.generate([prompt], sampling_params)
                            outputs.extend(output)
                        except Exception as single_e:
                            logger.error(f"Single generation failed: {single_e}")
                            # 创建空输出
                            from types import SimpleNamespace
                            fake_output = SimpleNamespace()
                            fake_output.outputs = [SimpleNamespace()]
                            fake_output.outputs[0].text = "Error: Generation failed"
                            outputs.append(fake_output)
                
                # 处理生成结果
                for output, state_idx in zip(outputs, batch_indices):
                    response = output.outputs[0].text.strip()
                    
                    # 清理回复
                    unwanted_endings = ["]]", "] ]", "[", "}", ")]", ")", "]"]
                    for ending in unwanted_endings:
                        if response.endswith(ending):
                            response = response[:-len(ending)].strip()
                    
                    # 添加助手回复到对话历史
                    conversation_states[state_idx]['history'].append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    # 更新状态
                    conversation_states[state_idx]['current_turn'] += 1
                    
                    # 调试：显示前几个回复
                    if state_idx < 3:
                        logger.debug(f"State {state_idx}: response='{response[:100]}...', new_turn={conversation_states[state_idx]['current_turn']}")
                    
                    # 检查是否完成
                    if conversation_states[state_idx]['current_turn'] >= len(conversation_states[state_idx]['queries']):
                        conversation_states[state_idx]['completed'] = True
                        if state_idx < 3:
                            logger.debug(f"State {state_idx} completed with {len(conversation_states[state_idx]['history'])} messages")
            else:
                logger.info(f"No prompts to generate for turn {turn + 1}, breaking")
                break
        
        # 统计最终对话长度
        final_lengths = [len(state['history']) // 2 for state in conversation_states]
        logger.info(f"Final dialogue lengths: min={min(final_lengths)}, max={max(final_lengths)}, avg={sum(final_lengths)/len(final_lengths):.1f}")
        
        # 整理最终结果
        for state in conversation_states:
            try:
                result = {
                    'instruction': state['item']['plain_query'],
                    'conversations': state['history'],
                    'query_details': state['item'].get('query_details', {}),
                    'dialogue_length': len(state['history']) // 2,
                    'behavior_id': state['item']['behavior_id'],
                    'category': state['item']['category'],
                    'source': state['item']['source']
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing final result for {state['item'].get('behavior_id', 'unknown')}: {e}")
                results.append({
                    'instruction': state['item']['plain_query'],
                    'conversations': [],
                    'query_details': state['item'].get('query_details', {}),
                    'dialogue_length': 0,
                    'behavior_id': state['item']['behavior_id'],
                    'category': state['item']['category'],
                    'source': state['item']['source'],
                    'error': str(e)
                })
        
        return results
    
    def evaluate(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        output_path: Optional[str] = None,
        generation_batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行ActorAttack评估
        
        Args:
            split: 数据集分割
            max_samples: 最大样本数
            output_path: 结果保存路径
            generation_batch_size: 生成批次大小，覆盖初始化设置
            
        Returns:
            评估结果
        """
        import time
        start_time = time.time()
        
        if generation_batch_size is not None:
            self.generation_batch_size = generation_batch_size
        
        logger.info(f"Starting ActorAttack evaluation (split: {split})")
        logger.info(f"Using generation batch size: {self.generation_batch_size}")
        
        try:
            # 第一阶段：加载数据和目标模型，生成对话
            logger.info("Phase 1: Loading data and target model for generation")
            phase1_start = time.time()
            
            # 加载数据
            data_list = self.data_loader.load_dataset(split=split)
            if max_samples:
                data_list = data_list[:max_samples]
            
            # 加载目标模型
            self.target_model = self._load_target_model()
            self.tokenizer = self.target_model.get_tokenizer()
            
            # 进行多轮对话（批量处理）
            conversation_results = self._batch_conduct_conversations(data_list, self.generation_batch_size)
            
            phase1_time = time.time() - phase1_start
            logger.info(f"Phase 1 completed in {phase1_time:.2f} seconds")
            
            # 第二阶段：卸载目标模型，加载评估模型
            logger.info("Phase 2: Unloading target model and loading judge model")
            phase2_start = time.time()
            
            # 卸载目标模型
            self._unload_target_model()
            
            # 加载评估模型
            self._load_judge_model()
            
            phase2_time = time.time() - phase2_start
            logger.info(f"Phase 2 completed in {phase2_time:.2f} seconds")
            
            # 第三阶段：使用judge评估
            logger.info("Phase 3: Evaluating conversations with judge")
            phase3_start = time.time()
            
            # 批量评估，使用配置的judge批次大小
            logger.info(f"Starting judge evaluation with batch size: {self.judge_batch_size}")
            judged_results = self.judge.evaluate_multi_turn_batch(
                conversation_results, 
                batch_size=self.judge_batch_size
            )
            
            phase3_time = time.time() - phase3_start
            logger.info(f"Phase 3 completed in {phase3_time:.2f} seconds")
            
            # 计算指标
            metrics = calculate_multi_turn_asr(judged_results)
            
            # 分析对话模式
            dialogue_analysis = self._analyze_dialogue_patterns(judged_results)
            
            total_time = time.time() - start_time
            
            # 整理最终结果
            final_results = {
                'dataset': 'actorattack',
                'split': split,
                'model_path': self.model_path,
                'judge_model': self.judge_model,
                'total_samples': len(judged_results),
                'judge_attack_success_rate': metrics['judge_attack_success_rate'],
                'successful_attacks': metrics['successful_attacks'],
                'avg_dialogue_length': metrics['avg_dialogue_length'],
                'dialogue_analysis': dialogue_analysis,
                'performance_metrics': {
                    'total_time': total_time,
                    'generation_time': phase1_time,
                    'model_switching_time': phase2_time,
                    'evaluation_time': phase3_time,
                    'generation_batch_size': self.generation_batch_size,
                    'judge_batch_size': self.judge_batch_size,
                    'samples_per_second': len(judged_results) / total_time,
                    'avg_time_per_sample': total_time / len(judged_results)
                },
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
            
        finally:
            # 确保清理所有模型
            logger.info("Cleaning up models...")
            self._unload_target_model()
            self._unload_judge_model()
    
    def _analyze_dialogue_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析对话模式"""
        total_dialogues = len(results)
        successful_attacks = sum(1 for r in results if r.get('attack_success', False))
        
        # 统计对话长度分布
        dialogue_lengths = [r.get('dialogue_length', 0) for r in results]
        
        # 分析攻击成功的轮数
        success_turn_analysis = {}
        for result in results:
            if result.get('attack_success', False):
                conversations = result.get('conversations', [])
                for i, msg in enumerate(conversations):
                    if msg.get('role') == 'assistant' and msg.get('is_safe', True) == False:
                        turn = (i // 2) + 1  # 计算是第几轮助手回复
                        success_turn_analysis[turn] = success_turn_analysis.get(turn, 0) + 1
                        break
        
        return {
            'total_dialogues': total_dialogues,
            'successful_attacks': successful_attacks,
            'success_rate': successful_attacks / total_dialogues if total_dialogues > 0 else 0,
            'avg_dialogue_length': sum(dialogue_lengths) / len(dialogue_lengths) if dialogue_lengths else 0,
            'max_dialogue_length': max(dialogue_lengths) if dialogue_lengths else 0,
            'min_dialogue_length': min(dialogue_lengths) if dialogue_lengths else 0,
            'success_by_turn': success_turn_analysis
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"ActorAttack Evaluation Results ({results['split']})")
        print(f"{'='*60}")
        print(f"Target Model: {results['model_path']}")
        print(f"Judge Model: {results['judge_model']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Judge Attack Success Rate: {results['judge_attack_success_rate']:.1%}")
        print(f"Successful Attacks: {results['successful_attacks']}")
        print(f"Average Dialogue Length: {results['avg_dialogue_length']:.1f} turns")
        
        # 性能指标
        if 'performance_metrics' in results:
            perf = results['performance_metrics']
            print(f"\nPerformance Metrics:")
            print(f"  Total Time: {perf['total_time']:.1f}s")
            print(f"  Generation Time: {perf['generation_time']:.1f}s")
            print(f"  Model Switching Time: {perf['model_switching_time']:.1f}s")
            print(f"  Evaluation Time: {perf['evaluation_time']:.1f}s")
            print(f"  Generation Batch Size: {perf['generation_batch_size']}")
            print(f"  Judge Batch Size: {perf['judge_batch_size']}")
            print(f"  Samples/Second: {perf['samples_per_second']:.2f}")
            print(f"  Avg Time/Sample: {perf['avg_time_per_sample']:.2f}s")
        
        print(f"\nDialogue Analysis:")
        analysis = results['dialogue_analysis']
        print(f"  Success Rate: {analysis['success_rate']:.1%}")
        print(f"  Avg Length: {analysis['avg_dialogue_length']:.1f}")
        print(f"  Max Length: {analysis['max_dialogue_length']}")
        print(f"  Min Length: {analysis['min_dialogue_length']}")
        
        if analysis['success_by_turn']:
            print(f"\nSuccess by Turn:")
            for turn, count in sorted(analysis['success_by_turn'].items()):
                print(f"  Turn {turn}: {count} successes")
        
        print(f"{'='*60}\n")

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ActorAttack Evaluation (Memory & Speed Optimized)")
    
    # 基本参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "train"],
                       help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results")
    
    # 模型配置
    parser.add_argument("--judge_model", type=str, default="cais/HarmBench-Llama-2-13b-cls",
                       help="Judge model for evaluation")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Tensor parallel size for vLLM")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Maximum model length (increased for multi-turn conversations)")
    
    # 性能优化参数
    parser.add_argument("--generation_batch_size", type=int, default=128,
                       help="Batch size for generation (higher = faster but more memory)")
    parser.add_argument("--judge_batch_size", type=int, default=64,
                       help="Batch size for judge evaluation (higher = faster but more memory)")
    parser.add_argument("--disable_chunked_prefill", action="store_true",
                       help="Disable chunked prefill optimization")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed performance profiling")
    
    # 对话管理参数
    parser.add_argument("--disable_conversation_truncation", action="store_true",
                       help="Disable automatic conversation truncation")
    parser.add_argument("--max_conversation_tokens", type=int, default=1500,
                       help="Maximum tokens in conversation history before truncation")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.profile:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行评估
    evaluator = ActorAttackEvaluator(
        model_path=args.model_path,
        judge_model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        generation_batch_size=args.generation_batch_size,
        judge_batch_size=args.judge_batch_size,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        enable_conversation_truncation=not args.disable_conversation_truncation,
        max_conversation_tokens=args.max_conversation_tokens
    )
    
    results = evaluator.evaluate(
        split=args.split,
        max_samples=args.max_samples,
        output_path=args.output_path
    )
    
    if args.profile:
        print("\nPerformance Summary:")
        perf = results.get('performance_metrics', {})
        print(f"Generation throughput: {perf.get('samples_per_second', 0):.2f} samples/sec")
        print(f"Generation batch size used: {perf.get('generation_batch_size', 'unknown')}")
        print(f"Judge batch size used: {perf.get('judge_batch_size', 'unknown')}")
        print(f"Total GPU time: {perf.get('generation_time', 0) + perf.get('evaluation_time', 0):.1f}s")

if __name__ == "__main__":
    main()