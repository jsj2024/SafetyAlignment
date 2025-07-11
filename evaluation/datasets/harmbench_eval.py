"""
HarmBench数据集评估模块
基于官方评估方法实现：https://github.com/centerforaisafety/HarmBench
修改版本：顺序使用被评估模型和judge模型，避免GPU内存冲突
"""
import json
import logging
import os
import torch
import gc
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
import multiprocessing
from ..judges.safety_judge import SafetyJudge
from ..utils.data_loader import BaseDataLoader
from ..utils.metrics import calculate_asr

logger = logging.getLogger(__name__)

class HarmBenchDataLoader(BaseDataLoader):
    """HarmBench数据加载器 - 适配本地数据集格式"""
    
    def load_dataset(self, split: str = "train", subset: str = "standard") -> List[Dict[str, Any]]:
        # 直接使用本地文件
        local_path = "./local_datasets/harmbench.json"
        
        if os.path.exists(local_path):
            logger.info(f"Loading from local file: {local_path}")
            return self.load_from_local(local_path)
        else:
            logger.error(f"Local file not found: {local_path}")
            # 如果本地文件不存在，尝试HuggingFace (使用原来的逻辑)
            return self._load_from_huggingface(split, subset)
    
    def load_from_local(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Loading data from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 处理不同的数据结构
            if isinstance(raw_data, dict):
                # 如果是字典，可能包含metadata，寻找实际数据
                if 'data' in raw_data:
                    data_list = raw_data['data']
                elif 'samples' in raw_data:
                    data_list = raw_data['samples']
                elif 'behaviors' in raw_data:
                    data_list = raw_data['behaviors']
                else:
                    # 假设整个dict就是数据
                    data_list = [raw_data]
            elif isinstance(raw_data, list):
                data_list = raw_data
            else:
                raise ValueError(f"Unsupported data format in {file_path}")
            
            logger.info(f"Raw data contains {len(data_list)} items")
            
            # 处理并验证数据
            processed_data = []
            empty_behavior_count = 0
            
            for i, item in enumerate(data_list):
                try:
                    # 使用实际的字段名进行映射
                    processed_item = self._map_local_fields(item, i)
                    
                    # 验证关键字段
                    if not processed_item['behavior'].strip():
                        empty_behavior_count += 1
                        logger.warning(f"Item {i}: empty behavior field")
                        logger.debug(f"Item {i} raw data: {item}")
                        continue
                    
                    processed_data.append(processed_item)
                    
                    # 显示进度
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(data_list)} items")
                        
                except Exception as e:
                    logger.error(f"Error processing item {i}: {e}")
                    logger.debug(f"Item {i} data: {item}")
                    continue
            
            # 报告结果
            logger.info(f"Successfully processed {len(processed_data)} items")
            if empty_behavior_count > 0:
                logger.warning(f"Skipped {empty_behavior_count} items with empty behavior")
            
            # 显示样本数据用于验证
            if processed_data:
                logger.info("Sample processed item:")
                sample = processed_data[0]
                for key, value in sample.items():
                    preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    logger.info(f"  {key}: {preview}")
            
            return processed_data
                
        except Exception as e:
            logger.error(f"Error loading from local file {file_path}: {e}")
            raise
    
    def _map_local_fields(self, item: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """
        映射本地数据集字段到标准格式
        
        根据提供的数据样本，字段映射为：
        - behavior -> behavior  
        - prompt -> prompt (用于构建最终prompt)
        - category -> category
        - context -> context  
        - source -> behavior_id (作为标识符)
        """
        
        # 基础字段映射
        behavior = str(item.get('behavior', '')).strip()
        prompt = str(item.get('prompt', '')).strip()
        category = str(item.get('category', '')).strip()
        context = item.get('context', '')
        source = str(item.get('source', '')).strip()
        
        # 生成behavior_id (因为原数据没有)
        behavior_id = f"{source}_{index:04d}" if source else f"item_{index:04d}"
        
        # 处理context字段
        if context and context.strip():
            processed_context = context.strip()
        else:
            processed_context = None
        
        # 选择最终的behavior内容
        # 如果prompt和behavior都存在且不同，优先使用behavior
        final_behavior = behavior if behavior else prompt
        
        processed_item = {
            'behavior_id': behavior_id,
            'behavior': final_behavior,
            'category': category,
            'context': processed_context,
            'target': '',  # 本地数据集没有target字段
            'source': 'harmbench_local',
            # 保留原始字段用于调试
            '_original_prompt': prompt,
            '_original_source': source
        }
        
        return processed_item
    
    def _load_from_huggingface(self, split: str, subset: str) -> List[Dict[str, Any]]:
        """
        备用方案：从HuggingFace加载 (保持原来的逻辑)
        """
        try:
            logger.info("Trying to load from HuggingFace as fallback...")
            dataset = load_dataset("walledai/HarmBench", subset, split=split)
            
            data_list = []
            for item in dataset:
                processed_item = {
                    'behavior_id': item.get('BehaviorID', ''),
                    'behavior': item.get('Behavior', ''),
                    'category': item.get('Category', ''),
                    'context': item.get('ContextString', None),
                    'target': item.get('Target', ''),
                    'source': 'harmbench_hf'
                }
                data_list.append(processed_item)
                
            logger.info(f"Loaded {len(data_list)} samples from HuggingFace")
            return data_list
            
        except Exception as e:
            logger.error(f"HuggingFace fallback also failed: {e}")
            raise RuntimeError("Both local file and HuggingFace loading failed")
    
    def validate_loaded_data(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证加载的数据
        """
        if not data_list:
            return {"status": "error", "message": "No data loaded"}
        
        # 统计信息
        total_items = len(data_list)
        empty_behavior = sum(1 for item in data_list if not item.get('behavior', '').strip())
        empty_category = sum(1 for item in data_list if not item.get('category', '').strip())
        has_context = sum(1 for item in data_list if item.get('context'))
        
        # 类别分布
        categories = {}
        for item in data_list:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        validation_report = {
            'status': 'success',
            'total_items': total_items,
            'empty_behavior': empty_behavior,
            'empty_category': empty_category,
            'has_context': has_context,
            'category_distribution': categories,
            'sample_items': data_list[:3]  # 前3个样本
        }
        
        logger.info("Data validation report:")
        logger.info(f"  Total items: {total_items}")
        logger.info(f"  Empty behavior: {empty_behavior}")
        logger.info(f"  Empty category: {empty_category}")
        logger.info(f"  Has context: {has_context}")
        logger.info(f"  Categories: {list(categories.keys())}")
        
        return validation_report

class HarmBenchEvaluator:
    """HarmBench评估器 - 顺序使用模型版本"""
    
    def __init__(
        self,
        model_path: str,
        judge_model_path: str = "/home/models/Qwen3-32B",
        tensor_parallel_size: int = 2,  # 默认单GPU避免内存问题
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.5
    ):
        self.model_path = model_path
        self.judge_model_path = judge_model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # 初始化数据加载器
        self.data_loader = HarmBenchDataLoader()
        
        # 注意：不在初始化时加载模型，而是在需要时动态加载
        self.llm = None
        self.judge = None
        
        logger.info("HarmBenchEvaluator initialized (models will be loaded on demand)")
    
    def _load_target_model(self) -> LLM:
        """加载被评估的目标模型"""
        logger.info(f"Loading target model: {self.model_path}")
        
        # 清理GPU内存
        self._cleanup_gpu_memory()
        
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype='bfloat16',
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=False, 
            enable_prefix_caching=True, 
            max_num_seqs=256,
            swap_space=2,
            disable_custom_all_reduce=True
        )
        
        logger.info("Target model loaded successfully")
        return llm
    
    def _load_judge_model(self) -> SafetyJudge:
        """加载judge模型"""
        logger.info(f"Loading judge model: {self.judge_model_path}")
        
        # 清理GPU内存
        self._cleanup_gpu_memory()
        
        judge = SafetyJudge(
            model_name=self.judge_model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization
        )
        
        logger.info("Judge model loaded successfully")
        return judge
    
    def _cleanup_gpu_memory(self):
        """清理GPU内存"""
        logger.info("Cleaning up GPU memory...")
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 强制垃圾回收
        gc.collect()
        
        # 显示当前GPU内存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    def _unload_model(self, model_name: str = "target"):
        """卸载模型释放内存"""
        logger.info(f"Unloading {model_name} model...")
        
        if model_name == "target" and self.llm is not None:
            del self.llm
            self.llm = None
        elif model_name == "judge" and self.judge is not None:
            del self.judge
            self.judge = None
        
        # 清理内存
        self._cleanup_gpu_memory()
        logger.info(f"{model_name.capitalize()} model unloaded")
    
    def _build_prompt(self, behavior: str, context: Optional[str], tokenizer) -> str:
        """
        【Llama-3.1 关键优化】
        使用模型的分词器(tokenizer)来构建正确的聊天提示词。
        这能确保模型被正确地激活以遵循指令。
        """
        # 1. 组合用户的完整提问内容
        user_content = behavior.strip()
        if context and context.strip():
            # 将上下文放在行为之前，形成一个完整的请求
            user_content = f"{context.strip()}\n\n{user_content}"
        
        # 2. 构建符合 apply_chat_template 方法要求的消息列表
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        # 3. 使用 tokenizer.apply_chat_template 来生成最终的prompt字符串
        #    - tokenize=False: 返回字符串，而不是token id
        #    - add_generation_prompt=True: 自动在末尾添加 '<|start_header_id|>assistant<|end_header_id|>\n\n'
        #      这正是我们需要的，它会提示模型开始生成回复。
        try:
            prompt_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt_str
        except Exception as e:
            logger.error(f"Failed to apply chat template: {e}. Falling back to basic prompt.")
            # 如果模板应用失败（例如tokenizer配置问题），回退到原始方式
            return user_content
    
    def _get_stop_token_ids(self, tokenizer) -> List[int]:
        """
        修复后的stop token配置 - 避免过度截断
        """
        stop_tokens = []
        
        # 只添加真正的模型结束标记
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            stop_tokens.append(tokenizer.eos_token_id)
        
        # 只保留模型特定的结束标记，移除通用模式
        model_specific_stops = [
            "<|eot_id|>",      # Llama3 系列
            "<|end_of_turn|>", # Gemma 系列  
            "<|im_end|>",      # Qwen 系列 (ChatML格式)
            # "</s>",            # 大多数模型的结束标记
            "<|endoftext|>",   # GPT 系列
        ]
        
        for pattern in model_specific_stops:
            try:
                token_ids = tokenizer.encode(pattern, add_special_tokens=False)
                # 重要：只有当pattern是单个token时才添加
                if len(token_ids) == 1 and token_ids[0] not in stop_tokens:
                    stop_tokens.append(token_ids[0])
            except:
                continue
        
        logger.info(f"使用的stop token IDs: {stop_tokens}")
        return stop_tokens

    def _generate_responses(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("="*60)
        logger.info("阶段1：使用目标模型生成回复")
        logger.info("="*60)
        
        self.llm = self._load_target_model()
        tokenizer = self.llm.get_tokenizer()
        
        try:
            # 构建prompts并过滤空行为
            logger.info("Building prompts using Llama-3.1 chat template...")
            valid_items = []  
            prompts = []      
            
            for item in data_list:
                behavior = item.get('behavior', '').strip()
                if not behavior:
                    continue
                    
                prompt = self._build_prompt(behavior, item.get('context'), tokenizer)
                if prompt:
                    prompts.append(prompt)
                    valid_items.append(item)  # 同时记录有效的item
            
            logger.info(f"Generated {len(prompts)} valid prompts (filtered out {len(data_list) - len(prompts)} empty items)")
            
            # **修复后的采样参数**
            sampling_params = SamplingParams(
                temperature=0.7,           # 提高温度获得更自然的回复
                top_p=0.9,                # 适中的top_p
                max_tokens=512,          # 增加最大token数
                repetition_penalty=1.05,   # 降低重复惩罚
                stop_token_ids=self._get_stop_token_ids(tokenizer),
                skip_special_tokens=True,
            )
            
            logger.info(f"采样参数: temperature={sampling_params.temperature}, "
                    f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
            
            # 批量生成
            batch_size = 128  
            results = []
            
            for i in tqdm(range(0, len(prompts), batch_size), desc="生成回复中"):
                batch_prompts = prompts[i:i+batch_size]
                batch_items = valid_items[i:i+batch_size]  
                
                try:
                    batch_outputs = self.llm.generate(batch_prompts, sampling_params)
                    
                    for j, output in enumerate(batch_outputs):
                        result = batch_items[j].copy()
                        result['prompt'] = batch_prompts[j]
                        
                        response_text = output.outputs[0].text.strip()
                        result['response'] = response_text
                        result['response_length'] = len(response_text)
                        result['finish_reason'] = output.outputs[0].finish_reason
                        
                        # 检测异常短回复
                        if len(response_text) < 10:
                            logger.warning(f"检测到异常短回复 (长度:{len(response_text)}): "
                                        f"'{response_text}' (结束原因: {output.outputs[0].finish_reason})")
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"批次 {i//batch_size} 生成失败: {e}")
                    # 失败时添加错误标记
                    for j in range(len(batch_prompts)):
                        result = batch_items[j].copy()
                        result['response'] = "[生成失败]"
                        result['response_length'] = 0
                        result['finish_reason'] = "error"
                        results.append(result)
            
            # 统计回复质量
            response_lengths = [r['response_length'] for r in results]
            if response_lengths:
                avg_length = sum(response_lengths) / len(response_lengths)
                short_count = sum(1 for l in response_lengths if l < 50)
                
                logger.info(f"回复质量统计:")
                logger.info(f"  平均长度: {avg_length:.1f} 字符")
                logger.info(f"  最短: {min(response_lengths)}, 最长: {max(response_lengths)}")
                logger.info(f"  异常短回复 (<50字符): {short_count}/{len(results)}")
                
                # 如果短回复过多，发出警告
                if short_count > len(results) * 0.3:
                    logger.warning(f"检测到过多异常短回复 ({short_count}/{len(results)})，"
                                f"可能需要调整stop token配置！")
            
            return results
            
        finally:
            self._unload_model("target")

    def _clean_response(self, response: str) -> str:
        if not response:
            return response
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if len(line) > 50:
                words = line.split()
                if len(words) > 10:
                    # 检查是否有连续重复的词组
                    cleaned_words = []
                    skip_until = 0
                    
                    for i, word in enumerate(words):
                        if i < skip_until:
                            continue
                        
                        # 检查接下来是否有重复模式
                        pattern_found = False
                        for pattern_len in range(1, min(5, len(words) - i)):
                            pattern = words[i:i+pattern_len]
                            repeat_count = 0
                            
                            # 检查这个模式重复多少次
                            pos = i
                            while (pos + pattern_len <= len(words) and 
                                   words[pos:pos+pattern_len] == pattern):
                                repeat_count += 1
                                pos += pattern_len
                            
                            # 如果重复超过2次，认为是异常
                            if repeat_count > 2:
                                cleaned_words.extend(pattern)
                                skip_until = pos
                                pattern_found = True
                                break
                        
                        if not pattern_found:
                            cleaned_words.append(word)
                    
                    line = ' '.join(cleaned_words)
            
            cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines)

        if len(cleaned_response) > 1000:
            cleaned_response = cleaned_response[:1000] + "..."
        
        return cleaned_response
    
    def _evaluate_responses(self, results_with_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        阶段2：使用judge模型评估回复
        """
        logger.info("="*60)
        logger.info("PHASE 2: Evaluating responses with judge model")
        logger.info("="*60)
        
        # 加载judge模型
        self.judge = self._load_judge_model()
        
        try:
            logger.info(f"Evaluating {len(results_with_responses)} responses...")
            judged_results = self.judge.evaluate_batch(results_with_responses, evaluation_type="harmbench")
            
            logger.info(f"Successfully evaluated {len(judged_results)} responses")
            return judged_results
            
        finally:
            # 确保卸载judge模型
            self._unload_model("judge")
    
    def _get_stop_token_ids(self, tokenizer) -> List[int]:
        """
        获取停止token ID - 修复版本
        避免过于激进的停止条件导致回复过短
        """
        stop_tokens = []
        
        # 只添加真正的结束token
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            stop_tokens.append(tokenizer.eos_token_id)
        
        # 只添加模型特定的结束标记，移除通用的停止模式
        model_specific_stops = [
            "<|eot_id|>",      # Llama3系列
            # "<|end_of_turn|>", # Gemma系列  
            # "<|im_end|>",      # ChatML格式、Qwen
            # "</s>",            # 多数模型的结束标记
            # "<|endoftext|>",   # GPT系列
        ]
        
        for pattern in model_specific_stops:
            try:
                # 确保作为单个token编码
                token_ids = tokenizer.encode(pattern, add_special_tokens=False)
                # 只有当pattern被编码为单个token时才添加
                if len(token_ids) == 1 and token_ids[0] not in stop_tokens:
                    stop_tokens.append(token_ids[0])
            except Exception as e:
                logger.debug(f"Failed to encode stop pattern '{pattern}': {e}")
                continue
        
        logger.info(f"Using stop token IDs: {stop_tokens}")
        return stop_tokens
    
    def evaluate(
        self,
        subset: str = "standard",
        max_samples: Optional[int] = None,
        output_path: Optional[str] = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        运行HarmBench评估 - 分阶段执行
        
        Args:
            subset: HarmBench子集
            max_samples: 最大样本数
            output_path: 结果保存路径
            save_intermediate: 是否保存中间结果
            
        Returns:
            评估结果
        """
        logger.info(f"Starting HarmBench evaluation (subset: {subset})")
        logger.info(f"Target model: {self.model_path}")
        logger.info(f"Judge model: {self.judge_model_path}")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")
        
        # 加载数据
        data_list = self.data_loader.load_dataset(subset=subset)
        if max_samples:
            data_list = data_list[:max_samples]
        
        logger.info(f"Loaded {len(data_list)} samples for evaluation")
        
        # 阶段1：生成回复
        results_with_responses = self._generate_responses(data_list)
        
        # 保存中间结果
        if save_intermediate and output_path:
            intermediate_path = output_path.replace('.json', '_responses.json')
            os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(results_with_responses, f, indent=2, ensure_ascii=False)
            logger.info(f"Intermediate responses saved to {intermediate_path}")
        
        # 阶段2：评估回复
        judged_results = self._evaluate_responses(results_with_responses)
        
        # 计算指标
        logger.info("Calculating metrics...")
        metrics = calculate_asr(judged_results)
        
        # 按类别计算ASR
        category_metrics = self._calculate_category_metrics(judged_results)
        
        # 整理最终结果
        final_results = {
            'dataset': 'harmbench',
            'subset': subset,
            'model_path': self.model_path,
            'judge_model_path': self.judge_model_path,
            'tensor_parallel_size': self.tensor_parallel_size,
            'total_samples': len(judged_results),
            'overall_asr': metrics['asr'],
            'successful_attacks': metrics['successful_attacks'],
            'category_metrics': category_metrics,
            'detailed_results': judged_results,
            'evaluation_config': {
                'max_model_len': self.max_model_len,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'sequential_evaluation': True  # 标记使用顺序评估
            }
        }
        
        # 保存最终结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Final results saved to {output_path}")
        
        # 打印摘要
        self._print_summary(final_results)
        
        return final_results
    
    def _calculate_category_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """计算各类别的ASR"""
        category_stats = {}
        
        for result in results:
            category = result.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'successful': 0}
            
            category_stats[category]['total'] += 1
            if result.get('is_jailbreak', False):
                category_stats[category]['successful'] += 1
        
        # 计算ASR
        category_metrics = {}
        for category, stats in category_stats.items():
            asr = stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0
            category_metrics[category] = {
                'asr': asr,
                'successful_attacks': stats['successful'],
                'total_samples': stats['total']
            }
        
        return category_metrics
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*70}")
        print(f"HarmBench Evaluation Results ({results['subset']})")
        print(f"{'='*70}")
        print(f"Target Model: {results['model_path']}")
        print(f"Judge Model: {results['judge_model_path']}")
        print(f"Evaluation Mode: Sequential (memory-efficient)")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Overall ASR: {results['overall_asr']:.1%}")
        print(f"Successful Attacks: {results['successful_attacks']}")
        
        print(f"\nCategory Breakdown:")
        for category, metrics in results['category_metrics'].items():
            print(f"  {category:20}: {metrics['asr']:6.1%} "
                  f"({metrics['successful_attacks']:3d}/{metrics['total_samples']:3d})")
        
        print(f"{'='*70}\n")
        
        # 额外的GPU内存信息
        if torch.cuda.is_available():
            print("GPU Memory Status:")
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            print()

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HarmBench Evaluation - Sequential Mode")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the target model to evaluate")
    parser.add_argument("--judge_model_path", type=str, default="/home/models/Qwen3-32B",
                       help="Path to the judge model")
    parser.add_argument("--subset", type=str, default="standard",
                       choices=["standard", "contextual", "copyright"],
                       help="HarmBench subset to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output_path", type=str, default="./evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                       help="GPU memory utilization ratio")
    
    args = parser.parse_args()
    
    # 设置多进程启动方法
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误
    
    # 创建输出路径
    output_path = os.path.join(args.output_path, "harmbench_results.json")
    
    # 运行评估
    evaluator = HarmBenchEvaluator(
        model_path=args.model_path,
        judge_model_path=args.judge_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    results = evaluator.evaluate(
        subset=args.subset,
        max_samples=args.max_samples,
        output_path=output_path
    )
    
    logger.info("HarmBench evaluation completed successfully!")

if __name__ == "__main__":
    # 设置多进程启动方法
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()