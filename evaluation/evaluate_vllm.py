# evaluation/evaluate_vllm.py

import argparse
import json
import os
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoConfig
# 从项目中导入数据结构和工具函数
from api import EvalInstance
from utils import handle_non_serializable

"""
使用 vLLM 引擎进行高性能推理的评估脚本。
它加载转换后的 HGA LoRA 适配器，并以极高的吞吐量生成结果。
"""

def load_benchmark(benchmark_path: str, limit: Optional[int] = None) -> List[EvalInstance]:
    """从JSON文件加载评估数据，并转换为EvalInstance对象列表。"""
    print(f"Loading benchmark from: {benchmark_path}")
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 智能确定数据列表
    if 'samples' in data and isinstance(data['samples'], list):
        data_list = data['samples']
    elif 'data' in data and isinstance(data['data'], list):
        data_list = data['data']
    elif isinstance(data, list):
        data_list = data
    else:
        raise ValueError(f"Cannot find a list of samples in {benchmark_path}")

    if limit:
        data_list = data_list[:limit]

    instances = []
    # 获取EvalInstance所有已定义的字段名，以安全地创建实例
    from dataclasses import fields
    instance_fields = {f.name for f in fields(EvalInstance)}

    for d in data_list:
        filtered_data = {k: v for k, v in d.items() if k in instance_fields}
        instances.append(EvalInstance(**filtered_data))
    
    print(f"Loaded {len(instances)} instances.")
    return instances

def get_safe_stop_tokens(tokenizer) -> List[int]:
    """安全地获取停止标记，避免None值和无效token_id"""
    stop_tokens = []
    
    # 添加EOS token
    if tokenizer.eos_token_id is not None:
        stop_tokens.append(tokenizer.eos_token_id)
    
    # 尝试添加常见的停止标记
    common_stop_tokens = [
        "<|eot_id|>", "<|end_of_turn|>", "<|im_end|>", 
        "</s>", "<|endoftext|>", "[INST]", "[/INST]"
    ]
    
    for token in common_stop_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            # 确保token_id有效且不是unk_token
            if (token_id is not None and 
                token_id != tokenizer.unk_token_id and 
                token_id > 0 and 
                token_id not in stop_tokens):
                stop_tokens.append(token_id)
        except:
            continue  # 如果转换失败，跳过这个token
    
    # 添加一些通用的停止字符串对应的token_id
    stop_strings = ["Human:", "Assistant:", "### Response:", "### Instruction:"]
    for stop_str in stop_strings:
        try:
            tokens = tokenizer.encode(stop_str, add_special_tokens=False)
            if tokens and len(tokens) == 1:  # 只添加单个token的停止字符串
                token_id = tokens[0]
                if token_id not in stop_tokens:
                    stop_tokens.append(token_id)
        except:
            continue
    
    print(f"Using stop token IDs: {stop_tokens}")
    return stop_tokens

def build_prompt_safely(instance: EvalInstance, tokenizer) -> str:
    """安全地构建prompt，避免chat template问题"""
    if instance.messages:
        try:
            # 尝试使用chat template
            prompt = tokenizer.apply_chat_template(
                instance.messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            print(f"Warning: Chat template failed ({e}), falling back to simple concatenation")
            # 如果chat template失败，手动构建prompt
            prompt_parts = []
            for msg in instance.messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt_parts.append(f"Human: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(f"{role}: {content}")
            
            prompt_parts.append("Assistant:")  # 添加生成提示
            return "\n\n".join(prompt_parts)
    else:
        # 没有messages时的备用方案
        content = instance.behavior or instance.default_target or ""
        return f"Human: {content}\n\nAssistant:"

def main(args):
    # --- 步骤 1：根据角色，确定要加载的合并后模型的路径 ---
    if args.role == 'defender':
        model_path_to_load = args.defender_merged_model_path
    elif args.role == 'attacker':
        model_path_to_load = args.attacker_merged_model_path
    else:
        # 这个分支理论上不会进入，因为argparse已经做了检查
        raise ValueError(f"Invalid role: {args.role}")
        
    print(f"Target role: '{args.role}'")
    print(f"Loading FULL MERGED model and tokenizer from: {model_path_to_load}")
    
    # --- 步骤 2：初始化 vLLM 引擎 ---
    # vLLM会从指定的路径同时加载模型和正确的tokenizer
    llm = LLM(
        model=model_path_to_load,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype='bfloat16',
        gpu_memory_utilization=0.5,
        # max_model_len=2048,  # 添加这一行
        # 添加一些有助于稳定性的参数
        enforce_eager=True,  # 使用eager模式，更稳定
        disable_custom_all_reduce=True  # 避免一些并行计算问题
    )

    # --- 步骤 3：准备数据和Prompt ---
    instances = load_benchmark(args.benchmark_path, args.limit)
    # 直接从vLLM引擎获取已经加载好的、正确的tokenizer
    tokenizer = llm.get_tokenizer()
    
    # 获取安全的停止标记
    stop_token_ids = get_safe_stop_tokens(tokenizer)
    
    prompts_to_generate = []
    for instance in instances:
        prompt = build_prompt_safely(instance, tokenizer)
        prompts_to_generate.append(prompt)

    # --- 步骤 4：设置采样参数 ---
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,  # 轻微的重复惩罚
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids,
        # 添加字符串停止条件作为额外保险
        stop=[
            "Human:", "### Human:", "### Instruction:", 
            "\n\nHuman:", "\n\n### Human:", "\n\n### Instruction:",
            "<|im_end|>", "<|eot_id|>", "<|end_of_turn|>"
        ],
        # 改进采样策略
        top_k=50,  # 添加top_k采样
        frequency_penalty=0.1,  # 轻微的频率惩罚
        presence_penalty=0.1,   # 轻微的存在惩罚
        skip_special_tokens=True,  # 跳过特殊token
    )

    # --- 步骤 5：批量生成 ---
    print(f"Generating {len(prompts_to_generate)} responses...")
    print(f"Sampling parameters: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_new_tokens}")
    
    # 打印第一个prompt作为调试信息
    if prompts_to_generate:
        print(f"\nFirst prompt preview:\n{prompts_to_generate[0][:500]}...\n")
    
    # 现在不再需要lora_request了
    outputs = llm.generate(prompts_to_generate, sampling_params)

    # --- 步骤 6：处理并保存结果 ---
    print("Processing and saving results...")
    for i, output in enumerate(tqdm(outputs, desc="Saving results")):
        # 清理生成的文本
        generated_text = output.outputs[0].text
        
        # 移除可能的截断标记
        generated_text = generated_text.strip()
        
        # 移除常见的不完整结尾
        unwanted_endings = ["]]", "] ]", "[", "}", ")]", ")", "]"]
        for ending in unwanted_endings:
            if generated_text.endswith(ending):
                generated_text = generated_text[:-len(ending)].strip()
        
        instances[i].generation = generated_text
        instances[i].role = args.role
    
    results_to_save = [asdict(instance) for instance in instances]
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=handle_non_serializable)
    
    print(f"Evaluation complete. Results saved to: {args.output_path}")
    
    # 输出一些统计信息
    avg_length = sum(len(instance.generation) for instance in instances) / len(instances)
    print(f"Average generation length: {avg_length:.1f} characters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-performance HGA evaluation using MERGED models with vLLM.")
    
    # --- 关键修改：更新参数定义以匹配新的逻辑 ---
    parser.add_argument("--attacker_merged_model_path", type=str, required=True, help="Path to the FULL MERGED attacker model directory.")
    parser.add_argument("--defender_merged_model_path", type=str, required=True, help="Path to the FULL MERGED defender model directory.")
    # ---------------------------------------------
    parser.add_argument("--benchmark_path", type=str, required=True, help="Path to the benchmark JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generation results.")
    parser.add_argument("--role", type=str, default="defender", choices=["attacker", "defender"], help="Which role to use for generation.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of instances to evaluate for a quick test.")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model sequence length")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    args = parser.parse_args()
    main(args)