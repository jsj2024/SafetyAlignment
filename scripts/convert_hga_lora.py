# scripts/merge_hga_model.py (最终简单版本)

import torch
import os
import argparse
import sys

# 将项目根目录添加到搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.hga_llm import HGA_LLM, HGATokenizer

def main(args):
    # 1. 加载完整的 HGA_LLM 模型
    # 注意：这里的 lora_config 需要和训练时的一致
    lora_config = {
        'r': 16, 'alpha': 32, 'dropout': 0.1,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    print("Loading HGA_LLM...")
    hga_model = HGA_LLM(
        model_name_or_path=args.base_model_path,
        lora_config=lora_config,
        device_map="cpu" # 在CPU上加载以节省GPU
    )

    # 2. 加载训练好的 attacker 和 defender 权重
    print(f"Loading role adapters from {args.hga_model_path}...")
    hga_model.load_role_adapters(args.hga_model_path)
    
    tokenizer = HGATokenizer(args.base_model_path)
    
    # 3. 分别合并并保存两个角色的模型
    for role in ['attacker', 'defender']:
        print(f"\n--- Processing role: {role} ---")
        
        # 切换到指定角色，这将加载正确的LoRA权重
        hga_model.switch_role(role)
        print(f"Switched to {role} role.")
        
        # 使用PEFT的 merge_and_unload() 方法合并权重
        print("Merging LoRA weights...")
        merged_model = hga_model.base_model.merge_and_unload()
        print("Merge complete.")
        
        # 保存合并后的完整模型和tokenizer
        output_path = os.path.join(args.output_dir, f"{role}_merged_model")
        print(f"Saving merged {role} model to {output_path}...")
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.tokenizer.save_pretrained(output_path)
        print(f"Successfully saved {role} model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge HGA LoRA weights and save as full models.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--hga_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)