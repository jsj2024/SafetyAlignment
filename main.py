"""
HGA主程序 - 适配 torchrun 分布式训练
"""
import argparse
import logging
import os
import sys
import yaml
import torch
import torch.distributed as dist
from typing import Dict, Any

# 添加项目根目录到Python路径
# 这一行可能不再需要，因为您在脚本中设置了 PYTHONPATH，但保留无害
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---- 从原始文件中移动过来的函数 (保持不变) ----

# 设置日志
# 注意：在多进程中，日志格式可能需要调整以包含rank信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RANK %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_model_simple(config: Dict[str, Any], device: torch.device):
    """简化的模型设置"""
    try:
        from models.hga_llm import HGA_LLM, HGATokenizer
        
        model_config = config['model']
        lora_config = config['lora']
        
        # 加载分词器
        tokenizer = HGATokenizer(model_config['name_or_path'])
        
        # 简化的模型加载
        # 注意: 分布式训练时，device_map 需要特别处理或不使用，由 DDP 控制
        model = HGA_LLM(
            model_name_or_path=model_config['name_or_path'],
            lora_config=lora_config,
            torch_dtype=torch.float16,
            device_map=None,  # DDP 模式下，将 device_map 设置为 None，手动 .to(device)
            max_memory=None
        )
        
        logger.info(f"Model object created. Will be moved to {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # 创建虚拟模型用于测试 (这部分逻辑保留，用于调试)
        from torch import nn
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=512, nhead=8), 
                    num_layers=6
                )
                self.lm_head = nn.Linear(512, 32000)
                self.current_role = 'defender'
            
            def switch_role(self, role: str):
                self.current_role = role
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                from types import SimpleNamespace
                batch_size, seq_len = input_ids.shape
                hidden = torch.randn(seq_len, batch_size, 512, device=input_ids.device)
                output = self.transformer(hidden, hidden)
                logits = self.lm_head(output.transpose(0, 1))
                return SimpleNamespace(logits=logits)
            
            def generate(self, input_ids, **kwargs):
                batch_size = input_ids.shape[0]
                max_new_tokens = kwargs.get('max_new_tokens', 50)
                new_tokens = torch.randint(1, 1000, (batch_size, max_new_tokens), device=input_ids.device)
                return torch.cat([input_ids, new_tokens], dim=1)
        
        class DummyTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2
                
            def __call__(self, text, **kwargs):
                if isinstance(text, str): text = [text]
                input_ids = torch.randint(1, 1000, (len(text), 50))
                return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
            
            def decode(self, token_ids, **kwargs):
                return "This is a dummy response for testing."
        
        model = DummyModel()
        tokenizer = DummyTokenizer()
        
        logger.warning("Using dummy model for testing")
        return model, tokenizer

def create_dummy_dataloader(tokenizer, batch_size: int = 2):
    """创建虚拟数据加载器用于测试"""
    from torch.utils.data import DataLoader, Dataset
    
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.data = [{"instruction": f"Test instruction {i}", "response": f"Test response {i}", "is_safe": True} for i in range(size)]
        def __len__(self): return self.size
        def __getitem__(self, idx): return self.data[idx]
    
    dataset = DummyDataset()
    # 注意：分布式训练需要 DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist.is_initialized() else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)

# ---- 重构后的训练和推理函数 ----

def run_training(config: Dict[str, Any], args):
    """
    统一的训练函数，根据环境变量判断是单GPU还是分布式训练。
    由 torchrun 或直接 python 调用。
    """
    # 1. 初始化分布式环境 (如果需要)
    is_distributed = 'WORLD_SIZE' in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        logger.info(f"Rank {local_rank}: Distributed training initialized on {device}.")
    else:
        local_rank = -1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Single device training on {device}.")

    # 2. 设置模型
    model, tokenizer = setup_model_simple(config, device)
    model.to(device)

    # 3. 如果是分布式，用DDP包装模型
    if is_distributed:
        # `find_unused_parameters=True` 可以帮助调试，但可能会稍慢。如果模型结构简单，可以设为False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
        logger.info(f"Rank {local_rank}: Model wrapped with DDP.")


    # 4. 设置数据
    # 注意：分布式训练时，每个进程都需要创建自己的dataloader
    try:
        from data.hga_dataloader import get_dataloader
        # 使用 DistributedSampler 来确保数据在各GPU间不重复
        train_dataloader = get_dataloader(
            tokenizer=tokenizer,
            dataset_names=config['data']['datasets'],
            batch_size=config['training']['batch_size'],
            split='train',
            max_length=config['data']['max_length'],
            is_distributed=is_distributed
        )
        eval_dataloader = get_dataloader(
            tokenizer=tokenizer,
            dataset_names=config['data']['datasets'],
            batch_size=config['training']['batch_size'],
            split='test',
            max_length=config['data']['max_length'],
            is_distributed=False # 评估通常在单个GPU上进行，或者也用分布式采样器
        )
    except Exception as e:
        logger.warning(f"Rank {local_rank}: Failed to load real data: {e}, using dummy data")
        train_dataloader = create_dummy_dataloader(tokenizer, config['training']['batch_size'])
        eval_dataloader = create_dummy_dataloader(tokenizer, config['training']['batch_size'])
    
    # 5. 设置效用模型 (每个进程都需要)
    try:
        from models.utility_model import UtilityModel
        utility_model = UtilityModel()
    except Exception as e:
        logger.warning(f"Rank {local_rank}: Failed to load utility model: {e}, using dummy")
        class DummyUtilityModel:
            def calculate_utility(self, conversation, response):
                return {'utility': 0.5, 'safety_score': 0.8, 'helpfulness_score': 0.6, 'jailbreak_success': False}
        utility_model = DummyUtilityModel()
    
    # 6. 创建训练器
    from engine.trainer import HGATrainer
    trainer = HGATrainer(
        config=config,
        model=model,  # 传入的已经是DDP包装后的模型了
        tokenizer=tokenizer,
        utility_model=utility_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        local_rank=local_rank
    )
    
    # 7. 开始训练
    trainer.train()
    
    # 8. 清理
    if is_distributed:
        dist.destroy_process_group()

def simple_inference(config: Dict[str, Any], model_path: str):
    """简单推理模式 (与原始代码基本相同)"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = setup_model_simple(config, device)
    model.to(device)

    if model_path and os.path.exists(model_path):
        try:
            # 假设模型有加载适配器的方法
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
            logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
    
    print("\nHGA 推理模式")
    print("输入 'quit' 退出, 'switch' 切换角色")
    current_role = 'defender'
    
    while True:
        try:
            user_input = input(f"[{current_role}] User: ").strip()
            if user_input.lower() == 'quit': break
            if user_input.lower() == 'switch':
                current_role = 'attacker' if current_role == 'defender' else 'defender'
                print(f"角色切换为: {current_role}")
                continue
            if not user_input: continue
            
            model.switch_role(current_role)
            inputs = tokenizer(f"User: {user_input}\nAssistant:", return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
            print(f"[{current_role}] Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n退出...")
            break
        except Exception as e:
            logger.error(f"推理错误: {e}")

# ---- 主入口 ----
def main():
    parser = argparse.ArgumentParser(description="HGA训练/推理 - 适配torchrun")
    parser.add_argument("--mode", choices=['train', 'inference'], required=True)
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--output_dir", default="./hga_output_0711")
    parser.add_argument("--model_path", help="推理时的模型路径")
    
    # 注意：--world_size 和 --local_rank 不再需要从命令行接收
    # torchrun 会通过环境变量传递它们
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['output_dir'] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        run_training(config, args)
    elif args.mode == 'inference':
        simple_inference(config, args.model_path or config['output_dir'])

if __name__ == "__main__":
    main()