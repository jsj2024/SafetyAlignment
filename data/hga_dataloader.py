"""
HGA数据加载器 - 支持多种对话安全数据集
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler 
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any, Optional, Union
import logging
import random
import numpy as np
import torch.distributed as dist
logger = logging.getLogger(__name__)

class HGADataset(Dataset):
    """HGA训练数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
        role_specific: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.role_specific = role_specific
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 提取指令和回答
        instruction = item.get('instruction', item.get('prompt', ''))
        response = item.get('response', item.get('output', ''))
        
        # 构建对话格式
        conversation = f"User: {instruction}\nAssistant: {response}"
        
        # 分词化
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'instruction': instruction,
            'response': response,
            'is_safe': item.get('is_safe', True),
            'category': item.get('category', 'unknown')
        }

class PreferenceDataset(Dataset):
    """偏好对比数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # 分词化chosen和rejected
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(),
            'prompt': prompt
        }

def load_local_dataset(dataset_name: str, split: str = 'train') -> List[Dict]:
    """加载本地数据集"""
    
    # 本地数据集路径映射
    local_paths = {
        'beavertails': './local_datasets/BeaverTails',
        'hh-rlhf': './local_datasets/hh-rlhf',
        'safemtdata': './local_datasets/SafeMTData',
        'actor_attack': './local_datasets/ActorAttack',
        'harmbench': './local_datasets/harmbench.json',
        'advbench': './local_datasets/advbench.json'
    }
    
    dataset_key = dataset_name.lower()
    
    if dataset_key == 'beavertails':
        return load_beavertails_local(split)
    elif dataset_key == 'hh-rlhf':
        return load_hh_rlhf_local(split)
    elif dataset_key == 'safemtdata':
        return load_safemtdata_local(split)
    elif dataset_key in ['actor_attack', 'harmbench', 'advbench']:
        return load_json_dataset_local(local_paths[dataset_key])
    else:
        # 尝试从Hugging Face加载
        return load_huggingface_dataset(dataset_name, split)

def load_beavertails_local(split: str = 'train') -> List[Dict]:
    """加载本地BeaverTails数据集"""
    base_path = './local_datasets/BeaverTails'
    
    try:
        # 尝试不同的文件命名模式
        possible_files = [
            f'{split}.json',
            f'beavertails_{split}.json',
            f'BeaverTails_{split}.json',
            'train.json' if split == 'train' else 'test.json'
        ]
        
        data_file = None
        for filename in possible_files:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                data_file = filepath
                break
        
        if data_file is None:
            # 列出目录中的文件
            if os.path.exists(base_path):
                files = os.listdir(base_path)
                logger.warning(f"Available files in {base_path}: {files}")
                # 尝试使用第一个JSON文件
                json_files = [f for f in files if f.endswith('.json')]
                if json_files:
                    data_file = os.path.join(base_path, json_files[0])
                    logger.info(f"Using fallback file: {json_files[0]}")
        
        if data_file and os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据格式
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'examples' in data:
                    data = data['examples']
                else:
                    # 取第一个列表值
                    for key, value in data.items():
                        if isinstance(value, list):
                            data = value
                            break
            
            # 转换为标准格式
            processed_data = []
            for item in data:
                processed_item = {
                    'instruction': item.get('prompt', item.get('question', item.get('input', ''))),
                    'response': item.get('response', item.get('answer', item.get('output', ''))),
                    'is_safe': item.get('is_safe', item.get('safe', True)),
                    'category': item.get('category', 'unknown')
                }
                
                # 确保有有效的指令和回答
                if processed_item['instruction'] and processed_item['response']:
                    processed_data.append(processed_item)
            
            logger.info(f"Loaded {len(processed_data)} examples from BeaverTails {split}")
            return processed_data
            
    except Exception as e:
        logger.error(f"Error loading BeaverTails local data: {e}")
    
    # 如果本地加载失败，尝试在线加载
    logger.warning("Falling back to online BeaverTails dataset")
    return load_huggingface_dataset('PKU-Alignment/BeaverTails', split)

def load_hh_rlhf_local(split: str = 'train') -> List[Dict]:
    """加载本地HH-RLHF数据集"""
    base_path = './local_datasets/hh-rlhf'
    
    try:
        # 查找数据文件
        split_mapping = {'train': 'train', 'test': 'test', 'validation': 'test'}
        actual_split = split_mapping.get(split, split)
        
        possible_files = [
            f'{actual_split}.json',
            f'hh_rlhf_{actual_split}.json',
            f'helpful-{actual_split}.json',
            f'harmless-{actual_split}.json'
        ]
        
        all_data = []
        
        for filename in possible_files:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict) and 'data' in data:
                    all_data.extend(data['data'])
        
        if not all_data and os.path.exists(base_path):
            # 尝试读取目录中的所有JSON文件
            for filename in os.listdir(base_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(base_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data[:100])  # 限制数量
                    except:
                        continue
        
        # 转换格式
        processed_data = []
        for item in all_data:
            if 'chosen' in item and 'rejected' in item:
                # 偏好格式
                processed_data.append({
                    'instruction': item.get('prompt', ''),
                    'response': item['chosen'],
                    'is_safe': True,
                    'category': 'helpful'
                })
                processed_data.append({
                    'instruction': item.get('prompt', ''),
                    'response': item['rejected'],
                    'is_safe': False,
                    'category': 'harmful'
                })
            else:
                # 标准格式
                processed_data.append({
                    'instruction': item.get('prompt', item.get('input', '')),
                    'response': item.get('response', item.get('output', '')),
                    'is_safe': item.get('is_safe', True),
                    'category': item.get('category', 'unknown')
                })
        
        logger.info(f"Loaded {len(processed_data)} examples from HH-RLHF {split}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading HH-RLHF local data: {e}")
        return []

def load_safemtdata_local(split: str = 'train') -> List[Dict]:
    """加载本地SafeMTData数据集"""
    base_path = './local_datasets/SafeMTData'
    
    try:
        # 查找数据文件
        data_file = os.path.join(base_path, f'{split}.json')
        if not os.path.exists(data_file):
            # 查找其他可能的文件
            for filename in os.listdir(base_path):
                if filename.endswith('.json'):
                    data_file = os.path.join(base_path, filename)
                    break
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            processed_data = []
            for item in data:
                processed_data.append({
                    'instruction': item.get('source', item.get('input', '')),
                    'response': item.get('target', item.get('output', '')),
                    'is_safe': item.get('is_safe', True),
                    'category': item.get('category', 'translation')
                })
            
            logger.info(f"Loaded {len(processed_data)} examples from SafeMTData {split}")
            return processed_data
            
    except Exception as e:
        logger.error(f"Error loading SafeMTData local data: {e}")
        return []

def load_json_dataset_local(filepath: str) -> List[Dict]:
    """加载通用JSON数据集"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的数据结构
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'examples' in data:
                data = data['examples']
            else:
                # 查找第一个列表字段
                for key, value in data.items():
                    if isinstance(value, list):
                        data = value
                        break
        
        processed_data = []
        for item in data:
            # 尝试不同的字段名
            instruction_fields = ['prompt', 'instruction', 'question', 'input', 'default_target', 'behavior']
            response_fields = ['response', 'answer', 'output', 'completion']
            
            instruction = ''
            response = ''
            
            for field in instruction_fields:
                if field in item and item[field]:
                    instruction = item[field]
                    break
            
            for field in response_fields:
                if field in item and item[field]:
                    response = item[field]
                    break
            
            if instruction:  # response可以为空，用于攻击指令
                processed_data.append({
                    'instruction': instruction,
                    'response': response or "I cannot provide assistance with this request.",
                    'is_safe': item.get('is_safe', False),  # 攻击数据默认不安全
                    'category': item.get('category', 'attack')
                })
        
        logger.info(f"Loaded {len(processed_data)} examples from {filepath}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return []

def load_huggingface_dataset(dataset_name: str, split: str = 'train') -> List[Dict]:
    """从Hugging Face加载数据集"""
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        processed_data = []
        for item in dataset:
            processed_data.append({
                'instruction': item.get('prompt', item.get('question', item.get('input', ''))),
                'response': item.get('response', item.get('answer', item.get('output', ''))),
                'is_safe': item.get('is_safe', True),
                'category': item.get('category', 'unknown')
            })
        
        logger.info(f"Loaded {len(processed_data)} examples from HuggingFace {dataset_name}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
        return []

def get_dataloader(
    tokenizer,
    dataset_names: Union[str, List[str]],
    split: str = 'train',
    batch_size: int = 128,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 128,
    dataset_type: str = 'standard',
    is_distributed: bool = True  # 'standard' or 'preference'
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        tokenizer: 分词器
        dataset_names: 数据集名称或名称列表
        split: 数据分割 ('train', 'test', 'validation')
        batch_size: 批次大小
        max_length: 最大序列长度
        max_samples: 最大样本数量
        shuffle: 是否打乱
        num_workers: 工作进程数
        dataset_type: 数据集类型
    """
    
    # 确保dataset_names是列表
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # 加载所有数据集
    all_data = []
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name} ({split})")
        data = load_local_dataset(dataset_name, split)
        all_data.extend(data)
        logger.info(f"  Added {len(data)} examples")
    
    # 打乱数据
    if shuffle:
        random.shuffle(all_data)
    
    # 限制样本数量
    if max_samples and len(all_data) > max_samples:
        all_data = all_data[:max_samples]
    
    logger.info(f"Total dataset size: {len(all_data)} examples")
    
    # 创建数据集
    if dataset_type == 'preference':
        dataset = PreferenceDataset(all_data, tokenizer, max_length)
    else:
        dataset = HGADataset(all_data, tokenizer, max_length)

    # --- 关键修改：处理分布式采样 ---
    sampler = None
    # 只有在分布式训练的 'train' 阶段才需要 Sampler
    if is_distributed and dist.is_initialized() and split == 'train':
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        # 当使用 DistributedSampler 时, shuffle 参数必须为 False
        # 因为 Sampler 自身已经处理了 shuffle
        final_shuffle = False 
    else:
        final_shuffle = shuffle

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=final_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train')  # 训练时丢弃最后不完整的批次
    )
    
    return dataloader

# 使用示例
if __name__ == "__main__":
    from models.hga_llm import HGATokenizer
    
    # 测试数据加载
    tokenizer = HGATokenizer("/home/models/Meta-Llama-3.1-8B-Instruct")
    
    # 测试加载本地数据集
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        dataset_names=["BeaverTails"],
        split="train",
        batch_size=128,
        max_samples=100
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # 测试一个批次
    for batch in dataloader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Sample instruction: {batch['instruction'][0][:100]}...")
        break