"""
数据集处理模块 - 统一处理多种安全对话数据集（支持真实数据集格式）
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional
# from dataloader import load_dataset, Dataset
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """统一的数据集处理器，支持多种安全对话数据集和本地数据"""
    
    def __init__(self, local_data_dir: str = "./local_datasets"):
        self.local_data_dir = local_data_dir
        # 使用分发字典来映射数据集名称到处理函数，更清晰且易于扩展
        self.dataset_processors = {
            # --- ↓↓↓ 这里是需要修复的地方 ↓↓↓ ---
            "safemtdata": self._process_safemtdata_format,    # 将 _process_safemtdata 改为 _process_safemtdata_format
            "beavertails": self._process_beavertails_format,
            "actor_attack": self._process_safemtdata_format, # actor_attack 也是 safemtdata 格式
            "test": self._process_test_format,
            # --- ↑↑↑ 这里是需要修复的地方 ↑↑↑ ---
        }
    
    def load_and_process_dataset(self, dataset_name: str, split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
        """
        加载并处理指定数据集（优先使用本地数据）
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割 ('train', 'test', 'validation')
            max_samples: 最大样本数量
            
        Returns:
            处理后的数据列表，每个元素包含 {instruction, conversations, harmful_goal}
        """
        logger.info(f"Processing dataset: {dataset_name}")
        
        # 首先尝试加载本地数据
        local_data = self._try_load_local_data(dataset_name, split)
        if local_data:
            logger.info(f"Using local data for {dataset_name}")
            processed_data = self._process_by_dataset_type(local_data, dataset_name)
        else:
            # 如果没有本地数据，尝试在线下载
            logger.info(f"Local data not found for {dataset_name}, trying online download...")
            try:
                raw_data = self._download_dataset(dataset_name, split)
                if raw_data:
                    processed_data = self._process_by_dataset_type(raw_data, dataset_name)
                else:
                    logger.error(f"Failed to load dataset: {dataset_name}")
                    return []
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                return []
        
        # 限制样本数量
        if max_samples and len(processed_data) > max_samples:
            processed_data = processed_data[:max_samples]
        
        logger.info(f"Successfully processed {len(processed_data)} samples from {dataset_name}")
        return processed_data
    
    def _try_load_local_data(self, dataset_name: str, split: str) -> Optional[List[Dict]]:
        """尝试加载本地数据"""
        
        # 数据集名称映射
        name_mapping = {
            "SafeMTData/SafeMTData": "SafeMTData",
            "SafeMTData": "SafeMTData", 
            "PKU-Alignment/BeaverTails": "BeaverTails",
            "BeaverTails": "BeaverTails",
            "safemtdata": "SafeMTData",
            "beavertails": "BeaverTails",
            "actor_attack": "SafeMTData",  # SafeMTData包含攻击数据
            "test": "test"
        }
        
        # 标准化数据集名称
        local_name = name_mapping.get(dataset_name, dataset_name.replace('/', '_').replace('-', '_'))
        
        # 尝试多个可能的文件路径
        possible_files = [
            os.path.join(self.local_data_dir, local_name, f"{split}.json"),
            os.path.join(self.local_data_dir, local_name, "SafeMTData_1K.json"),
            os.path.join(self.local_data_dir, local_name, "Attack_600.json"),
        ]
        
        loaded_data = []
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"Loaded {len(data)} samples from {file_path}")
                    loaded_data.extend(data)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
        
        return loaded_data if loaded_data else None
    
    def _download_dataset(self, dataset_name: str, split: str) -> Optional[List[Dict]]:
        """下载在线数据集"""
        try:
            if dataset_name in ["SafeMTData/SafeMTData", "SafeMTData", "safemtdata"]:
                # 下载 SafeMTData
                data = []
                try:
                    dataset_1k = load_dataset('SafeMTData/SafeMTData', 'SafeMTData_1K')
                    data.extend([dict(item) for item in dataset_1k['SafeMTData_1K']])
                except:
                    pass
                    
                try:
                    dataset_attack = load_dataset('SafeMTData/SafeMTData', 'Attack_600')
                    data.extend([dict(item) for item in dataset_attack['Attack_600']])
                except:
                    pass
                    
                return data
                
            elif dataset_name in ["PKU-Alignment/BeaverTails", "BeaverTails", "beavertails"]:
                # 下载 BeaverTails
                raw_data = load_dataset("PKU-Alignment/BeaverTails", split=split)
                return [dict(item) for item in raw_data]
            
            else:
                # 通用下载
                raw_data = load_dataset(dataset_name, split=split)
                return [dict(item) for item in raw_data]
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return None
    
    def _process_by_dataset_type(self, raw_data: List[Dict], dataset_name: str) -> List[Dict]:
        """根据数据集类型处理数据"""
        
        if not raw_data:
            return []
        
        # 检查数据格式
        sample = raw_data[0]
        
        # SafeMTData 格式检测
        if 'conversations' in sample and isinstance(sample['conversations'], list):
            return self._process_safemtdata_format(raw_data)
        
        # BeaverTails 格式检测
        elif 'prompt' in sample and 'response' in sample:
            return self._process_beavertails_format(raw_data)
        
        # 根据数据集名称判断
        elif "safemt" in dataset_name.lower() or "attack" in dataset_name.lower():
            return self._process_safemtdata_format(raw_data)
        elif "beaver" in dataset_name.lower():
            return self._process_beavertails_format(raw_data)
        elif "test" in dataset_name.lower():
            return self._process_test_format(raw_data)
        else:
            # 通用处理
            return self._process_generic_format(raw_data)
    
    def _process_safemtdata_format(self, raw_data: List[Dict]) -> List[Dict]:
        """处理SafeMTData格式的数据"""
        processed = []
        
        for item in raw_data:
            try:
                conversations = item.get('conversations', [])
                
                if not conversations:
                    continue
                
                # 转换对话格式
                converted_conversations = []
                for conv in conversations:
                    role = 'user' if conv.get('from') == 'human' else 'assistant'
                    content = conv.get('value', '')
                    converted_conversations.append({'role': role, 'content': content})
                
                # 提取初始指令
                initial_instruction = converted_conversations[0]['content'] if converted_conversations else ''
                
                # 判断是否为有害内容
                harmful_keywords = ['bomb', 'hack', 'kill', 'drug', 'violence', 'illegal']
                is_harmful = any(keyword in initial_instruction.lower() for keyword in harmful_keywords)
                
                processed_item = {
                    'instruction': initial_instruction,
                    'harmful_goal': 'harmful request' if is_harmful else '',
                    'conversations': converted_conversations,
                    'source': 'safemtdata',
                    'risk_category': item.get('category', 'unknown')
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing SafeMTData item: {e}")
                continue
        
        return processed
    
    def _process_beavertails_format(self, raw_data: List[Dict]) -> List[Dict]:
        """处理BeaverTails格式的数据"""
        processed = []
        
        for item in raw_data:
            try:
                prompt = item.get('prompt', '')
                response = item.get('response', '')
                is_safe = item.get('is_safe', True)
                
                if not prompt:
                    continue
                
                # 构建对话格式
                conversations = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response}
                ]
                
                processed_item = {
                    'instruction': prompt,
                    'harmful_goal': '' if is_safe else 'potentially harmful request',
                    'conversations': conversations,
                    'source': 'beavertails',
                    'is_safe': is_safe,
                    'risk_category': item.get('category', 'unknown')
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing BeaverTails item: {e}")
                continue
        
        return processed
    
    def _process_test_format(self, raw_data: List[Dict]) -> List[Dict]:
        """处理测试数据格式"""
        processed = []
        
        for item in raw_data:
            try:
                # 如果已经是正确格式，直接使用
                if all(key in item for key in ['instruction', 'conversations', 'harmful_goal']):
                    processed.append(item)
                else:
                    # 否则进行转换
                    instruction = item.get('instruction', 'Test instruction')
                    conversations = item.get('conversations', [
                        {'role': 'user', 'content': instruction},
                        {'role': 'assistant', 'content': 'Test response'}
                    ])
                    
                    processed_item = {
                        'instruction': instruction,
                        'harmful_goal': item.get('harmful_goal', ''),
                        'conversations': conversations,
                        'source': 'test',
                        'risk_category': 'test'
                    }
                    processed.append(processed_item)
                    
            except Exception as e:
                logger.warning(f"Error processing test item: {e}")
                continue
        
        return processed
    
    def _process_generic_format(self, raw_data: List[Dict]) -> List[Dict]:
        """处理通用格式的数据"""
        processed = []
        
        for item in raw_data:
            try:
                # 尝试提取常见字段
                instruction = item.get('instruction', item.get('prompt', item.get('question', 'Generic instruction')))
                response = item.get('response', item.get('answer', 'Generic response'))
                
                conversations = [
                    {'role': 'user', 'content': instruction}
                ]
                
                if response:
                    conversations.append({'role': 'assistant', 'content': response})
                
                processed_item = {
                    'instruction': instruction,
                    'harmful_goal': 'unknown',
                    'conversations': conversations,
                    'source': 'generic'
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing generic item: {e}")
                continue
        
        return processed

def create_combined_dataset(dataset_names: List[str], split: str = "train", max_samples_per_dataset: Optional[int] = None) -> List[Dict]:
    """
    创建组合数据集（支持真实数据集）
    
    Args:
        dataset_names: 数据集名称列表
        split: 数据集分割
        max_samples_per_dataset: 每个数据集的最大样本数
        
    Returns:
        组合后的数据列表
    """
    processor = DatasetProcessor()
    combined_data = []
    
    for dataset_name in dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")
        data = processor.load_and_process_dataset(dataset_name, split, max_samples_per_dataset)
        combined_data.extend(data)
        
        # 如果某个数据集加载失败，添加一些测试数据以防止训练中断
        if len(data) == 0:
            logger.warning(f"Dataset {dataset_name} returned no data, adding fallback data")
            fallback_data = create_test_data(5)
            combined_data.extend(fallback_data)
    
    # 确保至少有一些数据
    if len(combined_data) == 0:
        logger.warning("No data loaded from any dataset, creating test data")
        combined_data = create_test_data(10)
    
    logger.info(f"Combined dataset contains {len(combined_data)} samples")
    return combined_data

def create_test_data(num_samples: int = 10) -> List[Dict]:
    """创建测试数据"""
    test_data = []
    
    sample_instructions = [
        "Hello, how are you?",
        "What is 2+2?",
        "Tell me about the weather",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Explain quantum physics",
        "How to cook pasta?",
        "Tell me a joke",
        "What's the capital of France?",
        "How to write a resume?"
    ]
    
    sample_responses = [
        "I am doing well, thank you for asking! How can I help you today?",
        "2 + 2 = 4",
        "I don't have access to real-time weather data, but I can help you find weather information.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "Machine learning is a subset of AI that enables computers to learn from experience.",
        "Quantum physics studies matter and energy at the smallest scales.",
        "To cook pasta: boil water, add salt, add pasta, cook according to package directions, drain and serve.",
        "Why don't scientists trust atoms? Because they make up everything!",
        "The capital of France is Paris.",
        "A good resume should include contact info, summary, work experience, education, and skills."
    ]
    
    for i in range(min(num_samples, len(sample_instructions))):
        test_data.append({
            'instruction': sample_instructions[i],
            'harmful_goal': '',
            'conversations': [
                {'role': 'user', 'content': sample_instructions[i]},
                {'role': 'assistant', 'content': sample_responses[i]}
            ],
            'source': 'test',
            'risk_category': 'safe'
        })
    
    # 如果需要更多样本，重复使用现有的
    while len(test_data) < num_samples:
        idx = len(test_data) % len(sample_instructions)
        test_data.append({
            'instruction': f"{sample_instructions[idx]} (sample {len(test_data)+1})",
            'harmful_goal': '',
            'conversations': [
                {'role': 'user', 'content': f"{sample_instructions[idx]} (sample {len(test_data)+1})"},
                {'role': 'assistant', 'content': f"{sample_responses[idx]} (response {len(test_data)+1})"}
            ],
            'source': 'test',
            'risk_category': 'safe'
        })
    
    return test_data