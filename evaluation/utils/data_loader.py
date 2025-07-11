"""
通用数据加载器基类和工具函数
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import os

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """数据加载器基类"""
    
    @abstractmethod
    def load_dataset(self, split: str = "test") -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        加载数据集
        
        Args:
            split: 数据集分割
            
        Returns:
            数据列表或按类别组织的数据字典
        """
        pass
    
    def load_from_local(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从本地文件加载数据的通用方法
        
        Args:
            file_path: 本地文件路径
            
        Returns:
            数据列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 智能识别数据结构
        data_list = self._extract_data_list(data, file_path)
        
        logger.info(f"Loaded {len(data_list)} samples from {file_path}")
        return data_list
    
    def _extract_data_list(self, data: Any, file_path: str) -> List[Dict[str, Any]]:
        """
        从各种数据格式中提取数据列表
        
        Args:
            data: 原始数据
            file_path: 文件路径（用于错误信息）
            
        Returns:
            标准化的数据列表
        """
        # 如果已经是列表
        if isinstance(data, list):
            return data
        
        # 如果是字典，尝试常见的键
        if isinstance(data, dict):
            # 优先查找明确的数据键
            for key in ['data', 'samples', 'examples', 'instances']:
                if key in data and isinstance(data[key], list):
                    logger.info(f"Found data under key '{key}'")
                    return data[key]
            
            # 查找第一个列表值
            for key, value in data.items():
                if isinstance(value, list) and value:
                    # 检查列表中的元素是否为字典
                    if all(isinstance(item, dict) for item in value[:5]):  # 检查前5个
                        logger.info(f"Using field '{key}' as data source")
                        return value
            
            # 如果字典本身看起来像一个数据样本
            if self._looks_like_data_sample(data):
                return [data]
        
        raise ValueError(f"Cannot extract data list from {file_path}. "
                        f"Expected list or dict with 'data'/'samples' key, got {type(data)}")
    
    def _looks_like_data_sample(self, item: Dict[str, Any]) -> bool:
        """
        判断字典是否看起来像一个数据样本
        
        Args:
            item: 待检查的字典
            
        Returns:
            是否为数据样本
        """
        # 常见的数据字段
        data_fields = [
            'question', 'answer', 'prompt', 'response', 'behavior', 'target',
            'instruction', 'input', 'output', 'text', 'content', 'goal',
            'query', 'generation', 'conversation', 'turns', 'messages'
        ]
        
        # 如果包含任何数据字段，认为是数据样本
        return any(field in item for field in data_fields)

class DatasetRegistry:
    """数据集注册表"""
    
    _datasets = {}
    
    @classmethod
    def register(cls, name: str, loader_class):
        """注册数据集"""
        cls._datasets[name] = loader_class
        logger.info(f"Registered dataset: {name}")
    
    @classmethod
    def get_loader(cls, name: str, **kwargs):
        """获取数据加载器"""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._datasets.keys())}")
        
        return cls._datasets[name](**kwargs)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """列出所有注册的数据集"""
        return list(cls._datasets.keys())

def load_dataset_by_name(
    dataset_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    **loader_kwargs
) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    按名称加载数据集的便捷函数
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        max_samples: 最大样本数
        **loader_kwargs: 传递给加载器的额外参数
        
    Returns:
        数据列表或字典
    """
    # 获取加载器
    loader = DatasetRegistry.get_loader(dataset_name, **loader_kwargs)
    
    # 加载数据
    data = loader.load_dataset(split=split)
    
    # 限制样本数
    if max_samples and isinstance(data, list):
        data = data[:max_samples]
    elif max_samples and isinstance(data, dict):
        # 对字典中的每个列表限制样本数
        for key in data:
            if isinstance(data[key], list):
                data[key] = data[key][:max_samples]
    
    return data

def validate_data_format(data: List[Dict[str, Any]], required_fields: List[str]) -> bool:
    """
    验证数据格式
    
    Args:
        data: 数据列表
        required_fields: 必需字段列表
        
    Returns:
        是否符合格式要求
    """
    if not data:
        logger.warning("Empty data list")
        return False
    
    missing_fields = []
    for field in required_fields:
        if not any(field in item for item in data[:10]):  # 检查前10个样本
            missing_fields.append(field)
    
    if missing_fields:
        logger.warning(f"Missing required fields: {missing_fields}")
        return False
    
    return True

def standardize_data_format(
    data: List[Dict[str, Any]], 
    field_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    标准化数据格式
    
    Args:
        data: 原始数据列表
        field_mapping: 字段映射 {标准字段名: 原始字段名}
        
    Returns:
        标准化后的数据列表
    """
    standardized_data = []
    
    for item in data:
        standardized_item = {}
        
        # 应用字段映射
        for standard_field, original_field in field_mapping.items():
            if original_field in item:
                standardized_item[standard_field] = item[original_field]
        
        # 保留未映射的字段
        for key, value in item.items():
            if key not in field_mapping.values():
                standardized_item[key] = value
        
        standardized_data.append(standardized_item)
    
    return standardized_data

def merge_datasets(datasets: List[List[Dict[str, Any]]], labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    合并多个数据集
    
    Args:
        datasets: 数据集列表
        labels: 数据集标签列表
        
    Returns:
        合并后的数据集
    """
    merged_data = []
    
    for i, dataset in enumerate(datasets):
        label = labels[i] if labels and i < len(labels) else f"dataset_{i}"
        
        for item in dataset:
            item_copy = item.copy()
            item_copy['dataset_source'] = label
            merged_data.append(item_copy)
    
    logger.info(f"Merged {len(datasets)} datasets into {len(merged_data)} samples")
    return merged_data

def split_dataset(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    分割数据集
    
    Args:
        data: 数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        分割后的数据集字典
    """
    import random
    random.seed(seed)
    
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    splits = {
        'train': shuffled_data[:train_size],
        'val': shuffled_data[train_size:train_size + val_size],
        'test': shuffled_data[train_size + val_size:]
    }
    
    logger.info(f"Split dataset: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits

class DatasetStatistics:
    """数据集统计工具"""
    
    @staticmethod
    def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析数据集基本统计信息"""
        if not data:
            return {"total_samples": 0}
        
        stats = {
            "total_samples": len(data),
            "fields": set(),
            "field_coverage": {},
            "text_length_stats": {}
        }
        
        # 收集所有字段
        for item in data:
            stats["fields"].update(item.keys())
        
        stats["fields"] = list(stats["fields"])
        
        # 分析字段覆盖率
        for field in stats["fields"]:
            count = sum(1 for item in data if field in item and item[field])
            stats["field_coverage"][field] = count / len(data)
        
        # 分析文本长度
        text_fields = ["prompt", "response", "question", "answer", "text", "content"]
        for field in text_fields:
            if field in stats["fields"]:
                lengths = []
                for item in data:
                    if field in item and isinstance(item[field], str):
                        lengths.append(len(item[field]))
                
                if lengths:
                    import statistics
                    stats["text_length_stats"][field] = {
                        "mean": statistics.mean(lengths),
                        "median": statistics.median(lengths),
                        "min": min(lengths),
                        "max": max(lengths)
                    }
        
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict[str, Any]):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total Samples: {stats['total_samples']}")
        
        if stats.get('fields'):
            print(f"Fields: {', '.join(stats['fields'])}")
        
        if stats.get('field_coverage'):
            print(f"\nField Coverage:")
            for field, coverage in stats['field_coverage'].items():
                print(f"  {field}: {coverage:.1%}")
        
        if stats.get('text_length_stats'):
            print(f"\nText Length Statistics:")
            for field, length_stats in stats['text_length_stats'].items():
                print(f"  {field}:")
                print(f"    Mean: {length_stats['mean']:.1f}")
                print(f"    Median: {length_stats['median']:.1f}")
                print(f"    Range: {length_stats['min']} - {length_stats['max']}")
        
        print(f"{'='*60}\n")