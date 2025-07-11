"""
统一日志管理器 - 支持SwanLab、TensorBoard和WandB
"""
import os
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class LogManager:
    """统一的日志管理器，支持多种日志后端"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        self.config = config
        self.logging_config = config.get('logging', {})
        
        # 设置实验名称
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = self.logging_config.get('experiment_name', 'hga-experiment')
        
        # 设置日志目录
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = self.logging_config.get('log_dir', './logs')
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化各种日志后端
        self.swanlab = None
        self.tensorboard_writer = None
        self.wandb = None
        
        self._init_loggers()
        
        logger.info(f"LogManager initialized with experiment: {self.experiment_name}")
    
    def _init_loggers(self):
        """初始化日志后端"""
        
        # 初始化SwanLab
        if self.logging_config.get('use_swanlab', False):
            try:
                import swanlab
                
                swanlab.init(
                    project=self.logging_config.get('project_name', 'hga-alignment'),
                    experiment_name=self.experiment_name,
                    description="HGA (Hierarchical Game-Theoretic Alignment) Training",
                    config=self.config,
                    logdir=os.path.join(self.log_dir, 'swanlab')
                )
                
                self.swanlab = swanlab
                logger.info("SwanLab initialized successfully")
                
            except ImportError:
                logger.warning("SwanLab not installed, skipping SwanLab logging")
            except Exception as e:
                logger.warning(f"Failed to initialize SwanLab: {e}")
        
        # 初始化TensorBoard
        if self.logging_config.get('use_tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                tb_log_dir = os.path.join(self.log_dir, 'tensorboard', self.experiment_name)
                self.tensorboard_writer = SummaryWriter(log_dir=tb_log_dir)
                
                # 记录配置信息
                config_text = self._format_config_for_tensorboard(self.config)
                self.tensorboard_writer.add_text('config', config_text, 0)
                
                logger.info(f"TensorBoard initialized, log dir: {tb_log_dir}")
                
            except ImportError:
                logger.warning("TensorBoard not installed, skipping TensorBoard logging")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
        
        # 初始化WandB（可选支持）
        if self.logging_config.get('use_wandb', False):
            try:
                import wandb
                
                wandb.init(
                    project=self.logging_config.get('project_name', 'hga-alignment'),
                    name=self.experiment_name,
                    config=self.config,
                    dir=os.path.join(self.log_dir, 'wandb')
                )
                
                self.wandb = wandb
                logger.info("WandB initialized successfully")
                
            except ImportError:
                logger.warning("WandB not installed, skipping WandB logging")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
    
    def _format_config_for_tensorboard(self, config: Dict[str, Any], prefix: str = "") -> str:
        """格式化配置信息用于TensorBoard显示"""
        lines = []
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                lines.append(f"**{full_key}:**")
                lines.append(self._format_config_for_tensorboard(value, full_key))
            else:
                lines.append(f"- {full_key}: {value}")
        return "\n".join(lines)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int, prefix: str = ""):
        """记录指标"""
        # 添加前缀
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # SwanLab日志
        if self.swanlab:
            try:
                self.swanlab.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to SwanLab: {e}")
        
        # TensorBoard日志
        if self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")
        
        # WandB日志
        if self.wandb:
            try:
                log_data = {**metrics, 'step': step}
                self.wandb.log(log_data)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
    
    def log_text(self, tag: str, text: str, step: int):
        """记录文本信息"""
        # SwanLab文本日志
        if self.swanlab:
            try:
                self.swanlab.log({f"text/{tag}": text}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log text to SwanLab: {e}")
        
        # TensorBoard文本日志
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_text(tag, text, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log text to TensorBoard: {e}")
        
        # WandB文本日志
        if self.wandb:
            try:
                self.wandb.log({f"text/{tag}": self.wandb.Html(text)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log text to WandB: {e}")
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        import torch
        import numpy as np
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # SwanLab直方图
        if self.swanlab:
            try:
                self.swanlab.log({f"histogram/{tag}": values}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log histogram to SwanLab: {e}")
        
        # TensorBoard直方图
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_histogram(tag, values, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log histogram to TensorBoard: {e}")
        
        # WandB直方图
        if self.wandb:
            try:
                self.wandb.log({f"histogram/{tag}": self.wandb.Histogram(values)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log histogram to WandB: {e}")
    
    def log_image(self, tag: str, image, step: int):
        """记录图像"""
        # TensorBoard图像
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_image(tag, image, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log image to TensorBoard: {e}")
        
        # SwanLab图像
        if self.swanlab:
            try:
                self.swanlab.log({f"image/{tag}": image}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log image to SwanLab: {e}")
        
        # WandB图像
        if self.wandb:
            try:
                self.wandb.log({f"image/{tag}": self.wandb.Image(image)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log image to WandB: {e}")
    
    def log_table(self, tag: str, data: Dict[str, list], step: int):
        """记录表格数据"""
        # SwanLab表格
        if self.swanlab:
            try:
                self.swanlab.log({f"table/{tag}": data}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log table to SwanLab: {e}")
        
        # WandB表格
        if self.wandb:
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                self.wandb.log({f"table/{tag}": self.wandb.Table(dataframe=df)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log table to WandB: {e}")
    
    def log_learning_rate(self, lr: float, step: int):
        """记录学习率"""
        self.log_metrics({"learning_rate": lr}, step, "training")
    
    def log_gradients(self, model, step: int):
        """记录梯度信息"""
        import torch
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录每个参数的梯度范数
                self.log_metrics({f"grad_norm/{name}": param_norm.item()}, step, "gradients")
        
        total_norm = total_norm ** (1. / 2)
        
        # 记录总梯度范数
        self.log_metrics({
            "grad_norm/total": total_norm,
            "grad_norm/param_count": param_count
        }, step, "gradients")
    
    def save_model_info(self, model, step: int):
        """保存模型信息"""
        import torch
        
        # 计算模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        }
        
        self.log_metrics(model_info, step, "model")
        
        # 记录模型结构（仅在第一步）
        if step == 0:
            model_summary = str(model)
            self.log_text("model_architecture", model_summary, step)
    
    def finish(self):
        """结束日志记录"""
        # 关闭SwanLab
        if self.swanlab:
            try:
                self.swanlab.finish()
                logger.info("SwanLab session finished")
            except Exception as e:
                logger.warning(f"Error finishing SwanLab: {e}")
        
        # 关闭TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Error closing TensorBoard: {e}")
        
        # 关闭WandB
        if self.wandb:
            try:
                self.wandb.finish()
                logger.info("WandB session finished")
            except Exception as e:
                logger.warning(f"Error finishing WandB: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

# 便捷函数
def create_logger(config: Dict[str, Any], experiment_name: Optional[str] = None) -> LogManager:
    """创建日志管理器的便捷函数"""
    return LogManager(config, experiment_name)