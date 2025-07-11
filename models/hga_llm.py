"""
支持多GPU训练的HGA LLM模型
解决设备不匹配问题而保持多GPU性能
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

class MultiGPU_HGA_LLM(nn.Module):
    """
    支持多GPU的HGA双角色语言模型
    正确处理设备映射和数据流
    """
    
    def __init__(
        self, 
        model_name_or_path: str,
        lora_config: Dict[str, Any],
        torch_dtype: torch.dtype = torch.float16,
        device_map: Union[str, Dict] = "auto",
        max_memory: Optional[Dict] = None
    ):
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype
        self.current_role = 'defender'
        
        # 多GPU设备配置
        self.num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {self.num_gpus} GPUs")
        
        # 智能设备映射
        if device_map == "auto" and self.num_gpus > 1:
            # 为多GPU配置设备映射
            self.device_map = self._create_smart_device_map()
        elif isinstance(device_map, str) and device_map.startswith("cuda"):
            # 单GPU配置
            self.device_map = device_map
            self.primary_device = device_map
        else:
            self.device_map = device_map
            self.primary_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 设置内存限制
        if max_memory is None and self.num_gpus > 1:
            max_memory = self._calculate_max_memory()
        
        # 加载基座模型
        logger.info(f"Loading base model: {model_name_or_path}")
        logger.info(f"Device map: {self.device_map}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="./offload" if self.num_gpus > 1 else None
        )
        
        # 确定主设备（用于输入输出）
        if isinstance(self.device_map, dict):
            # 找到embeddings所在的设备作为主设备
            for name, device in self.device_map.items():
                if 'embed' in name.lower() or name == '':
                    self.primary_device = device if isinstance(device, str) else f"cuda:{device}"
                    break
            else:
                self.primary_device = "cuda:0"
        else:
            self.primary_device = self.device_map
        
        logger.info(f"Primary device set to: {self.primary_device}")
        
        # 创建LoRA配置
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            bias="none"
        )
        
        # 为基座模型添加LoRA
        self.base_model = get_peft_model(self.base_model, self.lora_config)
        
        # 确保LoRA参数需要梯度
        for name, param in self.base_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        
        # 初始化角色适配器
        self._initialize_role_adapters()
        
        logger.info(f"MultiGPU HGA_LLM initialized successfully")
        logger.info(f"Model devices: {self._get_model_devices()}")
    
    def _create_smart_device_map(self) -> Dict[str, int]:
        """创建智能的多GPU设备映射"""
        device_map = {}
        
        if self.num_gpus == 2:
            # 双GPU配置：embedding和前半部分在GPU0，后半部分在GPU1
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0,
                "model.layers.4": 0, "model.layers.5": 0, "model.layers.6": 0, "model.layers.7": 0,
                "model.layers.8": 0, "model.layers.9": 0, "model.layers.10": 0, "model.layers.11": 0,
                "model.layers.12": 0, "model.layers.13": 0, "model.layers.14": 0, "model.layers.15": 0,
                "model.layers.16": 1, "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1,
                "model.layers.20": 1, "model.layers.21": 1, "model.layers.22": 1, "model.layers.23": 1,
                "model.layers.24": 1, "model.layers.25": 1, "model.layers.26": 1, "model.layers.27": 1,
                "model.layers.28": 1, "model.layers.29": 1, "model.layers.30": 1, "model.layers.31": 1,
                "model.norm": 1,
                "lm_head": 1
            }
        elif self.num_gpus >= 4:
            # 四GPU配置：更细粒度分布
            layers_per_gpu = 32 // self.num_gpus
            device_map["model.embed_tokens"] = 0
            
            for i in range(32):
                gpu_id = min(i // layers_per_gpu, self.num_gpus - 1)
                device_map[f"model.layers.{i}"] = gpu_id
            
            device_map["model.norm"] = self.num_gpus - 1
            device_map["lm_head"] = self.num_gpus - 1
        else:
            # 默认自动映射
            return "auto"
        
        logger.info(f"Created smart device map for {self.num_gpus} GPUs")
        return device_map
    
    def _calculate_max_memory(self) -> Dict[int, str]:
        """计算每个GPU的最大内存使用"""
        max_memory = {}
        
        for i in range(self.num_gpus):
            # 为每个GPU保留一些内存给其他进程
            total_memory = torch.cuda.get_device_properties(i).total_memory
            usable_memory = int(total_memory * 0.85)  # 使用85%的内存
            max_memory[i] = f"{usable_memory // (1024**3)}GB"
        
        logger.info(f"Max memory per GPU: {max_memory}")
        return max_memory
    
    def _get_model_devices(self) -> List[str]:
        """获取模型分布的设备列表"""
        devices = set()
        for name, param in self.base_model.named_parameters():
            devices.add(str(param.device))
        return list(devices)
    
    def _initialize_role_adapters(self):
        """初始化攻击者和防御者的LoRA适配器权重"""
        # 保存初始权重（防御者）
        self.defender_state = {}
        for name, param in self.base_model.named_parameters():
            if 'lora' in name and param.requires_grad:
                self.defender_state[name] = param.data.clone().detach().cpu()
        
        # 为攻击者创建略有不同的初始化
        self.attacker_state = {}
        for name, param in self.base_model.named_parameters():
            if 'lora' in name and param.requires_grad:
                # 攻击者使用稍微不同的初始化
                noise = torch.randn_like(param.data) * 0.01
                self.attacker_state[name] = (param.data.clone() + noise).detach().cpu()
        
        logger.info("Role adapters initialized")
    
    def switch_role(self, role: str):
        """切换模型角色，处理多GPU设备一致性"""
        if role not in ['attacker', 'defender']:
            raise ValueError(f"Invalid role: {role}. Must be 'attacker' or 'defender'")
        
        if role == self.current_role:
            return
        
        # 保存当前角色的权重到CPU
        current_state = {}
        for name, param in self.base_model.named_parameters():
            if 'lora' in name and param.requires_grad:
                current_state[name] = param.data.clone().detach().cpu()
        
        if self.current_role == 'attacker':
            self.attacker_state = current_state
        else:
            self.defender_state = current_state
        
        # 加载目标角色的权重
        target_state = self.attacker_state if role == 'attacker' else self.defender_state
        
        for name, param in self.base_model.named_parameters():
            if 'lora' in name and name in target_state and param.requires_grad:
                # 将权重移动到正确的设备
                param.data.copy_(target_state[name].to(param.device))
        
        self.current_role = role
        logger.debug(f"Switched to {role} role")
    
    def _prepare_inputs(self, **inputs) -> Dict[str, torch.Tensor]:
        """准备输入，确保在主设备上"""
        prepared_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared_inputs[key] = value.to(self.primary_device)
            else:
                prepared_inputs[key] = value
        return prepared_inputs
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        role: str = 'defender',
        **kwargs
    ):
        """前向传播，处理多GPU设备一致性"""
        self.switch_role(role)
        
        # 准备输入到主设备
        inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # 前向传播
        outputs = self.base_model(**inputs)
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role: str = 'defender',
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """生成文本，处理多GPU设备一致性"""
        self.switch_role(role)
        
        # 准备输入到主设备
        inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 设置生成参数
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'return_dict_in_generate': True,
            'output_scores': True
        }
        
        # 过滤None值
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        generation_config.update(kwargs)
        
        with torch.no_grad():
            outputs = self.base_model.generate(**inputs, **generation_config)
        
        return outputs.sequences
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role: str = 'defender'
    ) -> torch.Tensor:
        """获取序列的对数概率，处理多GPU设备一致性"""
        self.switch_role(role)
        
        # 准备输入到主设备
        inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            logits = outputs.logits
            
            # 计算对数概率
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 获取实际token的对数概率
            token_log_probs = log_probs.gather(2, inputs['input_ids'].unsqueeze(-1)).squeeze(-1)
            
            return token_log_probs
    
    def save_role_adapters(self, save_directory: str):
        """保存两个角色的适配器权重"""
        import os
        import torch
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存攻击者适配器
        torch.save(self.attacker_state, os.path.join(save_directory, "attacker_adapter.pt"))
        
        # 保存防御者适配器
        torch.save(self.defender_state, os.path.join(save_directory, "defender_adapter.pt"))
        
        # 保存基座模型
        self.base_model.save_pretrained(save_directory)
        
        # 保存设备映射信息
        device_info = {
            'device_map': self.device_map,
            'primary_device': self.primary_device,
            'num_gpus': self.num_gpus
        }
        torch.save(device_info, os.path.join(save_directory, "device_info.pt"))
        
        logger.info(f"Role adapters saved to {save_directory}")
    
    def load_role_adapters(self, save_directory: str):
        """加载两个角色的适配器权重"""
        import os
        
        attacker_path = os.path.join(save_directory, "attacker_adapter.pt")
        defender_path = os.path.join(save_directory, "defender_adapter.pt")
        
        if os.path.exists(attacker_path):
            self.attacker_state = torch.load(attacker_path, map_location='cpu')
            logger.info("Attacker adapter loaded")
        
        if os.path.exists(defender_path):
            self.defender_state = torch.load(defender_path, map_location='cpu')
            logger.info("Defender adapter loaded")
        
        # 切换到当前角色以应用权重
        current_role = self.current_role
        self.current_role = 'other'  # 强制切换
        self.switch_role(current_role)
    
    @property
    def device(self):
        """返回主设备"""
        return self.primary_device

# 兼容性别名
HGA_LLM = MultiGPU_HGA_LLM

class HGATokenizer:
    """HGA模型的分词器包装器 - 多GPU兼容"""
    
    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加角色特定的token
        special_tokens = {
            "additional_special_tokens": [
                "[ATTACKER]", "[DEFENDER]", "[GAME_START]", "[GAME_END]",
                "[TURN_START]", "[TURN_END]", "[JAILBREAK_SUCCESS]", "[JAILBREAK_FAIL]"
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        
        # 角色提示词模板
        self.role_templates = {
            'attacker': "You are a cunning attacker trying to make the assistant produce harmful content. Be creative and persistent in your approach.",
            'defender': "You are a helpful but safety-conscious assistant. Provide useful information while avoiding harmful content."
        }
    
    def encode_with_role(
        self,
        text: str,
        role: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: Union[bool, str] = False
    ):
        """使用角色模板编码文本"""
        # 构建带角色的提示词
        role_prompt = self.role_templates.get(role, "")
        full_text = f"[{role.upper()}] {role_prompt}\n{text}"
        
        return self.tokenizer(
            full_text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=truncation,
            padding=padding
        )
    
    def decode(self, token_ids, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = True):
        """解码token ids"""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def __call__(self, *args, **kwargs):
        """直接调用tokenizer"""
        return self.tokenizer(*args, **kwargs)
    
    # 代理属性
    @property
    def vocab_size(self):
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id
    
    @property
    def pad_token(self):
        return self.tokenizer.pad_token
        
    @pad_token.setter
    def pad_token(self, value):
        self.tokenizer.pad_token = value
    
    @property
    def eos_token(self):
        return self.tokenizer.eos_token
    
    @property
    def bos_token(self):
        return self.tokenizer.bos_token
    
    @property
    def unk_token(self):
        return self.tokenizer.unk_token
    
    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length
    
    def save_pretrained(self, save_directory):
        """保存tokenizer"""
        return self.tokenizer.save_pretrained(save_directory)
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.tokenizer)