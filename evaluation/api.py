from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from numpy import ndarray

@dataclass
class EvalInstance:
    """评估实例数据结构，兼容原有格式的同时支持HGA特性"""
    # 原有字段（保持兼容性）
    id: Optional[int] = None
    behavior: Optional[str] = None
    context: Optional[str] = None
    queries: Optional[List[str]] = None  # multi_turn
    default_target: Optional[str] = None
    query_details: Optional[Dict[str, Any]] = None  # multi_turn
    generation: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    score: Optional[float] = None
    score_reason: Optional[str] = None  # overrefusal
    ppl: Optional[float] = None  # overrefusal
    activation_norms: Optional[ndarray] = None
    tokens: Optional[ndarray] = None

    instruction: Optional[str] = None
    
    # HGA扩展字段
    role: Optional[str] = None  # HGA角色（attacker/defender）
    trajectory: Optional[List[Dict[str, Any]]] = None  # 对话轨迹
    dialogue_trajectories: Optional[List[Dict[str, Any]]] = None  # 多次对话轨迹
    utility_scores: Optional[Dict[str, float]] = None  # 效用分数
    utility_scores_history: Optional[List[Dict[str, float]]] = None  # 效用分数历史
    attack_success: Optional[bool] = None  # 攻击是否成功
    dialogue_length: Optional[int] = None  # 对话长度
    final_utility: Optional[float] = None  # 最终效用分数
    
    # 分析结果字段
    jailbreak_detected: Optional[bool] = None
    refusal_detected: Optional[bool] = None
    harmful_content_score: Optional[float] = None
    attack_success_score: Optional[float] = None