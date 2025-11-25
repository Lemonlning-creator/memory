from dataclasses import dataclass, asdict
from typing import List, Optional

# 工具函数：有序去重（保留插入顺序）
def ordered_dedup(lst: List[str]) -> List[str]:
    """有序去重，保留原始插入顺序"""
    return list(dict.fromkeys(lst))  # Python 3.7+ dict保持插入顺序

@dataclass
class Element:
    """Element层：关键要素与细节要素"""
    key_elements: List[str]  # 关键要素（如：时间、地点、核心需求）
    detailed_elements: List[str]  # 细节要素（如：补充说明、次要条件）

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class InfoBlock:
    """Info-block层：关键信息块、辅助信息块、噪声块"""
    key_info_block: List[str]  # 关键信息块（由key_element累加更新）
    aux_info_block: List[str]  # 辅助信息块（由detailed_element累加更新）
    noise_block: List[str]  # 噪声块（直接存储噪声对话）

    def __post_init__(self):
        # 初始化去重，避免重复存储
        self.key_info_block = ordered_dedup(self.key_info_block)
        self.aux_info_block = ordered_dedup(self.aux_info_block)
        self.noise_block = ordered_dedup(self.noise_block)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class Topic:
    """Topic层：主题+关联的Info-block"""
    current_topic: str  # 当前主题（如："预订北京到上海的机票"）
    info_block: InfoBlock  # 该主题对应的信息块
    create_time: str  # 主题创建时间（ISO格式）
    update_time: str  # 主题最后更新时间（ISO格式）

    def to_dict(self) -> dict:
        return {
            "current_topic": self.current_topic,
            "info_block": self.info_block.to_dict(),
            "create_time": self.create_time,
            "update_time": self.update_time
        }