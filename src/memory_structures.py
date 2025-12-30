from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Memory:
    """记忆结构：topic-content-keywords三层结构"""
    topic: str  # 主题
    content: str  # 内容总结
    keywords: list[str]  # 关键词列表
    create_time: str  # 创建时间（ISO格式）
    update_time: str  # 更新时间（ISO格式）

    def __post_init__(self):
        # 确保关键词去重
        self.keywords = list(dict.fromkeys(self.keywords))
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @staticmethod
    def get_current_time() -> str:
        """获取ISO格式当前时间"""
        return datetime.now().isoformat()