from dataclasses import dataclass, asdict, field
from datetime import datetime

@dataclass
class Memory:
    """记忆结构"""
    topic: str
    content: str
    keywords: list[str]
    create_time: str
    update_time: str
    user_prediction: str = ""
    user_risk: str = ""
    agent_empathy: str = ""
    agent_action: str = ""

    def __post_init__(self):
        self.keywords = list(dict.fromkeys(self.keywords))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def get_current_time() -> str:
        return datetime.now().isoformat()


@dataclass
class UserProfile:
    """用户画像三维度结构"""
    # 过去：累积的稳定画像（存储在 user_domain.json）
    past: dict = field(default_factory=dict)
    # 现在：当前对话中识别的用户状态
    present: dict = field(default_factory=lambda: {
        "emotion": "",        # 当前情绪
        "intent": "",         # 当前意图
        "context": ""         # 当前情境
    })
    # 将来：结合过去画像+相关记忆+当前状态的预测
    future: dict = field(default_factory=lambda: {
        "prediction": "",     # 预测结果
        "risk": "",           # 潜在风险
        "basis": ""           # 预测依据
    })

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentPersona:
    """智能体人设三维度结构"""
    # 过去：稳定不变的人设（存储在 self_domain.json）
    past: dict = field(default_factory=dict)
    # 现在：当前是否与用户达成情感共情
    present: dict = field(default_factory=lambda: {
        "empathy": "",        # 共情内容
        "emotion_state": ""   # 智能体当前情绪状态
    })
    # 将来：基于人设和预测，智能体将采取的引导措施
    future: dict = field(default_factory=lambda: {
        "action": "",         # 引导行动
        "style": ""           # 表达风格（由人设决定）
    })

    def to_dict(self) -> dict:
        return asdict(self)