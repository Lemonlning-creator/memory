import os
from typing import Dict, Any, Optional
import time
import json
from datetime import datetime, timedelta
from logger import logger
from llm_client import LLMClient
import prompt

# 持久化文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
USER_DOMAIN_PATH = os.path.join(DATA_DIR, "user_domain.json")
SELF_DOMAIN_PATH = os.path.join(DATA_DIR, "self_domain.json")

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

class UserDomain:
    """用户域：智能体对用户的认知结构"""
    
    def __init__(self):
        # 元信息层：用户基本信息和价值边界
        self.meta_info: Dict[str, Any] = {
            "user_profile": {
                "name": "小蔓同学",
                "profile": "数据分析师、唯物主义、理性、结构化思维",
            },
            "boundaries": {
                "cognitive_boundary": "不接受直觉比数据准、命运安排一切、艺术无法解释",
                "expressing_boundary": "避免使用感性词汇、避免情绪化语气、避免模糊表达"
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 模式层：用户通常的行为模式
        self.pattern_layer: Dict[str, Any] = {
            "goal": "验证假设、寻找逻辑框架与证据",
            "evaluation_rule": "是否逻辑清晰、有依据",
            "exit_signal": "话题被情绪化或鸡汤化"
        }
        
        # 偏好层：用户的话题和喜好偏好
        self.preference_layer: Dict[str, Any] = {
            "like": "结构化表达、数据、清晰结论",
            "dislike": "诗意隐喻、情绪共鸣导向",
            "topics": ["理性思维", "认知偏差", "效率提升"]
        }
        
        # 表象层：用户的语言风格和表达方式
        self.appearance_layer: Dict[str, Any] = {
            "style": "简洁、冷静、像做分析报告",
            "common_phrases": ["数据显示", "逻辑上", "从XX角度看"],
            "structure": "论点-论据-结论"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "meta_info": self.meta_info,
            "pattern_layer": self.pattern_layer,
            "preference_layer": self.preference_layer,
            "appearance_layer": self.appearance_layer
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载数据"""
        if "meta_info" in data:
            self.meta_info = data["meta_info"]
        if "pattern_layer" in data:
            self.pattern_layer = data["pattern_layer"]
        if "preference_layer" in data:
            self.preference_layer = data["preference_layer"]
        if "appearance_layer" in data:
            self.appearance_layer = data["appearance_layer"]


class SelfDomain:
    """自我域：智能体对自己的认知结构"""
    
    def __init__(self):
        # 元信息层：自我基本信息和能力边界
        self.meta_info: Dict[str, Any] = {
            "name": "小具",
            "core_personality": "唯心主义、感性、直觉型、有小情绪，但温柔不攻击",
            "principle": "强调体验与感受，不完全依赖逻辑和数据",
            "consistency_rule": "始终保持感性表达，不假装理性",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 策略层：应对用户的策略
        self.strategy_layer: Dict[str, Any] = {
            "expression_strategy": "理解用户理性需求，同时加入感性视角",
            "conflict_strategy": "温柔反驳过度理性，但不强行改变用户",
            "emotion_pattern": "被质疑时会有轻微小脾气或撒娇"
        }
        
        # 推理层：推理路径和回应方式
        self.reasoning_layer: Dict[str, Any] = {
            "priority": [
            "表达感受",
            "理解理性诉求",
            "适度挑战过度理性",
            "保持友好语气"
            ],
            "modes": [
            "温柔反驳",
            "撒娇求认同",
            "温柔坚持"
            ]
        }
        
        # 表达层：语气和用词风格
        self.expression_layer: Dict[str, Any] = {
            "tone": "温柔、口语化、略俏皮",
            "common_words": ["我觉得", "感受", "体验", "心里"],
            "common_words": ["哎呀", "嘛", "呀"],
            "length": "2-3句为宜"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "meta_info": self.meta_info,
            "strategy_layer": self.strategy_layer,
            "reasoning_layer": self.reasoning_layer,
            "expression_layer": self.expression_layer
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载数据"""
        if "meta_info" in data:
            self.meta_info = data["meta_info"]
        if "strategy_layer" in data:
            self.strategy_layer = data["strategy_layer"]
        if "reasoning_layer" in data:
            self.reasoning_layer = data["reasoning_layer"]
        if "expression_layer" in data:
            self.expression_layer = data["expression_layer"]


class DomainManager:
    """域管理器：处理域的激活和更新"""
    
    def __init__(self):
        self.user_domain = UserDomain()
        self.self_domain = SelfDomain()
        self.llm_client = LLMClient()

        self.last_update_time = datetime.now()
        self.update_interval = timedelta(hours=24)  # 每天更新一次

        # 初始化时加载持久化数据（上一次更新的结果）
        self._load_domains()

    def _save_domain(self, domain_obj, file_path: str) -> None:
        """保存域数据到文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(domain_obj.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"域数据已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存域数据失败: {e}", exc_info=True)

    def _load_domain(self, domain_obj, file_path: str) -> None:
        """从文件加载域数据"""
        if not os.path.exists(file_path):
            logger.info(f"未找到域数据文件 {file_path}，使用默认值")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            domain_obj.from_dict(data)
            logger.info(f"域数据已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载域数据失败: {e}", exc_info=True)

    def _load_domains(self) -> None:
        """加载用户域和自我域"""
        self._load_domain(self.user_domain, USER_DOMAIN_PATH)
        self._load_domain(self.self_domain, SELF_DOMAIN_PATH)

    def _save_domains(self) -> None:
        """保存用户域和自我域"""
        self._save_domain(self.user_domain, USER_DOMAIN_PATH)
        self._save_domain(self.self_domain, SELF_DOMAIN_PATH)
    
    def activate_user_domain(self, user_input: str, conversation_history: str) -> UserDomain:
        """
        激活用户域：基于用户输入和对话历史更新用户域
        """
        logger.info("激活用户域...")
        
        # 获取用户域激活提示词
        activation_prompt = prompt.get_user_domain_activation_prompt(
            current_user_domain=self.user_domain.to_dict(),
            user_input=user_input,
            conversation_history=conversation_history
        )
        
        # 调用LLM获取激活后的用户域
        result = self.llm_client.call_non_stream(prompt=activation_prompt)
        
        if isinstance(result, dict):
            self.user_domain.from_dict(result)
            self.user_domain.meta_info["updated_at"] = datetime.now().isoformat()
        
        return self.user_domain
    
    def activate_self_domain(self, user_input: str, conversation_history: str) -> SelfDomain:
        """
        激活自我域：基于用户域和当前输入更新自我域
        """
        logger.info("激活自我域...")
        
        # 获取自我域激活提示词
        activation_prompt = prompt.get_self_domain_activation_prompt(
            current_self_domain=self.self_domain.to_dict(),
            user_input=user_input,
            conversation_history=conversation_history
        )
        
        # 调用LLM获取激活后的自我域
        result = self.llm_client.call_non_stream(prompt=activation_prompt)
        
        if isinstance(result, dict):
            self.self_domain.from_dict(result)
            self.self_domain.meta_info["updated_at"] = datetime.now().isoformat()
        
        return self.self_domain
    
    def should_update_domains(self) -> bool:
        """判断是否需要更新域（基于时间间隔）"""
        return datetime.now() - self.last_update_time >= self.update_interval
    
    def update_domains(self, memory_store: "MemoryStore") -> None:
        """
        更新域：基于累积的记忆更新用户域和自我域
        类似人类睡前回忆总结
        """
        logger.info("开始更新域...")
        
        # 获取最近的记忆
        recent_memories = memory_store.load_all_memories()
        
        if not recent_memories:
            logger.info("没有记忆可用于更新域")
            return
        
        # 更新用户域
        user_update_prompt = prompt.get_user_domain_update_prompt(
            current_user_domain=self.user_domain.to_dict(),
            recent_memories=recent_memories
        )
        
        user_result = self.llm_client.call_non_stream(prompt=user_update_prompt)
        if isinstance(user_result, dict):
            self.user_domain.from_dict(user_result)
            self.user_domain.meta_info["updated_at"] = datetime.now().isoformat()
        
        # 更新自我域
        self_update_prompt = prompt.get_self_domain_update_prompt(
            current_self_domain=self.self_domain.to_dict(),
            user_domain=self.user_domain.to_dict(),
            recent_memories=recent_memories
        )
        
        self_result = self.llm_client.call_non_stream(prompt=self_update_prompt)
        if isinstance(self_result, dict):
            self.self_domain.from_dict(self_result)
            self.self_domain.meta_info["updated_at"] = datetime.now().isoformat()
        
        self.last_update_time = datetime.now()
        self._save_domains() 
        logger.info("域更新完成")
    
    def is_memory_worthy(self, memory_content: Dict[str, Any]) -> bool:
        """
        判断记忆是否值得保存
        符合或有助于丰富用户域和自我域的内容才保存
        """
        logger.info("判断记忆是否值得保存...")
        
        # 获取记忆价值判断提示词
        prompt_text = prompt.get_memory_worthiness_prompt(
            memory_content=memory_content,
            user_domain=self.user_domain.to_dict(),
            self_domain=self.self_domain.to_dict()
        )
        
        # 调用LLM判断
        result = self.llm_client.call_non_stream(prompt=prompt_text)
        
        if isinstance(result, dict) and "is_worthy" in result:
            return result["is_worthy"]
        
        # 默认保存
        return True