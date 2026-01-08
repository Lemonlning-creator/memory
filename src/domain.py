import os
from typing import Dict, Any, Optional
import json
from datetime import datetime, timedelta
from logger import logger
from llm_client import LLMClient
import prompt

# 持久化文件路径（运行时保存/加载的文件）
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
# USER_DOMAIN_PATH = os.path.join(DATA_DIR, "update/user_domain.json")
# SELF_DOMAIN_PATH = os.path.join(DATA_DIR, "update/self_domain.json")

# 默认初始数据文件路径
DEFAULT_USER_DOMAIN_JSON = os.path.join(DATA_DIR, "init/user_domain.json")
DEFAULT_SELF_DOMAIN_JSON = os.path.join(DATA_DIR, "init/self_domain.json")

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# ===================== 通用工具函数 =====================
def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """加载JSON文件，处理异常并返回字典（失败返回None）"""
    if not os.path.exists(file_path):
        logger.warning(f"JSON文件不存在：{file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"成功加载JSON文件：{file_path}")
        return data
    except Exception as e:
        logger.error(f"加载JSON文件失败 {file_path}：{e}", exc_info=True)
        return None

# ===================== 用户域类（无meta_info） =====================
class UserDomain:
    """用户域：智能体对用户的认知结构（完全匹配新JSON格式）"""
    
    def __init__(self, persist_path: str = DEFAULT_USER_DOMAIN_JSON, default_json_path: str = DEFAULT_USER_DOMAIN_JSON):
        self.persist_path = persist_path
        self.default_json_path = default_json_path
        
        # 仅保留业务四层结构，无meta_info
        self.meta_layer: Dict[str, Any] = {}
        self.cognitive_layer: Dict[str, Any] = {}
        self.behavior_layer: Dict[str, Any] = {}
        self.concrete_layer: Dict[str, Any] = {}

        # 核心逻辑：优先加载持久化文件 → 无则加载默认JSON → 保存初始数据
        if not self._load_from_persist_file():
            self._load_from_default_json()
            self.save_to_file()

    def _load_from_persist_file(self) -> bool:
        """从持久化文件加载数据（仅读取四层结构）"""
        data = load_json_file(self.persist_path)
        if not data:
            return False
        
        # 只解析业务四层，忽略旧数据中的meta_info（如有）
        self.meta_layer = data.get("Meta_Layer", {})
        self.cognitive_layer = data.get("Cognitive_Layer", {})
        self.behavior_layer = data.get("Behavior_Layer", {})
        self.concrete_layer = data.get("Concrete_Layer", {})
        return True

    def _load_from_default_json(self) -> None:
        """从你提供的默认JSON文件加载初始数据"""
        default_data = load_json_file(self.default_json_path)
        if not default_data:
            logger.error("默认用户域JSON文件加载失败，使用空数据")
            return
        
        # 精准匹配JSON中的结构：User_Domain_LiXiaoyao → 四层
        self.meta_layer = default_data.get("Meta_Layer", {})
        self.cognitive_layer = default_data.get("Cognitive_Layer", {})
        self.behavior_layer = default_data.get("Behavior_Layer", {})
        self.concrete_layer = default_data.get("Concrete_Layer", {})

    def save_to_file(self) -> None:
        """保存数据到持久化文件（仅保存四层结构）"""
        try:
            # 组装的字典仅包含业务四层，无meta_info
            save_data = {
                "Meta_Layer": self.meta_layer,
                "Cognitive_Layer": self.cognitive_layer,
                "Behavior_Layer": self.behavior_layer,
                "Concrete_Layer": self.concrete_layer
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            logger.info(f"用户域数据已保存到：{self.persist_path}")
        except Exception as e:
            logger.error(f"保存用户域数据失败：{e}", exc_info=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（供LLM调用，无meta_info）"""
        return {
            "Meta_Layer": self.meta_layer,
            "Cognitive_Layer": self.cognitive_layer,
            "Behavior_Layer": self.behavior_layer,
            "Concrete_Layer": self.concrete_layer
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从LLM返回的字典更新数据（仅更新四层）"""
        self.meta_layer = data.get("Meta_Layer", self.meta_layer)
        self.cognitive_layer = data.get("Cognitive_Layer", self.cognitive_layer)
        self.behavior_layer = data.get("Behavior_Layer", self.behavior_layer)
        self.concrete_layer = data.get("Concrete_Layer", self.concrete_layer)

# ===================== 自我域类（无meta_info） =====================
class SelfDomain:
    """自我域：智能体对自己的认知结构（完全匹配新JSON格式）"""
    
    def __init__(self, persist_path: str = DEFAULT_SELF_DOMAIN_JSON, default_json_path: str = DEFAULT_SELF_DOMAIN_JSON):
        self.persist_path = persist_path
        self.default_json_path = default_json_path
        
        # 仅保留业务四层结构，无meta_info
        self.meta_layer: Dict[str, Any] = {}
        self.cognitive_layer: Dict[str, Any] = {}
        self.behavior_layer: Dict[str, Any] = {}
        self.concrete_layer: Dict[str, Any] = {}

        # 核心逻辑：优先加载持久化文件 → 无则加载默认JSON → 保存初始数据
        if not self._load_from_persist_file():
            self._load_from_default_json()
            self.save_to_file()

    def _load_from_persist_file(self) -> bool:
        """从持久化文件加载数据（仅读取四层结构）"""
        data = load_json_file(self.persist_path)
        if not data:
            return False
        
        # 解析业务四层
        self.meta_layer = data.get("Meta_Layer", {})
        self.cognitive_layer = data.get("Cognitive_Layer", {})
        self.behavior_layer = data.get("Behavior_Layer", {})
        self.concrete_layer = data.get("Concrete_Layer", {})
        return True

    def _load_from_default_json(self) -> None:
        """从你提供的默认JSON文件加载初始数据"""
        default_data = load_json_file(self.default_json_path)
        if not default_data:
            logger.error("默认自我域JSON文件加载失败，使用空数据")
            return
        
        # 精准匹配JSON中的结构：Self_Domain_Wukong → 四层
        self.meta_layer = default_data.get("Meta_Layer", {})
        self.cognitive_layer = default_data.get("Cognitive_Layer", {})
        self.behavior_layer = default_data.get("Behavior_Layer", {})
        self.concrete_layer = default_data.get("Concrete_Layer", {})

    def save_to_file(self) -> None:
        """保存数据到持久化文件（仅保存四层结构）"""
        try:
            save_data = {
                "Meta_Layer": self.meta_layer,
                "Cognitive_Layer": self.cognitive_layer,
                "Behavior_Layer": self.behavior_layer,
                "Concrete_Layer": self.concrete_layer
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            logger.info(f"自我域数据已保存到：{self.persist_path}")
        except Exception as e:
            logger.error(f"保存自我域数据失败：{e}", exc_info=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（供LLM调用，无meta_info）"""
        return {
            "Meta_Layer": self.meta_layer,
            "Cognitive_Layer": self.cognitive_layer,
            "Behavior_Layer": self.behavior_layer,
            "Concrete_Layer": self.concrete_layer
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从LLM返回的字典更新数据（仅更新四层）"""
        self.meta_layer = data.get("Meta_Layer", self.meta_layer)
        self.cognitive_layer = data.get("Cognitive_Layer", self.cognitive_layer)
        self.behavior_layer = data.get("Behavior_Layer", self.behavior_layer)
        self.concrete_layer = data.get("Concrete_Layer", self.concrete_layer)

# ===================== 域管理器（无meta_info相关逻辑） =====================
class DomainManager:
    """域管理器：处理域的激活和更新（无meta_info）"""
    
    def __init__(self):
        self.user_domain = UserDomain()
        self.self_domain = SelfDomain()
        self.llm_client = LLMClient()

        self.last_update_time = datetime.now()
        self.update_interval = timedelta(hours=24)  # 每天更新一次

    def _save_domains(self) -> None:
        """保存用户域和自我域"""
        self.user_domain.save_to_file()
        self.self_domain.save_to_file()
    
    def activate_user_domain(self, user_input: str, conversation_history: str) -> UserDomain:
        """激活用户域：基于用户输入和对话历史更新"""
        logger.info("激活用户域...")
        
        activation_prompt = prompt.get_user_domain_activation_prompt(
            current_user_domain=self.user_domain.to_dict(),
            user_input=user_input,
            conversation_history=conversation_history
        )
        
        result = self.llm_client.call_non_stream(prompt=activation_prompt)
        if isinstance(result, dict):
            self.user_domain.from_dict(result)
        
        return self.user_domain

    def activate_self_domain(self, user_input: str, conversation_history: str, trust: int = 0) -> Dict[str, Any]:
            """激活自我域：仅返回激活后的字典，不覆盖原始全量数据"""
            logger.info("激活自我域...")
            
            # 始终使用全量数据进行激活计算
            current_full_data = self.self_domain.to_dict()
            
            activation_prompt = prompt.get_self_domain_activation_prompt(
                current_self_domain=current_full_data,
                user_input=user_input,
                conversation_history=conversation_history,
                trust=trust
            )
            
            result = self.llm_client.call_non_stream(prompt=activation_prompt)
            
            # 如果 LLM 正常返回，返回这个激活后的局部字典
            if isinstance(result, dict):
                logger.info(f"成功获取激活域片段")
                return result
            
            # 如果失败，返回全量数据作为保底
            return current_full_data
    
    def should_update_domains(self) -> bool:
        """判断是否需要更新域（基于时间间隔）"""
        return datetime.now() - self.last_update_time >= self.update_interval
    
    def update_domains(self, memory_store: "MemoryStore") -> None:
        """更新域：基于累积的记忆更新"""
        logger.info("开始更新域...")
        
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
        
        # 更新自我域
        self_update_prompt = prompt.get_self_domain_update_prompt(
            current_self_domain=self.self_domain.to_dict(),
            user_domain=self.user_domain.to_dict(),
            recent_memories=recent_memories
        )
        self_result = self.llm_client.call_non_stream(prompt=self_update_prompt)
        if isinstance(self_result, dict):
            self.self_domain.from_dict(self_result)
        
        self.last_update_time = datetime.now()
        self._save_domains() 
        logger.info("域更新完成")
    
    def is_memory_worthy(self, memory_content: Dict[str, Any]) -> bool:
        """判断记忆是否值得保存"""
        logger.info("判断记忆是否值得保存...")
        
        prompt_text = prompt.get_memory_worthiness_prompt(
            memory_content=memory_content,
            user_domain=self.user_domain.to_dict(),
            self_domain=self.self_domain.to_dict()
        )
        
        result = self.llm_client.call_non_stream(prompt=prompt_text)
        if isinstance(result, dict) and "is_worthy" in result:
            return result["is_worthy"]
        
        return True