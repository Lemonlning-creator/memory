from typing import Optional, List
from memory_structures import Memory
from noise_detector import NoiseDetector
from llm_client import LLMClient
from prompt import (
    boundary_detection_prompt,
    get_topic_initialize_prompt,
    get_noise_detection_prompt,
    get_topic_summary_prompt,
    get_content_summary_prompt,
    get_keywords_extract_prompt
)
from logger import logger
import json

class MemoryBuilder:
    """记忆构建器：管理对话buffer和记忆生成"""
    
    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.llm_client = LLMClient()
        self.buffer: List[str] = []
        self.current_topic: Optional[str] = None
        # 累积当前话题内所有轮次的三维度分析
        self._user_predictions: List[str] = []
        self._user_risks: List[str] = []
        self._agent_empathies: List[str] = []
        self._agent_actions: List[str] = []
    
    def _format_round_dialog(self, user_input: str, agent_response: str) -> str:
        """格式化一轮对话"""
        return f"user: {user_input.strip()}\nagent: {agent_response.strip()}"
    
    def _detect_topic_boundary(self, new_round_dialog: str) -> bool:
        """检测话题是否更换"""
        try:
            conversation_history = "\n\n".join(self.buffer) if self.buffer else ""
            prompt = boundary_detection_prompt(
                conversation_history=conversation_history,
                new_messages=new_round_dialog
            )
            result = self.llm_client.call_non_stream(prompt=prompt)
            
            if not isinstance(result, dict):
                logger.warning("话题边界检测结果解析失败，默认未更换")
                return False
            
            logger.info(f"话题边界检测结果：{result}")
            return result.get("topic_changed", False)
        except Exception as e:
            logger.error(f"话题边界检测失败：{str(e)}", exc_info=True)
            return False
    
    def _initialize_topic(self, first_dialog: str) -> str:
        """初始化话题"""
        try:
            prompt = get_topic_initialize_prompt(first_dialog=first_dialog)
            result = self.llm_client.call_non_stream(prompt=prompt)
            
            if isinstance(result, dict) and "topic" in result:
                return result["topic"].strip()
            else:
                fallback = first_dialog[:30].strip()
                logger.warning(f"主题初始化失败，使用 fallback: {fallback}")
                return fallback
        except Exception as e:
            logger.error(f"主题初始化失败：{str(e)}", exc_info=True)
            return first_dialog[:30].strip()
    
    def _summarize_topic(self) -> str:
        """总结当前buffer中的对话主题"""
        try:
            prompt = get_topic_summary_prompt(dialogs=self.buffer)
            result = self.llm_client.call_non_stream(prompt=prompt)
            
            if isinstance(result, dict) and "topic" in result:
                return result["topic"].strip()
            else:
                fallback = self.current_topic or "未命名主题"
                logger.warning(f"主题总结失败，使用 fallback: {fallback}")
                return fallback
        except Exception as e:
            logger.error(f"主题总结失败：{str(e)}", exc_info=True)
            return self.current_topic or "未命名主题"
    
    def _summarize_content(self) -> str:
        """总结当前buffer中的对话内容"""
        try:
            prompt = get_content_summary_prompt(dialogs=self.buffer)
            result = self.llm_client.call_non_stream(prompt=prompt)
            
            if isinstance(result, dict) and "content" in result:
                return result["content"].strip()
            else:
                fallback = "\n".join(self.buffer[-3:])  # 取最后三轮作为 fallback
                logger.warning(f"内容总结失败，使用 fallback: {fallback}")
                return fallback
        except Exception as e:
            logger.error(f"内容总结失败：{str(e)}", exc_info=True)
            return "\n".join(self.buffer[-3:])
    
    def _extract_keywords(self) -> List[str]:
        """提取当前buffer中的对话关键词"""
        try:
            prompt = get_keywords_extract_prompt(dialogs=self.buffer)
            result = self.llm_client.call_non_stream(prompt=prompt)
            
            if isinstance(result, dict) and "keywords" in result and isinstance(result["keywords"], list):
                return [str(k).strip() for k in result["keywords"] if k.strip()]
            else:
                logger.warning("关键词提取失败，返回空列表")
                return []
        except Exception as e:
            logger.error(f"关键词提取失败：{str(e)}", exc_info=True)
            return []
    
    def process_dialog(self, user_input: str, agent_response: str,
                       user_profile=None, agent_persona=None) -> Optional[Memory]:
        """处理一轮对话，返回需要保存的记忆（如果有的话）"""
        current_round = self._format_round_dialog(user_input, agent_response)
        logger.info(f"处理对话轮次：{current_round}")

        # 每轮都追加三维度分析到累积列表
        if user_profile:
            if user_profile.future.get("prediction"):
                self._user_predictions.append(user_profile.future["prediction"])
            if user_profile.future.get("risk"):
                self._user_risks.append(user_profile.future["risk"])
        if agent_persona:
            if agent_persona.present.get("empathy"):
                self._agent_empathies.append(agent_persona.present["empathy"])
            if agent_persona.future.get("action"):
                self._agent_actions.append(agent_persona.future["action"])

        # 1. 首轮对话
        if not self.buffer:
            logger.info("首轮对话，初始化话题和buffer")
            self.buffer.append(current_round)
            self.current_topic = self._initialize_topic(current_round)
            return None

        # 2. 非首轮对话：检测话题是否更换
        topic_changed = self._detect_topic_boundary(current_round)

        if not topic_changed:
            self.buffer.append(current_round)
            logger.info("话题未更换，已添加到buffer")
            return None

        else:
            logger.info("话题已更换，处理当前buffer")
            is_noise = self.noise_detector.is_noise(
                dialog=current_round,
                topic_context=f"当前旧主题：{self.current_topic}"
            )
            if is_noise:
                logger.info("检测到噪声，已忽略该轮对话")
                return None

            memory = self._build_memory()

            # 重置buffer和累积列表，准备新话题
            self.buffer = [current_round]
            self.current_topic = self._initialize_topic(current_round)
            self._user_predictions = []
            self._user_risks = []
            self._agent_empathies = []
            self._agent_actions = []

            logger.info(f"已生成记忆：{memory.topic}")
            return memory
    
    def _build_memory(self) -> Memory:
        """用当前 buffer 和累积的三维度分析构建记忆"""
        create_time = Memory.get_current_time()
        return Memory(
            topic=self._summarize_topic(),
            content=self._summarize_content(),
            keywords=self._extract_keywords(),
            create_time=create_time,
            update_time=create_time,
            user_prediction=" | ".join(dict.fromkeys(self._user_predictions)),
            user_risk=" | ".join(dict.fromkeys(self._user_risks)),
            agent_empathy=" | ".join(dict.fromkeys(self._agent_empathies)),
            agent_action=" | ".join(dict.fromkeys(self._agent_actions)),
        )

    def finalize_memory(self, user_profile=None, agent_persona=None) -> Optional[Memory]:
        """对话结束时，处理剩余的buffer内容"""
        if not self.buffer:
            return None
        logger.info("对话结束，处理剩余buffer内容")
        return self._build_memory()