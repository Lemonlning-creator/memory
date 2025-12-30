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
        self.buffer: List[str] = []  # 存储当前话题的对话
        self.current_topic: Optional[str] = None  # 当前话题
    
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
    
    def process_dialog(self, user_input: str, agent_response: str) -> Optional[Memory]:
        """处理一轮对话，返回需要保存的记忆（如果有的话）"""
        current_round = self._format_round_dialog(user_input, agent_response)
        logger.info(f"处理对话轮次：{current_round}")
        
        # 1. 首轮对话
        if not self.buffer:
            logger.info("首轮对话，初始化话题和buffer")
            self.buffer.append(current_round)
            self.current_topic = self._initialize_topic(current_round)
            return None  # 首轮不保存记忆
        
        # 2. 非首轮对话：检测话题是否更换
        topic_changed = self._detect_topic_boundary(current_round)
        
        if not topic_changed:
            # 2.1 话题未更换：添加到buffer
            self.buffer.append(current_round)
            logger.info("话题未更换，已添加到buffer")
            return None
        
        else:
            # 2.2 话题更换：处理当前buffer
            logger.info("话题已更换，处理当前buffer")
            
            # 判断是否为噪声
            is_noise = self.noise_detector.is_noise(
                dialog=current_round,
                topic_context=f"当前旧主题：{self.current_topic}"
            )
            
            if is_noise:
                # 噪声：弃掉该轮对话，不影响当前buffer
                logger.info("检测到噪声，已忽略该轮对话")
                return None
            
            # 非噪声：生成记忆
            topic = self._summarize_topic()
            content = self._summarize_content()
            keywords = self._extract_keywords()
            create_time = Memory.get_current_time()
            
            memory = Memory(
                topic=topic,
                content=content,
                keywords=keywords,
                create_time=create_time,
                update_time=create_time
            )
            
            # 清空buffer，准备新话题
            self.buffer = []
            self.current_topic = self._initialize_topic(current_round)
            # 将当前轮对话添加到新buffer
            self.buffer.append(current_round)
            
            logger.info(f"已生成记忆并保存：{topic}")
            return memory
    
    def finalize_memory(self) -> Optional[Memory]:
        """对话结束时，处理剩余的buffer内容"""
        if not self.buffer:
            return None
        
        logger.info("对话结束，处理剩余buffer内容")
        
        # 生成记忆
        topic = self._summarize_topic()
        content = self._summarize_content()
        keywords = self._extract_keywords()
        create_time = Memory.get_current_time()
        
        memory = Memory(
            topic=topic,
            content=content,
            keywords=keywords,
            create_time=create_time,
            update_time=create_time
        )
        
        return memory