from typing import Optional, List
from memory_structures import Topic
from datetime import datetime
import prompt
from llm_client import LLMClient
import config
from logger import logger
import json

class TopicManager:
    """主题管理器：仅负责Topic的初始化与更新（不涉及InfoBlock）"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    @staticmethod
    def _get_current_time() -> str:
        """获取ISO格式当前时间"""
        return datetime.now().isoformat()
    
    def initialize_topic(self, first_dialog: str) -> str:
        """
        初始化主题字符串（仅返回主题文本）
        :param first_dialog: 第一轮对话文本
        :return: 初始化的主题字符串
        """
        init_prompt = prompt.get_topic_initialize_prompt(first_dialog=first_dialog)
        # logger.info(f"主题初始化提示词：{init_prompt}")
        result = self.llm_client.call_non_stream(prompt=init_prompt)
        # # print("topic init result:", type(result))

        # if isinstance(result, str):
        #     try:
        #         result_json = json.loads(result)
        #     except json.JSONDecodeError as e:
        #         logger.error(f"JSON字符串解析失败：{e}")
        
        if isinstance(result, dict) and "topic" in result:
            return result["topic"].strip()[:config.TOPIC_INIT_MAX_LENGTH]
        else:
            fallback = first_dialog[:config.TOPIC_INIT_MAX_LENGTH].strip()
            logger.warning(f"主题初始化失败，使用 fallback: {fallback}")
            return fallback
    
    def should_update_topic(
        self, 
        current_topic_str: str, 
        current_key_info: List[str], 
        new_key_elements: List[str]
    ) -> bool:
        """
        判断是否需要更新主题（仅基于主题文本和关键要素）
        :param current_topic_str: 当前主题文本
        :param current_key_info: 当前关键信息块
        :param new_key_elements: 新提取的关键要素
        :return: 是否需要更新
        """
        update_prompt = prompt.get_topic_update_prompt(
            current_topic=current_topic_str,
            current_key_info=current_key_info,
            new_key_elements=new_key_elements
        )
        result = self.llm_client.call_non_stream(prompt=update_prompt)
        
        if isinstance(result, dict) and "need_update" in result:
            return result["need_update"]
        else:
            logger.warning(f"主题更新判断失败，默认不更新")
            return False
    
    def update_topic(
        self, 
        current_topic_str: str, 
        current_key_info: List[str], 
        new_key_elements: List[str]
    ) -> str:
        """
        更新主题文本（仅返回新主题字符串）
        :param current_topic_str: 当前主题文本
        :param current_key_info: 当前关键信息块
        :param new_key_elements: 新提取的关键要素
        :return: 新主题文本
        """
        update_prompt = prompt.get_topic_update_prompt(
            current_topic=current_topic_str,
            current_key_info=current_key_info,
            new_key_elements=new_key_elements
        )
        result = self.llm_client.call_non_stream(prompt=update_prompt)
        
        if isinstance(result, dict) and "new_topic" in result:
            return result["new_topic"].strip()[:config.TOPIC_UPDATE_MAX_LENGTH]
        else:
            logger.warning(f"主题更新失败，保留原主题")
            return current_topic_str