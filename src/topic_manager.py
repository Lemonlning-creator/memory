from typing import Optional, List
from memory_structures import Topic, InfoBlock
from datetime import datetime
import prompt
from llm_client import LLMClient
import config
from logger import logger

class TopicManager:
    """主题管理器：基于大模型初始化/更新主题"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    @staticmethod
    def _get_current_time() -> str:
        """获取ISO格式当前时间"""
        return datetime.now().isoformat()
    
    def initialize_topic(self, first_dialog: str) -> Topic:
        """
        初始化新主题（大模型提炼）
        :param first_dialog: 第一轮对话文本（非噪声）
        :return: 初始化后的Topic实例
        """
        # 1. 生成提示词
        init_prompt = prompt.get_topic_initialize_prompt(first_dialog=first_dialog)
        
        # 2. 大模型非流式调用
        result = self.llm_client.call_non_stream(prompt=init_prompt)
        
        # 3. 处理结果（解析失败则用原始对话截取）
        if isinstance(result, dict) and "topic" in result:
            initial_topic = result["topic"].strip()[:config.TOPIC_INIT_MAX_LENGTH]
            logger.info(f"主题初始化成功：对话={first_dialog}，主题={initial_topic}")
        else:
            initial_topic = first_dialog[:config.TOPIC_INIT_MAX_LENGTH].strip()
            logger.warning(f"主题初始化解析失败，降级使用对话截取：主题={initial_topic}")
        
        # 4. 初始化信息块
        initial_info_block = InfoBlock(
            key_info_block=[],
            aux_info_block=[],
            noise_block=[]
        )
        
        return Topic(
            current_topic=initial_topic,
            info_block=initial_info_block,
            create_time=self._get_current_time(),
            update_time=self._get_current_time()
        )
    
    def should_update_topic(self, current_topic: Topic, new_key_elements: List[str]) -> bool:
        """
        判断是否需要更新主题（大模型判断）
        :param current_topic: 当前主题实例
        :param new_key_elements: 本轮提取的关键要素
        :return: True=需要更新，False=无需更新
        """
        # 1. 生成提示词
        update_prompt = prompt.get_topic_update_prompt(
            current_topic=current_topic.current_topic,
            current_key_info=current_topic.info_block.key_info_block,
            new_key_elements=new_key_elements
        )
        
        # 2. 大模型非流式调用
        result = self.llm_client.call_non_stream(prompt=update_prompt)
        
        # 3. 处理结果（解析失败则不更新）
        if isinstance(result, dict) and "need_update" in result:
            return result["need_update"]
        else:
            logger.warning(f"主题更新判断失败，默认不更新：当前主题={current_topic}")
            return False
    
    def update_topic(self, current_topic: Topic, new_key_elements: List[str]) -> Topic:
        """
        更新主题（大模型生成新主题）
        :param current_topic: 当前主题实例
        :param new_key_elements: 本轮提取的关键要素
        :return: 更新后的Topic实例
        """
        # 1. 生成提示词（复用更新判断的提示词）
        update_prompt = prompt.get_topic_update_prompt(
            current_topic=current_topic.current_topic,
            current_key_info=current_topic.info_block.key_info_block,
            new_key_elements=new_key_elements
        )
        
        # 2. 大模型非流式调用
        result = self.llm_client.call_non_stream(prompt=update_prompt)
        print("topic:", result)
        
        # 3. 处理结果（解析失败则返回原主题）
        if isinstance(result, dict) and "new_topic" in result:
            new_topic_str = result["new_topic"].strip()[:config.TOPIC_UPDATE_MAX_LENGTH]  # 新增配置项，限制主题长度
            logger.info(f"主题更新成功：旧主题={current_topic.current_topic}，新主题={new_topic_str}")
        else:
            new_topic_str = current_topic.current_topic
            logger.warning(f"主题更新解析失败，保留原主题：{new_topic_str}")
        
        return Topic(
            current_topic=new_topic_str,
            info_block=current_topic.info_block,
            create_time=current_topic.create_time,
            update_time=self._get_current_time()
        )