from typing import List
from memory_structures import Element
import prompt
from llm_client import LLMClient
from logger import logger

class ElementExtractor:
    """要素提取器：基于大模型提取关键要素和细节要素"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def extract_elements(self, dialog: str, topic: str) -> Element:
        """
        提取要素入口方法（大模型驱动）
        :param dialog: 当前对话文本（非噪声）
        :param topic: 当前主题
        :return: Element实例（包含key_elements和detailed_elements）
        """
        # 1. 生成提示词
        extract_prompt = prompt.get_element_extract_prompt(dialog=dialog, current_topic=topic)
        # logger.info(f"要素提取提示词：{extract_prompt}")
        
        # 2. 大模型非流式调用
        result = self.llm_client.call_non_stream(prompt=extract_prompt)
        # print("element:", result)
        
        # 严格校验返回格式
        if not isinstance(result, dict):
            logger.warning("LLM返回非字典格式，使用空要素")
            return Element([], [])
        
        key_elements = result.get("key_elements", [])
        detailed_elements = result.get("detailed_elements", [])
        # 强制类型转换（避免非列表类型导致的错误）
        key_elements = key_elements if isinstance(key_elements, list) else []
        detailed_elements = detailed_elements if isinstance(detailed_elements, list) else []
        return Element(key_elements, detailed_elements)