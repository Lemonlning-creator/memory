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
        
        # 2. 大模型非流式调用
        result = self.llm_client.call_non_stream(prompt=extract_prompt)
        print("element:", result)
        
        # 3. 处理结果（解析失败则返回空要素）
        key_elements = []
        detailed_elements = []
        if isinstance(result, dict):
            key_elements = result.get("key_elements", [])
            detailed_elements = result.get("detailed_elements", [])
            # 安全校验：确保是列表类型
            key_elements = key_elements if isinstance(key_elements, list) else []
            detailed_elements = detailed_elements if isinstance(detailed_elements, list) else []
            logger.info(f"要素提取成功：关键要素={key_elements}，细节要素={detailed_elements}")
        else:
            logger.warning(f"要素提取解析失败，返回空要素：对话={dialog}")
        
        return Element(key_elements=key_elements, detailed_elements=detailed_elements)