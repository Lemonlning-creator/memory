# from typing import bool
from llm_client import LLMClient
from prompt import get_noise_detection_prompt
from logger import logger

class NoiseDetector:
    def __init__(self):
        self.llm_client = LLMClient()
    
    def is_noise(self, dialog: str, topic_context: str = "") -> bool:
        """
        判断对话是否为无意义临时噪声
        :param dialog: 待判断的对话轮次
        :param topic_context: 主题上下文（辅助模型判断）
        :return: 是否为噪声
        """
        prompt = get_noise_detection_prompt(dialog=dialog, topic_context=topic_context)
        logger.debug(f"噪声检测提示词：{prompt}")
        result = self.llm_client.call_non_stream(prompt=prompt)
        
        if not isinstance(result, dict) or "is_noise" not in result:
            logger.warning("噪声检测结果解析失败，默认视为非噪声")
            return False
        
        return result["is_noise"]