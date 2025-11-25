from typing import Optional
import config

class NoiseDetector:
    """噪声检测器：基于主题相关性判断对话是否为噪声"""
    
    @staticmethod
    def calculate_relevance(dialog: str, topic: Optional[str]) -> float:
        """
        计算对话与主题的相关性（简单实现：关键词匹配率，可替换为LLM语义相似度）
        :param dialog: 当前对话文本
        :param topic: 当前主题（新主题时为None）
        :return: 相关性得分（0-1）
        """
        if not topic:
            return 1.0  # 新主题无相关性计算，直接返回1.0（跳过噪声判断）
        
        # 提取主题关键词（简单分词，可替换为jieba等工具）
        topic_keywords = [w for w in topic.strip().split() if len(w) >= 2]
        if not topic_keywords:
            return 0.0
        
        # 计算对话中匹配的主题关键词占比
        dialog_words = set(dialog.strip().split())
        match_count = sum(1 for kw in topic_keywords if kw in dialog_words)
        return match_count / len(topic_keywords)
    
    def is_noise(self, dialog: str, topic: Optional[str]) -> bool:
        """
        判断对话是否为噪声
        :param dialog: 当前对话文本
        :param topic: 当前主题（新主题时为None）
        :return: True=噪声，False=非噪声
        """
        # 新主题第一轮对话跳过噪声判断
        if not topic and config.NEW_TOPIC_SKIP_NOISE_CHECK:
            return False
        
        relevance = self.calculate_relevance(dialog, topic)
        return relevance < config.NOISE_RELEVANCE_THRESHOLD