from typing import List
from memory_structures import InfoBlock, ordered_dedup
from logger import logger

class InfoBlockManager:
    """信息块管理器：负责InfoBlock的初始化与更新"""
    
    @staticmethod
    def initialize_info_block() -> InfoBlock:
        """初始化空的InfoBlock"""
        return InfoBlock(
            key_info_block=[],
            aux_info_block=[],
            noise_block=[]
        )
    
    @staticmethod
    def update_info_block(
        current_info_block: InfoBlock,
        new_key_elements: List[str],
        new_detailed_elements: List[str]
    ) -> InfoBlock:
        """
        更新InfoBlock（合并新要素并去重）
        :param current_info_block: 当前信息块
        :param new_key_elements: 新提取的关键要素
        :param new_detailed_elements: 新提取的细节要素
        :return: 更新后的信息块
        """
        # 关键信息块更新
        updated_key = current_info_block.key_info_block + new_key_elements
        updated_key = ordered_dedup(updated_key)
        
        # 辅助信息块更新
        updated_aux = current_info_block.aux_info_block + new_detailed_elements
        updated_aux = ordered_dedup(updated_aux)
        
        # 噪声块保持不变（单独通过add_noise方法更新）
        return InfoBlock(
            key_info_block=updated_key,
            aux_info_block=updated_aux,
            noise_block=current_info_block.noise_block
        )
    
    @staticmethod
    def add_noise(current_info_block: InfoBlock, noise: str) -> InfoBlock:
        """添加噪声到信息块"""
        updated_noise = current_info_block.noise_block + [noise]
        updated_noise = ordered_dedup(updated_noise)
        return InfoBlock(
            key_info_block=current_info_block.key_info_block,
            aux_info_block=current_info_block.aux_info_block,
            noise_block=updated_noise
        )