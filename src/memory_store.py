import json
import os
from typing import List, Optional, Dict, Callable
from memory_structures import Memory
from logger import logger
import config
from domain import DomainManager 
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

class MemoryStore:
    """记忆存储管理器：负责记忆的持久化存储"""
    
    def __init__(self, is_worthy_func: Optional[Callable[[Dict], bool]] = None):
        self.memory_path = config.MEMORY_JSONL_PATH
        self.is_worthy_func = is_worthy_func
        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        # 初始化向量模型（用于检索）
        self.embedding_model = SentenceTransformer('/amax/xidian_ty/ln/memory/models/paraphrase-multilingual-MiniLM-L12-v2')
    

    ########################记忆直接存储方法（不含域约束判断）########################
    # def save_memory(self, memory: Memory) -> bool:
    #     """保存记忆到JSONL文件"""
    #     try:
    #         with open(self.memory_path, 'a', encoding='utf-8') as f:
    #             json.dump(memory.to_dict(), f, ensure_ascii=False)
    #             f.write('\n')
    #         logger.info(f"记忆已保存：{memory.topic}")
    #         return True
    #     except Exception as e:
    #         logger.error(f"保存记忆失败：{str(e)}", exc_info=True)
    #         return False

    ################################域约束下的记忆存储#############################
    def save_memory(self, memory: Memory) -> bool:
        """
        保存记忆到JSONL文件（增加域约束判断）
        :param memory: 要保存的记忆对象
        :return: 是否成功保存
        """
        # 将记忆转换为字典格式，用于判断
        memory_dict = memory.to_dict()
        
        # 如果有判断函数，先判断
        if self.is_worthy_func and not self.is_worthy_func(memory_dict):
            logger.info(f"记忆不符合域约束，不保存：{memory.topic}")
            return False
        
        # 如果符合约束，执行保存操作
        try:
            with open(self.memory_path, 'a', encoding='utf-8') as f:
                json.dump(memory_dict, f, ensure_ascii=False)
                f.write('\n')
            logger.info(f"记忆已保存：{memory.topic}")
            return True
        except Exception as e:
            logger.error(f"保存记忆失败：{str(e)}", exc_info=True)
            return False
    
    def load_all_memories(self) -> List[Dict]:
        """从JSONL文件加载所有记忆"""
        memories = []
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue  # 跳过空行
                        try:
                            memories.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"第 {line_num} 行 JSON 解析失败: {e.msg}，原始内容: {line}")
                        except Exception as e:
                            logger.error(f"第 {line_num} 行读取失败: {str(e)}", exc_info=True)
            logger.info(f"已加载 {len(memories)} 条记忆")
        except Exception as e:
            logger.error(f"加载记忆失败：{str(e)}", exc_info=True)
        return memories
    
    def get_latest_memory(self) -> Optional[Dict]:
        """获取最新的一条记忆"""
        memories = self.load_all_memories()
        return memories[-1] if memories else None
    
    def clear_all_memories(self) -> bool:
        """清空所有记忆"""
        try:
            if os.path.exists(self.memory_path):
                os.remove(self.memory_path)
            logger.info("所有记忆已清空")
            return True
        except Exception as e:
            logger.error(f"清空记忆失败：{str(e)}", exc_info=True)
            return False
        
    def retrieve_related_memories(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        根据用户输入检索相关记忆（向量匹配）
        :param query: 用户输入
        :param top_k: 返回最相关的 top_k 条记忆
        :return: 按相似度排序的记忆列表
        """
        memories = self.load_all_memories()
        if not memories:
            logger.info("没有记忆可供检索")
            return []
        
        # 限制 top_k 不超过记忆数量
        top_k = min(top_k, len(memories))
        
        # 将记忆内容拼接成一个字符串（用于向量表示）
        memory_texts = [
            f"{mem['topic']} {mem['content']} {' '.join(mem['keywords'])}"
            for mem in memories
        ]
        
        # 向量化用户输入和记忆文本
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        memory_embeddings = self.embedding_model.encode(memory_texts, convert_to_tensor=True)
        
        # 计算余弦相似度
        cos_scores = util.cos_sim(query_embedding, memory_embeddings)[0]
        
        # 按相似度排序
        if torch.is_tensor(cos_scores):
            # 使用 torch.topk 替代 numpy 的排序，效率更高，且不需要离开 GPU
            _, top_indices = torch.topk(cos_scores, k=top_k)
            top_results = top_indices.flatten().tolist()
        else:
            # 原有的 numpy 逻辑作为备选
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        
        # 组装结果
        results = []
        for idx in top_results:
            if cos_scores[idx] > 0.3:  # 相似度阈值，可调整
                results.append({
                    "memory": memories[idx],
                    "similarity": float(cos_scores[idx].cpu().numpy())
                })
        
        # 按相似度降序排列
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        logger.info(f"检索到 {len(results)} 条相关记忆")
        logger.info(f"相关记忆内容：{results}")
        return results