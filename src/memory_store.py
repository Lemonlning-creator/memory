import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
import json
from typing import List, Optional, Dict
from memory_structures import Memory
from logger import logger
import config
from domain import DomainManager 
import numpy as np
from sentence_transformers import SentenceTransformer, util

class MemoryStore:
    """记忆存储管理器：负责记忆的持久化存储"""
    
    def __init__(self):
        self.memory_path = config.MEMORY_JSONL_PATH
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    

    def save_memory(self, memory: Memory) -> bool:
        """保存记忆到JSONL文件"""
        try:
            with open(self.memory_path, 'a', encoding='utf-8') as f:
                json.dump(memory.to_dict(), f, ensure_ascii=False)
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
        
    def retrieve_related_memories(self, query: str, top_k: int = 5,
                                   user_prediction: str = "", user_risk: str = "") -> List[Dict]:
        """
        检索相关记忆，结合用户画像预测/风险做语义增强
        """
        memories = self.load_all_memories()
        if not memories:
            return []
        top_k = min(top_k, len(memories))

        # 将预测和风险拼入查询，增强语义匹配
        enhanced_query = query
        if user_prediction:
            enhanced_query += f" {user_prediction}"
        if user_risk:
            enhanced_query += f" {user_risk}"

        # 记忆文本：同时包含三维度字段，提升召回率
        memory_texts = []
        for mem in memories:
            text = f"{mem.get('topic','')} {mem.get('content','')} {' '.join(mem.get('keywords',[]))}"
            if mem.get('user_prediction'):
                text += f" {mem['user_prediction']}"
            if mem.get('user_risk'):
                text += f" {mem['user_risk']}"
            memory_texts.append(text)

        query_embedding = self.embedding_model.encode(enhanced_query, convert_to_tensor=True).cpu()
        memory_embeddings = self.embedding_model.encode(memory_texts, convert_to_tensor=True).cpu()

        cos_scores = util.cos_sim(query_embedding, memory_embeddings)[0].numpy()
        sorted_indices = np.argsort(-cos_scores)[:top_k]

        results = []
        for idx in sorted_indices:
            if cos_scores[idx] > 0.3:
                results.append({
                    "memory": memories[idx],
                    "similarity": float(cos_scores[idx])
                })

        logger.info(f"检索到 {len(results)} 条相关记忆（增强查询：{enhanced_query[:50]}）")
        return results