import json
import os
from typing import List, Optional, Dict
from memory_structures import Memory
from logger import logger
import config

class MemoryStore:
    """记忆存储管理器：负责记忆的持久化存储"""
    
    def __init__(self):
        self.memory_path = config.MEMORY_JSONL_PATH
        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
    
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