import json
from typing import Optional, List
from memory_structures import Topic
from logger import logger
import config

class MemoryStore:
    """记忆存储：单例模式，新增JSONL保存上一个主题记忆"""
    _instance: Optional["MemoryStore"] = None
    current_memory: Optional[Topic] = None
    _jsonl_path: str
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_memory = None  # 当前活跃主题记忆
            cls._instance._jsonl_path = config.MEMORY_JSONL_PATH  # JSONL存储路径
        return cls._instance
    
    def get_current_memory(self) -> Optional[Topic]:
        """获取当前活跃主题记忆"""
        return self.current_memory
    
    def update_memory(self, new_memory: Topic) -> None:
        """更新当前活跃主题记忆"""
        self.current_memory = new_memory
        logger.info(f"当前记忆更新：主题={new_memory.current_topic}，更新时间={new_memory.update_time}")
    
    def save_prev_memory_to_jsonl(self, prev_memory: Topic) -> bool:
        """
        将上一个主题记忆保存到JSONL文件（追加模式）
        :param prev_memory: 上一个已完成的主题记忆
        :return: 保存成功与否
        """
        if not prev_memory:
            logger.warning("尝试保存空的上一个记忆，跳过")
            return False
        
        try:
            # 转换为字典格式
            memory_dict = prev_memory.to_dict()
            # 追加写入JSONL（每行一个JSON对象）
            with open(self._jsonl_path, "a", encoding=config.JSONL_ENCODING) as f:
                json.dump(memory_dict, f, ensure_ascii=False, indent=None)
                f.write("\n")  # 换行分隔不同记忆
            logger.info(f"上一个主题记忆已保存到JSONL：主题={prev_memory.current_topic}，文件路径={self._jsonl_path}")
            return True
        except Exception as e:
            logger.error(f"保存上一个记忆到JSONL失败：{str(e)}", exc_info=True)
            return False
    
    def save_current_memory_on_exit(self) -> bool:
        """退出时保存当前活跃主题记忆（若存在）"""
        if not self.current_memory:
            logger.info("无当前活跃记忆，退出时无需保存")
            return False
        
        return self.save_prev_memory_to_jsonl(prev_memory=self.current_memory)
    
    def load_all_memories_from_jsonl(self) -> List[dict]:
        """从JSONL加载所有历史主题记忆（可选调用）"""
        memories = []
        try:
            with open(self._jsonl_path, "r", encoding=config.JSONL_ENCODING) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    memory_dict = json.loads(line)
                    memories.append(memory_dict)
            logger.info(f"从JSONL加载历史记忆成功，共{len(memories)}个主题")
        except FileNotFoundError:
            logger.warning(f"JSONL记忆文件不存在：{self._jsonl_path}")
        except Exception as e:
            logger.error(f"加载JSONL记忆失败：{str(e)}", exc_info=True)
        return memories
    
    def reset_memory(self) -> None:
        """重置当前活跃记忆（不影响JSONL历史）"""
        self.current_memory = None
        logger.info("当前活跃记忆已重置")