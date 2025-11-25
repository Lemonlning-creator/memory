import logging
from typing import Optional
import config

def get_logger(name: str = "memory_system") -> logging.Logger:
    """
    获取配置好的日志器（单例效果）
    :param name: 日志器名称
    :return: 配置后的Logger实例
    """
    # 避免重复配置
    if logging.getLogger(name).handlers:
        return logging.getLogger(name)
    
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # 配置文件处理器（追加模式，避免覆盖）
    file_handler = logging.FileHandler(config.LOG_FILE_PATH, mode="a", encoding=config.JSONL_ENCODING)
    file_handler.setLevel(config.LOG_LEVEL)
    
    # 配置日志格式
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    
    return logger

# 全局日志实例（所有模块共享）
logger = get_logger()