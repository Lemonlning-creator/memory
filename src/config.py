# 噪声判断配置
NOISE_RELEVANCE_THRESHOLD = 0.3  # 对话与主题相关性阈值（低于此值判定为噪声）
NEW_TOPIC_SKIP_NOISE_CHECK = True  # 新主题第一轮对话跳过噪声判断

# Element提取配置
KEY_ELEMENT_MIN_LENGTH = 2  # 关键要素最小长度
KEY_ELEMENT_KEYWORDS = ["时间", "地点", "人物", "事件", "数量", "目标"]  # 关键要素关键词（可扩展）
DETAILED_ELEMENT_FILTER = ["补充", "说明", "细节", "备注"]  # 细节要素识别关键词

# Topic更新配置
TOPIC_UPDATE_THRESHOLD = 0.6  # 关键信息变化占比阈值（超过则更新主题）
TOPIC_INIT_MAX_LENGTH = 20  # 初始主题最大长度

# 存储配置
MEMORY_PERSIST_PATH = "./memory_cache.json"  # 记忆持久化路径（可选）

# LLM配置
LLM_PROVIDER = "openai"  # 支持 "openai"、"zhipu"、"qianfan" 等
LLM_MODEL = "qwen-max"  # 模型名称（根据提供商调整）
LLM_API_KEY = "sk-7c0d7996ace54d368aa8a6f05b0078e0"  # 替换为你的API密钥
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 自定义Base URL（如智谱、千帆的专属URL）
LLM_TEMPERATURE = 0.3  # 大模型温度（越低越稳定）
LLM_MAX_TOKENS = 512  # 最大输出 tokens

# 流式输出配置
STREAM_CHUNK_DELAY = 0.05  # 流式输出字符间隔（控制打印速度）

# 持久化配置（JSONL）
MEMORY_JSONL_PATH = "./output/memory_store.jsonl"  # 主题记忆存储路径（JSONL格式）
JSONL_ENCODING = "utf-8"  # JSONL文件编码

# 日志配置
LOG_FILE_PATH = "./logs/chat.log"  # 日志文件路径
LOG_LEVEL = "INFO"  # 日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"  # 日志格式
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # 时间戳格式