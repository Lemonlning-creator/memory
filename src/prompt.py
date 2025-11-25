"""
Prompt Templates
"""

def boundary_detection_prompt(conversation_history: str, new_messages: str) -> str:    
    """
    边界检测提示词
    :param conversation_history: 对话历史
    :param new_messages: 新增消息
    :return: 完整提示词
    """    
    return """
    你是一名对话边界检测专家，需要判断新增对话是否与当前主题完全无关（即话题更换），非轻微相关/边缘相关。

    当前对话历史：
    {conversation_history}

    新增消息：
    {new_messages}

    请从以下方面仔细分析，判断是否应开启新片段：

    1. **话题变更（最高优先级）**：

    -新增消息是否引入完全不同的话题？
    -对话是否从一个具体事件切换到另一个事件？
    -对话是否从一个问题转向无关的新问题？

    2. **意图转换**：

    -对话目的是否发生变化？（例如，从闲聊转为求助，从讨论工作转为讨论个人生活）
    -当前话题的核心问题是否已得到解答或充分讨论？

    3. **时间标记**：

    -是否存在时间过渡标记（“之前”“刚才”“对了”“哦对了”“另外” 等）？
    -消息之间的时间间隔是否超过 30 分钟？

    4. **结构信号**：

    -是否存在明确的话题过渡表述（“换个话题”“说到这个”“插个问题” 等）？
    -是否有表明当前话题已结束的总结性语句？

    5. **内容相关性**：

    -新增消息与之前讨论内容的关联度如何？（关联度＜30% 时考虑拆分）
    -是否涉及完全不同的人物、地点或事件？

    决策原则：
    -优先保证话题独立性：每个片段应围绕一个核心话题或事件展开
    -存疑时优先拆分：不确定时，倾向于开启新片段
    -保持合理长度：单个片段通常不应超过 10-15 条消息

    输出要求：
    必须严格按照以下格式返回，不要添加任何额外文字！
    1. JSON 内容需严格包裹在 ```json 和 ``` 之间（代码块格式）
    2. 仅包含指定字段，不得新增其他字段
    3. 字段类型严格匹配（topic_changed 为布尔值，confidence 为 0.0-1.0 的浮点数）

    输出格式示例：
    ```json
    {
        "topic_changed": true/false,
        "confidence": 0.0-1.0
    }
    ```

    注意事项：
    -若对话历史为空（即当前为第一条消息），返回 false
    -检测到明确话题变更时，即使对话过渡自然，也需拆分
    -每个片段应是独立完整的对话单元，可单独理解
"""

def get_element_extract_prompt(dialog: str, current_topic: str) -> str:
    """
    Element提取提示词：要求大模型结构化输出key_element和detailed_element
    :param dialog: 当前对话文本
    :param current_topic: 当前主题
    :return: 完整提示词
    """
    return """
    任务：基于当前对话主题，从用户和智能体的对话中提取关键要素（key_element）和细节要素（detailed_element）。
    定义说明：
    - 关键要素（key_element）：与当前主题强相关的核心信息，如时间、地点、人物、事件、数量、核心需求等，影响主题核心含义的信息。
    - 细节要素（detailed_element）：辅助性、补充性信息，不改变主题核心，但丰富主题细节，如补充说明、次要条件、偏好等。
    
    当前主题：{current_topic}
    用户对话：{dialog}
    
    输出要求：
    必须严格按照以下JSON格式输出，不要添加任何额外文字，否则会导致解析失败！
    1. JSON 内容需严格包裹在 ```json 和 ``` 之间（代码块格式）
    2. 仅包含指定字段，不得新增其他字段
    3. 无对应要素时，字段值为空数组 []
    4. 要素需去重，保持简洁（每个要素不超过 10 字）

    输出格式示例：
    ```json
    {
        "key_elements": ["要素1", "要素2", ...],  // 无则为空数组
        "detailed_elements": ["要素1", "要素2", ...]  // 无则为空数组
    }
    ```
    """

def get_topic_initialize_prompt(first_dialog: str) -> str:
    """
    主题初始化提示词：从第一轮对话提炼核心主题
    :param first_dialog: 第一轮对话文本
    :return: 完整提示词
    """
    return """
    任务：从用户的第一轮对话中提炼核心主题，主题需简洁明了（不超过20字），准确反映用户核心需求或话题。

    特殊处理规则：
    1. 若用户对话是简单问候、寒暄（如"hello"、"你好"、"嗨"、"早上好"等），主题统一提炼为"打招呼/问候"。
    2. 若用户对话是无明确核心需求的闲聊（如"今天天气不错"），主题提炼为"日常闲聊"。
    3. 若用户对话有明确需求/话题（如"求推荐午饭"、"想改代码"），提炼具体核心主题。

    用户对话：{first_dialog}
    
    输出要求：
    仅输出提炼后的主题文本，不要添加任何额外文字！
    1. JSON 内容需严格包裹在 ```json 和 ``` 之间（代码块格式）
    2. 仅包含 "topic" 字段，字段值为提炼后的主题文本
    3. 主题文本长度不超过 20 字

    输出格式示例：
    ```json
    {
        "topic": "提炼后的主题文本"
    }
    ```
    """

def get_topic_update_prompt(current_topic: str, current_key_info: list, new_key_elements: list) -> str:
    """
    主题更新判断与生成提示词：判断是否需要更新主题，如需则生成新主题
    :param current_topic: 当前主题
    :param current_key_info: 当前关键信息块
    :param new_key_elements: 新增关键要素
    :return: 完整提示词
    """
    return """
    任务：基于当前主题、已有关键信息和新增关键要素，判断是否需要更新主题。
    判定规则：
    1. 若新增关键要素导致主题核心含义发生变化（如需求变更、话题切换），则需要更新主题。
    2. 若新增关键要素仅补充细节（不改变核心），则不需要更新主题。
    
    当前主题：{current_topic}
    已有关键信息：{current_key_info}
    新增关键要素：{new_key_elements}
    
    输出要求：
    必须严格按照以下JSON格式输出，不要添加任何额外文字！
    1. JSON 内容需严格包裹在 ```json 和 ``` 之间（代码块格式）
    2. 仅包含指定字段，不得新增其他字段
    3. need_update 为布尔值，new_topic 为字符串（不超过20字）
    4. 不需要更新时，new_topic 字段值为原主题文本   

    输出格式示例：
    ```json 
    {
        "need_update": true/false,  // 是否需要更新主题
        "new_topic": "更新后的主题文本"  // 不需要更新则填原主题，需要则填新主题（不超过20字）
    }
    ```
    """

def get_agent_response_prompt(user_input: str, current_memory: dict) -> str:
    """
    智能体回复提示词：结合当前记忆生成自然回复（流式输出用）
    :param user_input: 用户最新输入
    :param current_memory: 当前完整记忆（Topic+Info-block）
    :return: 完整提示词
    """

    # 安全取值：避免 KeyError，无对应键时返回默认值
    topic = current_memory.get("current_topic", "无")
    info_block = current_memory.get("info_block", {})  # 先获取 info_block，默认空字典
    key_info = info_block.get("key_info_block", [])     # 安全获取关键信息
    aux_info = info_block.get("aux_info_block", [])     # 安全获取辅助信息

    return f"""
    你是一个聊天智能体“小具”，需要基于用户输入和当前对话记忆，生成自然、连贯的回复。
    回复规则：
    1. 优先参考当前记忆中的主题和关键信息，确保回复与上下文相关。
    2. 忽略记忆中的噪声信息，不基于噪声进行回复。
    3. 回复要简洁友好，符合日常对话逻辑，不要暴露记忆结构细节。
    
    当前对话记忆：
    主题：{topic}
    关键信息：{key_info if key_info else "无"}
    辅助信息：{aux_info if aux_info else "无"}
    
    用户最新输入：{user_input}
    
    输出要求：
    仅输出自然语言回复文本，不要添加任何额外格式！
    """