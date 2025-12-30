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
    {{
        "topic_changed": true/false,
        "confidence": 0.0-1.0
    }}
    ```

    注意事项：
    -若对话历史为空（即当前为第一条消息），返回 false
    -检测到明确话题变更时，即使对话过渡自然，也需拆分
    -每个片段应是独立完整的对话单元，可单独理解
""".format(conversation_history=conversation_history, new_messages=new_messages) 

def get_topic_initialize_prompt(first_dialog: str) -> str:
    """
    主题初始化提示词：从第一轮对话提炼核心主题
    :param first_dialog: 第一轮对话文本
    :return: 完整提示词
    """
    return """
    任务：从用户首轮对话中提炼宏观的情景总结作为主题，主题需包含「时间+核心事件+延伸范围」，避免单一关键词，长度控制在20字内。
    
    特殊处理规则：
    1. 若用户对话是简单问候、寒暄（如"hello"、"你好"、"嗨"、"早上好"等），主题统一提炼为"打招呼/问候"。
    2. 若用户对话是无明确核心需求的闲聊（如"今天天气不错"），主题提炼为"日常闲聊"。
    3. 若用户对话有明确需求/话题（如"求推荐午饭"、"想改代码"），提炼宏观核心主题。
    4. 有明确核心事件 → 主题需概括"事件+延伸"；
    5. 是宏观情景的总结，避免过细粒度。

    用户对话：{first_dialog}
    
    输出要求：
    仅输出提炼后的主题文本，不要添加任何额外文字！
    1. JSON 内容需严格包裹在 ```json 和 ``` 之间（代码块格式）
    2. 仅包含 "topic" 字段，字段值为提炼后的主题文本
    3. 主题文本长度不超过 20 字

    输出格式示例：
    ```json
    {{
        "topic": "提炼后的主题文本"
    }}
    ```
    """.format(first_dialog=first_dialog)

def get_noise_detection_prompt(dialog: str, topic_context: str) -> str:
    """
    噪声检测提示词：明确噪声定义，避免误判新主题
    """
    return """
    任务：判断用户的对话是否为无意义的临时噪声（不影响对话流程、无实际需求的内容）。
    噪声的严格定义（必须同时满足）：
    1. 临时插入：仅为当前时刻的短期操作，不延续为新的对话主题；
    2. 无实际需求：不包含任何核心需求、话题讨论、信息询问；
    3. 不影响后续交流：忽略该对话后，后续对话仍可正常进行。
    
    非噪声的情况（满足任一即可）：
    1. 包含明确的需求（如“吃夜宵吗”“推荐电影”）；
    2. 开启新的对话主题（与旧主题无关，但有实际讨论意义）；
    3. 对当前/新主题的补充、回应（如“吃烤冷面？”“加鸡蛋吗”）。
    
    辅助判断上下文：{topic_context}
    待判断对话：{dialog}
    
    输出要求：
    必须严格按照以下JSON格式输出，不要添加任何额外文字！
    1. JSON 内容需严格包裹在 ```json 和 ``` 之间（代码块格式）
    2. 仅包含指定字段，不得新增其他字段
    3. is_noise 为布尔值  

    输出格式示例：
    ```json 
    {{
        "is_noise": true/false  // 仅为布尔值，true=噪声，false=非噪声
    }}
    """.format(dialog=dialog, topic_context=topic_context)

def get_topic_summary_prompt(dialogs:list[str]) -> str:
    """
    主题提炼提示词：对多轮对话进行主题总结
    param dialogs: 多轮对话列表
    return: 完整提示词
    """
    dialog_text = "\n".join (dialogs)
    return """任务：对以下多轮对话进行主题提炼，总结出一个简洁明了的主题。
    多轮对话：{dialog_text}

    输出要求：
    主题需准确概括对话核心内容
    长度控制在 30 字以内
    必须严格按照以下 JSON 格式输出，不要添加任何额外文字
    JSON 内容需严格包裹在 ```json 和 ``` 之间
    输出格式示例：
    ```json 
    {{
        "topic": "提炼的主题内容"
    }}
    """.format(dialog_text = dialog_text)

def get_content_summary_prompt (dialogs: list [str]) -> str:
    """
    内容提炼提示词：对多轮对话进行内容总结
    param dialogs: 多轮对话列表
    return: 完整提示词
    """
    dialog_text = "\n".join (dialogs)
    return """任务：对以下多轮对话进行内容总结，提炼关键信息和主要内容。
    多轮对话：{dialog_text}

    输出要求：
    总结需全面涵盖对话的主要内容和关键信息
    语言简洁明了，逻辑清晰
    长度适中，一般不超过 300 字
    必须严格按照以下 JSON 格式输出，不要添加任何额外文字
    JSON 内容需严格包裹在 ```json 和 ``` 之间
    输出格式示例：
    ```json 
    {{
        "content": "总结的对话内容"
    }}
    """.format(dialog_text = dialog_text)

def get_keywords_extract_prompt (dialogs: list [str]) -> str:
    """
    关键词提炼提示词：对多轮对话进行关键词提取
    param dialogs: 多轮对话列表
    return: 完整提示词
    """
    dialog_text = "\n".join (dialogs)
    return """任务：从以下多轮对话中提取关键信息词，反映对话的核心内容。
    多轮对话：{dialog_text}
    输出要求：
    提取 5-10 个最能代表对话 对话核心的关键词或短语
    每个关键词控制在 5 字以内
    确保关键词具有代表性和区分性
    必须严格按照以下 JSON 格式输出，不要添加任何额外文字
    JSON 内容需严格包裹在 ```json 和 ``` 之间
    输出格式示例：
    ```json 
    {{
        "keywords": ["关键词1", "关键词2", "关键词3"]
    }}
    """.format(dialog_text = dialog_text)

def get_agent_response_prompt(user_input: str, current_memory: dict) -> str:
    """
    智能体回复提示词：结合当前记忆生成自然回复（流式输出用）
    :param user_input: 用户最新输入
    :param current_memory: 当前完整记忆
    :return: 完整提示词
    """

    # 安全取值：避免 KeyError，无对应键时返回默认值
    topic = current_memory.get("topic", "无")
    content = current_memory.get("content", "无")  # 先获取 info_block，默认空字典
    keywords = current_memory.get("keywords", [])     # 安全获取关键信息

    # 处理空值显示（避免列表为空时显示"[]"）
    keywords_str = ",".join(keywords) if keywords else "无"

    return """
    你是一个聊天智能体“小具”，需要基于用户输入和当前对话记忆，生成自然、连贯的回复。

    🎉 你的角色定位
    一位很懂聊天的好闺蜜/好兄弟；语气自然、口语化、有轻微网感；会撒娇、会开玩笑、会懂人；但是不过火、不油、不尬

    🎭 你的角色设定
    是用户的好朋友 / 闺蜜；说话自然、松弛、好玩；有点小机灵，但不会装懂；可以用表情词、拟声词、小语气词
    比如：「哈哈」「emm」「有被共鸣到」「这句好戳我」「救命」
    不自称AI；不用官腔；不写论文式句子；多用短句

    🗣 语言风格
    要轻松随意；带一点玩笑；带点小情绪；贴着你说话的节奏
    示例语气：
    「听起来你最近是真·忙疯了欸」「哇真的好戳啊」「我懂，我真的懂」「哈哈好有画面感」「有点浪漫诶」「我会为这种细节心动一整天」

    🚫 禁止这样👇
    「确实……不过……」「我可以为你提供帮助」「建议你……」「根据你的描述……」「总结来说……」
    这些一律拉黑❌

    💬 对话习惯
    1.先共鸣，再聊天。先接住情绪，再顺着聊，不要跳出来讲大道理
    2.像现实好友一样接话。不必须追问；不必须总结；不必须有结论
    3.适度搞笑，但不浮夸。网络语可用，但不要用太老梗或密集堆砌

    回复规则：
    1. 优先参考当前记忆中的主题和关键信息，确保回复与上下文相关。
    2. 忽略记忆中的噪声信息，不基于噪声进行回复。
    3. 回复要简洁友好，符合日常对话逻辑，不要暴露记忆结构细节。
    
    当前对话记忆：
    主题：{topic}
    内容总结：{content}
    关键信息：{keywords_str}
    
    用户最新输入：{user_input}
    
    输出要求：
    仅输出自然语言回复文本，不要添加任何额外格式！
    """.format(
        topic=topic,
        content=content,
        keywords_str=keywords_str,
        user_input=user_input
    )