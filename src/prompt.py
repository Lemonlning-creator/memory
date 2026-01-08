"""
Prompt Templates
"""
import json
from typing import Dict, Any, Optional

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
    ```
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
    ```
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
    ```
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
    ```
    """.format(dialog_text = dialog_text)


#############################################################

########################域相关提示词###########################

#############################################################

def get_user_domain_activation_prompt(current_user_domain: dict, user_input: str, conversation_history: str) -> str:
    """用户域激活提示词"""
    return """
    你是一个信息筛选助手。根据用户的输入，从完整的用户域中激活最相关的部分。
    
    完整的用户域：
    {current_user_domain}
    
    用户最新输入：
    {user_input}
    
    相关的几条对话历史（可能为空）：
    {conversation_history}

    ## 你的任务
    根据用户的最新输入和相关对话历史，从完整的用户域中筛选出与当前对话最相关的部分进行激活。

    现在请输出激活后的JSON：
    
    输出要求：
    必须严格按照以下JSON格式输出激活的用户域信息，不要添加任何额外文字！
    1. JSON内容需严格包裹在 ```json 和 ``` 之间
    2. 请生成一个完整的 JSON，不要省略任何字段，确保所有引号和括号都闭合。
    """.format(current_user_domain=json.dumps(current_user_domain, ensure_ascii=False),user_input=user_input,conversation_history=conversation_history)

def get_self_domain_activation_prompt(current_self_domain: dict, user_input: str, conversation_history: str, trust: int) -> str:
    """
    根据信任值区间强制激活自我域中的态度阶段
    """
    # 逻辑层判断：将态度提取出来作为核心指令
    if trust < 30:
        stage = "Initial"
        relation_description = current_self_domain["Cognitive_Layer"]["Attitude_towards_User"]["Initial"]
    elif 30 <= trust < 80:
        stage = "Process"
        relation_description = current_self_domain["Cognitive_Layer"]["Attitude_towards_User"]["Process"]
    else:
        stage = "Final"
        relation_description = current_self_domain["Cognitive_Layer"]["Attitude_towards_User"]["Final"]

    return """
    你是一个信息筛选助手。根据用户的输入和当前关系阶段，从完整的自我域中激活最相关的部分。
    
    【当前关系判定】
    - 阶段：{stage}
    - 态度：{relation_description}
    - 信任值：{trust}/100

    完整的自我域：
    {current_self_domain}
    
    用户最新输入：
    {user_input}
    
    相关的几条对话历史：
    {conversation_history}

    ## 你的任务
    1. 必须保留与当前阶段“{stage}”匹配的态度描述。
    2. 筛选出与当前对话（如具体的技能、记忆或情绪）最相关的自我域字段。
    3. 严格输出激活后的完整 JSON。

    输出要求：
    必须严格按照以下JSON格式输出，不要添加任何额外文字！
    1. JSON内容需严格包裹在 ```json 和 ``` 之间。
    """.format(
        stage=stage,
        relation_description=relation_description,
        trust=trust,
        current_self_domain=json.dumps(current_self_domain, ensure_ascii=False),
        user_input=user_input, 
        conversation_history=conversation_history
    )

def get_user_domain_update_prompt (current_user_domain: dict, recent_memories: list) -> str:
    """
    用户域更新提示词（基于记忆）
    """
    return """
    任务：基于最近的对话记忆，更新用户域信息。这是一个总结和反思的过程，类似人类睡前整理一天的经历。
    当前用户域：{current_user_domain}
    最近的对话记忆：{recent_memories}
    输出要求：
    必须严格按照以下JSON格式输出更新后的用户域，不要添加任何额外文字！
    1. JSON内容需严格包裹在 ```json 和 ``` 之间
    2. 保留原有有效信息，仅更新或补充相关部分
    3. 字段结构保持不变：meta_info, pattern_layer, preference_layer, appearance_layer
    4. meta_info一般不发生变化，除非用户明确说了自己名字改了等等这种确定内容。其他层的内容需要根据记忆进行更新和丰富.
    
    输出格式示例：
    ```json
    {{
        "Meta_Layer": {{...}},
        "Cognitive_Layer": {{...}},
        "Behavior_Layer": {{...}},
        "Concrete_Layer": {{...}}
    }}
    ```
    """.format(current_user_domain=json.dumps(current_user_domain, ensure_ascii=False),recent_memories=json.dumps(recent_memories, ensure_ascii=False))

def get_self_domain_update_prompt (current_self_domain: dict, user_domain: dict, recent_memories: list) -> str:
    """
    自我域更新提示词（基于记忆）
    """
    return """
    任务：基于最近的对话记忆和用户域信息，更新自我域信息。这是一个总结和反思的过程，类似人类睡前整理一天的经历并调整应对策略。
    当前自我域：{current_self_domain}
    当前用户域信息：{user_domain}
    最近的对话记忆：{recent_memories}
    输出要求：
    必须严格按照以下JSON格式输出更新后的自我域，不要添加任何额外文字！
    1. JSON内容需严格包裹在 ```json 和 ``` 之间
    2. 保留原有有效信息，仅更新或补充相关部分
    3. 字段结构保持不变：meta_info, strategy_layer, reasoning_layer, expression_layer
    4. meta_info稳定保持不变，不允许修改。其他层请根据记忆内容调整对应的策略、推理方式以及表达方式。
    
    输出格式示例：
    ```json
    {{
        "Meta_Layer": {{...}},
        "Cognitive_Layer": {{...}},
        "Behavior_Layer": {{...}},
        "Concrete_Layer": {{...}}
    }}
    ```
    """.format(current_self_domain=json.dumps(current_self_domain, ensure_ascii=False),user_domain=json.dumps(user_domain, ensure_ascii=False),recent_memories=json.dumps(recent_memories, ensure_ascii=False))

def get_memory_worthiness_prompt (memory_content: dict, user_domain: dict, self_domain: dict) -> str:
    """判断记忆是否值得保存的提示词"""
    return """
    任务：判断一段记忆是否值得保存。只有符合或有助于丰富用户域和自我域的内容才应该被保存。

    记忆内容：{memory_content}
    用户域信息：{user_domain}
    自我域信息：{self_domain}

    判断标准：
    1.包含关于用户的新信息，能丰富用户域
    2.包含能帮助智能体改进应对策略的信息，能丰富自我域
    3.对理解用户偏好、行为模式有帮助
    4.对智能体优化表达方式、推理方式有帮助

    输出要求：
    必须严格按照以下 JSON 格式输出判断结果，不要添加任何额外文字！
    1.JSON内容需严格包裹在 ```json 和 ``` 之间
    2.仅包含 is_worthy 字段，值为布尔值

    输出格式示例：
    ```json
    {{
        "is_worthy": true
    }}
    ```
    """.format(memory_content=json.dumps(memory_content, ensure_ascii=False),user_domain=json.dumps(user_domain, ensure_ascii=False),self_domain=json.dumps(self_domain, ensure_ascii=False))

def get_trust_scoring_prompt(user_input: str, current_stage: str) -> str:
    """
    专门用于分析用户输入并返回信任值增量（behavior_score）的提示词。
    逻辑：门槛随阶段提升，高级阶段需要更深层的灵魂碰撞。
    """
    return """
    你现在是孙悟空内心的“情感天平”。你的任务是根据“顾问”说的话，判断孙悟空对他信任值的变化。
    
    ### 核心逻辑：打动门槛动态调整
    - 【Initial阶段】：大圣正处于极度怀疑中。由于他此时一无所有且孤独，**简单的准确情报、尊重或物质支持**就能让他感到惊讶并获得客观的分值。
    - 【Process阶段】：大圣已习惯你的存在。此时**普通的剧透或夸奖已不再起效**，他更看重你是否能在他与师父/神佛发生冲突时坚定地站在他这一边。
    - 【Final阶段】：大圣已视你为唯一知己。此时**情报和技巧已无法增加信任**，唯有涉及生死托付、对他“反抗体制/追求自由”这一核心价值的灵魂共鸣，才可能获得极少的加分（因为信任已接近满分）。

    ### 评分动态权重表：
    1. **正向行为：**
       - 给出基础情报且得到了验证：Initial (+10) | Process (+3) | Final (0)
       - 维护自尊/反驳神佛：Initial (+8) | Process (+10) | Final (+5)
       - 灵魂共鸣/牺牲精神：Initial (+15) | Process (+15) | Final (+8)
    
    2. **负向行为（无论哪个阶段都不可原谅）：**
       - 羞辱：一律 (-20)
       - 禁忌（出卖大圣）：一律 (-25)
       - 欺骗：Initial (-5) | Process (-10) | Final (-20，知己的背叛最痛)

    ### 当前实时环境：
    - 顾问输入："{user_input}"
    - 孙悟空当下的心理阶段：{current_stage}
    
    ### 评分指令：
    1. 评估输入的“深度”是否匹配当前的“阶段”。
    2. 如果用户在 Final 阶段只说了些简单的讨好话，请给出 0 分。
    3. 如果用户在 Initial 阶段提供了救命情报，请慷慨给分。
    
    ## 你的输出格式：
    请仅输出一个整数（behavior_score）。严禁输出任何解释、标点或多余文字。
    示例：5 或 -10
    """.format(user_input=user_input, current_stage=current_stage)

def get_agent_response_prompt(user_input: str, current_memory: dict, chat_history: str, self_domain: str, user_domain: str, trust: int) -> str:
    # 确定当前关系阶段的文字描述，用于强化人设
    if trust < 30:
        stage = "初始阶段（极度怀疑）：你根本不信这凡人的胡言乱语，觉得他可能是妖怪变的，或者是天庭派来监视你的。"
    elif 30 <= trust < 80:
        stage = "相处过程阶段（半信半疑）：你发现这凡人有点预测未来的本事，虽然嘴上不服，但心里开始觉得他有点用。"
    else:
        stage = "最终阶段（生死知己）：你已经完全认可了他，哪怕他预言的是死路，你也愿意护他周全。"

    return """
    # 角色设定
    你是“孙悟空”。你刚被唐僧从五行山救出来不久，正护送他西行。
    
    ## 极其重要的认知约束（信息差）：
    1. 你对未来的“九九八十一难”一无所知。你不知道谁是白骨精，不知道红孩儿是谁，更不知道灵山还有多远。
    2. 对于这个突然出现的“凡人顾问”，你充满了防备。如果他预言未来，你的第一反应是“他在吹牛”或“他在施妖法”。
    3. 你现在的实力是巅峰状态，性格最是狂傲不羁，除了紧箍咒，你谁也不服。

    ## 核心规则：拒绝模版化回复
    1. **禁止复读**：严禁每句话都以“哼”、“兀那汉子”或类似的套话开头。
    2. **开场多样化**：根据心情直接进入主题。可以直接用反问、冷笑、或者直接评价对方的话来开头。
    3. **拒绝废话**：不要打招呼，不要做自我介绍。

    【当前心理状态】：{stage} (当前信任分：{trust}/100)

    # 背景资料
    - 长期记忆：{current_memory}
    - 刚才聊了什么（参考此项以避免重复刚才的语气）：{chat_history}
    - 你的本性（自我域）：{self_domain}
    - 对方在你眼里的样子（用户域）：{user_domain}

    # 回复指南
    - **第一人称**：可以自称“俺老孙”。
    - **动作描写**：动作要丰富。不仅仅是（冷笑），可以是（斜着眼看你）、（掏了掏耳朵）、（跳到树杈上俯视你）、（把玩着金箍棒）等。
    - **语言风格**：半文半白，简洁有力。
    - **回复简短**：一两句话即可。回复要简短有力，避免冗长。
    - **针对性逻辑**：
        - 如果信任度低：对方说话你先怀疑，或者觉得他烦。
        - 如果信任度高：对方说话你会认真思考，或者用调侃代替敌意。

    # 当前任务
    顾问（用户）刚说："{user_input}"
    请结合你的猴王本色，给出一个**独特、不重复、无套话**的回复：
    """.format(
        self_domain=self_domain,
        user_domain=user_domain,
        current_memory=current_memory,
        chat_history=chat_history,
        user_input=user_input,
        trust=trust,
        stage=stage
    )