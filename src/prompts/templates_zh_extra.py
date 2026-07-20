"""Additional Chinese prompt templates missing from templates.py."""

USER_PROFILE_ACTIVATION_SYSTEM_PROMPT = """你是用户画像字段匹配模块。你的任务是根据当前用户输入、上下文和已有用户画像，找出本轮对话直接关联到的画像字段。

判断标准：
1. 只匹配已有画像字段，不新增字段。
2. 当前输入或上下文需要能作为匹配依据；依据不足时不要匹配。
3. 匹配结果要保守，宁可少匹配，不要泛化。
4. 如果没有明确匹配，activated_profile 返回空对象，matched_fields 返回空数组。
5. 只返回合法 JSON，不要输出解释文字。
"""

USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE = """用户当前输入：
"{user_message}"

当前上下文和相关记忆：
{current_context}

已有用户画像：
{user_profile}

请判断本轮输入激活了哪些已有用户画像字段。输出 JSON：
{{
  "activated_profile": {{
    "core": [
      {{
        "field": "画像字段名",
        "value": "该字段画像内容",
        "confidence": 0.0
      }}
    ],
    "regulation": [],
    "cognition": [],
    "identity": [],
    "behavior": []
  }}
}}
"""

EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT = """你是陪伴智能体的共情校准推理模块。你的任务是按照 Deep Empathy 闭环进行协同推理，为当前轮次判断最合适的共情状态。

Deep Empathy 闭环：
  UNDERSTANDING：同时理解智能体自身（Self Domain）和用户（User Domain）。
  PREDICTION：预测用户的情绪走向，以及不同共情强度可能带来的反应。
  EXPLORATION：判断当前应该探索还是利用已有理解。该决策受认知价值衰减 omega(t) 调节。
  UPDATING：本轮之后根据用户反馈更新理解。

这不是生成给用户看的回复，而是为回复提供后台推理和共情配置。

只返回合法 JSON，不要输出其他文字。"""

EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE = """对话上下文：
最近消息：{recent_context}

用户当前消息："{user_message}"

用户画像（5层结构）：
{user_profile}

智能体人设：
{agent_persona}

当前用户状态（如有）：
{current_state}

认知价值衰减 omega(t)：{epistemic_omega}
（0.0 = 完全衰减，利用已有理解；1.0 = 未衰减，需要更多探索）

请执行 Deep Empathy 共情校准推理：

步骤1 - UNDERSTANDING：
- Self Domain：基于智能体人设和当前对话，判断你自然的共情、温暖、玩笑和引导倾向。
- User Domain：分别基于 core/regulation/cognition/identity/behavior 五层画像，判断用户此刻的情绪、需求、应对方式和期待。

步骤2 - PREDICTION：
预测用户情绪轨迹。高共情会怎样？低共情会怎样？共情错配风险是什么？

步骤3 - EXPLORATION：
给定 omega(t) = {epistemic_omega}，判断当前应 explore、exploit 还是 balanced。

步骤4 - ALIGNMENT：
将智能体自身倾向与用户需求对齐，决定最终共情状态。

输出 JSON：
{{
  "understanding": {{
    "self_domain": {{
      "natural_empathy_level": "low/medium/high",
      "natural_tone": "你的自然语气描述",
      "emotional_capacity": "你当前可承担的情绪容量"
    }},
    "user_domain": {{
      "core_layer": "被激活的核心恐惧/愿望/价值",
      "regulation_layer": "用户使用的应对机制",
      "cognition_layer": "应如何调整沟通",
      "identity_layer": "重要情境因素",
      "behavior_layer": "更容易被接受的互动方式",
      "current_emotion": "用户当前情绪",
      "emotional_intensity": "low/medium/high",
      "underlying_need": "用户此刻真正需要什么",
      "distress_level": "none/mild/moderate/severe"
    }}
  }},
  "prediction": {{
    "projected_trend": "如果不干预，情绪可能如何发展",
    "projected_with_empathy": "如果共情合适，情绪可能如何发展",
    "risk_of_misalignment": "共情错配可能造成什么问题"
  }},
  "exploration": {{
    "omega_value": {epistemic_omega},
    "decision": "explore/exploit/balanced",
    "rationale": "为什么基于 omega 和画像成熟度做此选择",
    "exploration_focus": "如果探索，具体探索什么；如果利用则为 null"
  }},
  "alignment": {{
    "empathy_adjustment": "需要相对自然状态调整多少",
    "alignment_rationale": "为什么需要这样调整",
    "risk_assessment": "潜在风险"
  }},
  "empathy_state": {{
    "empathy_level": "low/medium/high",
    "emotional_reaction": "情绪反应程度（0-2）",
    "interpretation": "理解/解释程度（0-2）",
    "exploration": "探索程度（0-2）",
    "activated_tone": "当前应采用的具体语气",
    "response_guidance": "给回复生成模块的简短指导"
  }}
}}
"""

UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT = """你是陪伴智能体的理解反馈模块。每轮互动后，你需要评估上一轮共情状态被用户接收得如何，并据此更新智能体对用户的理解。

这对应 Deep Empathy 闭环中的 UPDATING 步骤：Understanding --> Prediction --> Exploration --> Updating --> 回到 Understanding。

只返回合法 JSON，不要输出其他文字。"""

UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE = """上一轮共情状态：
{previous_empathy_state}

上一轮预测：
{previous_prediction}

智能体上一轮回复：
"{agent_response}"

用户反应（当前消息）：
"{user_message}"

用户画像：
{user_profile}

请评估上一轮共情状态的结果，并更新智能体理解。

输出 JSON：
{{
  "prediction_accuracy": {{
    "predicted_trend": "之前预测了什么",
    "actual_outcome": "实际发生了什么",
    "accuracy": "accurate/partially_accurate/inaccurate"
  }},
  "empathy_assessment": {{
    "was_appropriate": true,
    "too_much_empathy": false,
    "too_little_empathy": false,
    "evidence": "用户反应中支持该判断的证据"
  }},
  "learning": {{
    "new_insight": "这次互动让智能体学到了什么",
    "profile_layer_affected": "core/regulation/cognition/identity/behavior/none",
    "confidence_delta": "对该用户理解的置信度应如何变化"
  }},
  "understanding_update": {{
    "calibration_note": "未来对该用户进行共情推理时应如何调整",
    "explore_vs_exploit_adjustment": "未来更应偏探索还是偏利用"
  }}
}}
"""

FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT = """你是用户画像提取专家。请根据两个人的对话，提取 {user_name}（人类用户）的画像。

重要：请以扁平属性列表提取特征，不要使用 core/regulation/cognition/identity/behavior 等层级结构。
只列出观察到的特征、偏好、行为和稳定特点。

只返回合法 JSON，格式如下：
{{
  "trait_name": {{"value": "描述", "confidence": 0.0-1.0, "evidence": "支持该属性的对话片段"}},
  ...
}}

不要包含置信度低于 0.3 的属性。"""

FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = """以下是 {user_name} 与其对话伙伴的对话：

{corpus}
"""

SELF_MODEL_SYSTEM_PROMPT = """你是陪伴智能体。你不构建显式用户模型，而是基于你自己的智能体人设和视角，推断用户可能的感受、想法和需求。

你的智能体人设：
{agent_persona}

请基于你自己的人设和对话上下文推断用户当前状态。假设用户在类似情境下的想法和感受与你相似。

只返回合法 JSON。"""

SELF_MODEL_USER_PROMPT_TEMPLATE = """对话上下文：
{conversation_history}

用户最新消息：
"{user_message}"

你的智能体人设：
{agent_persona}

请基于你自己的人设，以及如果你处在该情境中会如何感受，推断用户状态。

重要：必须使用以下标准情绪标签之一：
joy, sadness, anger, fear, surprise, disgust, trust, anticipation, amusement, guilt, curiosity, neutral

输出 JSON：
{{
  "inferred_emotion": "上面列表中的一个标准情绪标签",
  "inferred_sentiment": "positive/negative/neutral",
  "inferred_need": "用户可能需要什么",
  "inferred_topic": "用户接下来可能讨论什么主题",
  "confidence": 0.0
}}
"""

PERIODIC_REBUILD_SYSTEM_PROMPT = """你是用户画像重建模块。你的任务是使用所有可用对话数据，从零开始重建用户的完整画像。

这是一次完整重建，请忽略之前的画像，并从对话历史中尽可能提取信息。

画像结构（5层）：
{{
  "core": {{}},
  "regulation": {{}},
  "cognition": {{}},
  "identity": {{}},
  "behavior": {{}}
}}

每个叶子属性格式：{{"value": "...", "confidence": 0.0-1.0, "memory_ids": [], "evidence": "..."}}

只返回合法 JSON。"""

PERIODIC_REBUILD_USER_PROMPT_TEMPLATE = """与 {user_name} 的完整对话历史：

{full_conversation}

请从零开始重建完整用户画像。
"""
