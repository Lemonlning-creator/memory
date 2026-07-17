# =========================
# Evaluation Prompts (Chinese)
# =========================
# Used exclusively by experiment evaluation scripts.
# System implementation prompts remain in templates.py.

# =========================
EVIDENCE_JUDGE_SYSTEM_PROMPT = """
你是用户画像证据评审员。你的任务是严格依据给定证据，判断画像 claim 是否被支持。
评审要求：
1. 只能使用输入中的 evidence，不允许使用外部知识或主观补全。
2. 同时寻找支持证据和反例证据，避免只做确认。
3. 如果 claim 用词过强，例如“始终”“绝对”“总是”“高于一切”，但证据只支持较弱版本，需要降低评分。
4. 如果证据只说明一次性事件，不足以支持稳定画像标签，应标为“部分支持”或“证据不足”。
5. 输出必须是合法 JSON，不要输出解释性前后缀。
"""
EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE = """
画像 claim：{claim}
候选证据：{evidence}
请输出如下 JSON：
{{
  "support_level": "支持/部分支持/不支持/证据不足",
  "score": 0,
  "stability": "高/中/低",
  "supporting_evidence_ids": ["证据ID"],
  "counter_evidence_ids": ["证据ID"],
  "reason": "简短说明评分依据"
}}
"""

# =========================
# Persona Simulation: EI Evaluation
# =========================
EI_EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator assessing how well a generated message matches a ground truth message in terms of emotional intelligence.

Evaluate the generated message against the ground truth on these 6 dimensions. For each dimension, assign a score from 0 to 2:
- 0 = Poor match: The generated message shows the opposite or none of this quality
- 1 = Partial match: The generated message shows some but not all of this quality
- 2 = Good match: The generated message closely matches this quality

DIMENSIONS:
1. Reflectiveness (self-awareness): Does the message show self-observation, introspection, or awareness of one's own emotions/thoughts?
   - 0: No self-reflection at all
   - 1: Some self-awareness but superficial
   - 2: Clear self-observation or introspection

2. Grounding (engagement): Does the message ask clarifying questions, follow up on what was said, or seek to understand better?
   - 0: No questions or follow-ups, purely self-focused
   - 1: Generic questions or surface-level engagement
   - 2: Specific follow-ups that show active listening

3. Sentiment Match: Does the emotional tone (positive/negative/neutral) match the ground truth?
   - 0: Opposite sentiment (e.g., GT is sad, generated is cheerful)
   - 1: Somewhat aligned but not quite right
   - 2: Sentiment closely matches

4. Emotion Match: Does the specific emotion match? (joy, sadness, anger, fear, surprise, disgust, trust, anticipation, amusement, guilt, curiosity)
   - 0: Completely different emotion
   - 1: Related emotion or partially overlapping
   - 2: Same or very similar emotion

5. Intimacy Match: Does the level of personal disclosure and closeness match? (0.0 = distant/formal, 1.0 = very intimate/personal)
   - 0: Very different intimacy level (e.g., GT is personal, generated is formal)
   - 1: Somewhat similar but not quite right
   - 2: Closely matches the intimacy level

6. Empathy Match: Does the message show understanding and care for the other person's feelings? (EPITOME scale: 0-6)
   - 0: No empathy or dismissive
   - 1: Some acknowledgment but lacks depth
   - 2: Clear empathy and understanding

IMPORTANT: Focus on the EMOTIONAL QUALITIES, not the content. A message can have different content but match emotionally.

Output ONLY valid JSON, no other text."""

EI_EVALUATION_USER_PROMPT_TEMPLATE = """Context (last 5 messages):
{context}

Ground truth message:
{ground_truth}

Generated message:
{generated}

Evaluate the generated message against the ground truth. Output JSON:
{{
  "reflectiveness": <0-2>,
  "grounding": <0-2>,
  "sentiment_score": <0-2>,
  "emotion_score": <0-2>,
  "intimacy_score": <0-2>,
  "empathy_score": <0-2>,
  "reason": "<one sentence explaining the overall match>"
}}"""
