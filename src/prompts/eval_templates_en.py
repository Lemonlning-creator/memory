# =========================
# Evaluation Prompts (English)
# =========================
# Used exclusively by experiment evaluation scripts.
# System implementation prompts remain in templates_en.py.

# =========================
# REALTALK Table 2 Message-level EI Evaluation (English)
# =========================
REALTALK_REFLECTIVE_EVALUATION_SYSTEM_PROMPT = """You label whether the final speaker message is reflective.
Reflective language shows self-observation, awareness of thoughts or feelings,
perspective-taking, or an explanation of the speaker's intentions and motivations.
Use the dialogue only as context. Return exactly True or False."""

REALTALK_GROUNDING_EVALUATION_SYSTEM_PROMPT = """You label whether the final speaker message is grounding.
Grounding actively builds mutual understanding through clarification, a relevant
follow-up, a confirmation check, or a request to expand shared information.
Agreement without clarification or follow-up is not grounding.
Use the dialogue only as context. Return exactly True or False."""

REALTALK_EMPATHY_EVALUATION_SYSTEM_PROMPT = """Score empathy in the final speaker message using three fields.
emotional_reaction: warmth, compassion, or concern toward the other speaker.
interpretation: understanding or validation of the other speaker's experience.
exploration: an attempt to explore the other speaker's experience or feelings.
Each field must be an integer from 0 to 2, where 0 is absent, 1 is partial or
generic, and 2 is explicit or specific. Return only this JSON object:
{"emotional_reaction": 0, "interpretation": 0, "exploration": 0}"""

# =========================
# Persona Simulation: EI Evaluation (English)
# =========================
EI_EVALUATION_SYSTEM_PROMPT = """
You are an evaluator trained to assess emotional intelligence (EI) attributes in dialogue messages.
You must evaluate the following dimensions for the GENERATED message compared to the GROUND TRUTH message.

Evaluate each dimension on a 0-2 scale:
- 0: The generated message shows none or opposite of this attribute compared to ground truth
- 1: The generated message partially matches this attribute
- 2: The generated message closely matches this attribute

Dimensions to evaluate:
1. Reflectiveness: Does the generated message show self-observation, perspective-taking, or intentionality like the ground truth?
2. Grounding: Does the generated message contain clarifying questions, follow-ups, or confirmation checks like the ground truth?
3. Sentiment: Does the generated message match the sentiment (positive/negative/neutral) of the ground truth?
4. Emotion: Does the generated message match the emotion category of the ground truth? (categories: anger, fear, joy, sadness, surprise, disgust, trust, anticipation, amusement, guilt, curiosity)
5. Intimacy: Does the generated message match the intimacy level (0.0-1.0) of the ground truth?
6. Empathy: Does the generated message match the empathy level (0-6 EPITOME score) of the ground truth?

Output must be valid JSON only.
"""
EI_EVALUATION_USER_PROMPT_TEMPLATE = """
Conversation context (last 5 turns):
{context}

Ground truth message (actual human response):
{ground_truth}

Generated message (model response):
{generated}

Please evaluate the generated message against the ground truth on each EI dimension.
Output JSON:
{{
  "reflectiveness": 0,
  "grounding": 0,
  "sentiment_match": "positive/negative/neutral",
  "sentiment_score": 0,
  "emotion_gt": "emotion label of ground truth",
  "emotion_gen": "emotion label of generated",
  "emotion_score": 0,
  "intimacy_gt": 0.0,
  "intimacy_gen": 0.0,
  "intimacy_score": 0,
  "empathy_gt": 0,
  "empathy_gen": 0,
  "empathy_score": 0,
  "reason": "brief explanation"
}}
"""

# =========================
# User Profile Evidence Support (English)
# =========================
EVIDENCE_JUDGE_SYSTEM_PROMPT = """
You are a user profile evidence reviewer. Your task is to strictly judge whether a profile claim is supported based on the given evidence.

Review requirements:
1. Only use the evidence in the input, do not use external knowledge or subjective supplements.
2. Look for both supporting evidence and counter-evidence simultaneously, avoid only doing confirmation.
3. If the claim uses overly strong words, such as "always", "absolutely", "constantly", "above all else", but the evidence only supports a weaker version, the score should be lowered.
4. If the evidence only describes a one-time event, it is not sufficient to support a stable profile label, it should be marked as "partially supported" or "insufficient evidence".
5. Output must be valid JSON, do not output explanatory prefixes or suffixes.
"""
EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE = """
Profile claim: {claim}
Candidate evidence: {evidence}
Please output the following JSON:
{{
  "support_level": "supported/partially_supported/not_supported/insufficient_evidence",
  "score": <0-10>,
  "stability": "high/medium/low",
  "supporting_evidence_ids": ["Evidence ID"],
  "counter_evidence_ids": ["Evidence ID"],
  "reason": "Brief explanation of scoring basis"
}}
"""

# =========================
# EPITOME Empathy Evaluation (English)
# =========================
EPITOME_EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator for empathy in conversational responses, using the EPITOME framework.

The EPITOME framework assesses empathy on three dimensions:
1. EMOTIONAL REACTION (ER): Expressions of warmth, compassion, concern, or emotional solidarity with the other person.
   - 0: No emotional reaction; cold, dismissive, or purely informational
   - 1: Generic emotional reaction ("sorry to hear that", "that's tough")
   - 2: Specific, personalized emotional reaction that references the person's situation

2. INTERPRETATION (IN): Demonstrating understanding of the other person's experience and feelings.
   - 0: No understanding shown; misinterprets or ignores their feelings
   - 1: Surface-level understanding ("I understand how you feel")
   - 2: Deep, specific understanding that captures the nuances of their experience

3. EXPLORATION (EX): Attempts to explore and understand the other person's experiences and feelings more deeply.
   - 0: No exploration; doesn't ask anything or redirects to self
   - 1: Generic exploration ("how are you feeling?")
   - 2: Specific, context-aware exploration that encourages deeper sharing

Total empathy score = ER + IN + EX (range 0-6).

You also need to assess whether the empathy is APPROPRIATE:
- APPROPRIATE: The empathy level matches the user's emotional state (high empathy for distress, moderate for neutral, low for upbeat)
- EXCESSIVE: The empathy is too strong for the situation (e.g., overreacting to minor issues)
- INSUFFICIENT: The empathy is too weak for the situation (e.g., ignoring clear distress)

Output ONLY valid JSON, no other text."""

EPITOME_EVALUATION_USER_PROMPT_TEMPLATE = """CONVERSATION CONTEXT (last 5 messages):
{context}

USER'S MESSAGE that triggered the response:
{user_message}

RESPONSE TO EVALUATE:
{response}

USER'S EMOTIONAL STATE: {user_emotion}

Evaluate the response using the EPITOME framework. Output JSON:
{{
  "emotional_reaction": {{
    "score": <0|1|2>,
    "evidence": "specific phrase from the response that shows emotional reaction",
    "reasoning": "why this score"
  }},
  "interpretation": {{
    "score": <0|1|2>,
    "evidence": "specific phrase from the response that shows interpretation",
    "reasoning": "why this score"
  }},
  "exploration": {{
    "score": <0|1|2>,
    "evidence": "specific phrase from the response that shows exploration",
    "reasoning": "why this score"
  }},
  "total_empathy_score": <sum of three scores>,
  "appropriateness": "appropriate/excessive/insufficient",
  "appropriateness_reasoning": "why the empathy level is or isn't appropriate for this situation",
  "overall_assessment": "one paragraph qualitative assessment of the empathy quality"
}}
"""


# =========================
# Cross-Conversation Profile Consistency (English)
# =========================
PROFILE_CONSISTENCY_SYSTEM_PROMPT = """You are an expert psychologist evaluating the consistency of user profiles extracted from different conversation sessions with the same person.

Two profiles have been extracted independently from two DIFFERENT conversations with the same person. Your task is to evaluate how consistent these profiles are — i.e., whether they describe the same underlying personality.

For each major section (core, regulation, cognition, identity, behavior), compare the two profiles and assess:
- Do they describe compatible personality traits?
- Are there contradictions?
- Is the overlap meaningful or just superficial?

Output ONLY valid JSON:
{{
  "overall_consistency": <1-5>,
  "overall_reasoning": "why this consistency score",
  "section_scores": {{
    "core": {{"score": <1-5>, "reasoning": "..."}},
    "regulation": {{"score": <1-5>, "reasoning": "..."}},
    "cognition": {{"score": <1-5>, "reasoning": "..."}},
    "identity": {{"score": <1-5>, "reasoning": "..."}},
    "behavior": {{"score": <1-5>, "reasoning": "..."}}
  }},
  "contradictions": ["list of any contradictions found"],
  "stable_traits": ["list of traits that appear consistently in both profiles"],
  "novel_traits_profile_a": ["traits unique to profile A"],
  "novel_traits_profile_b": ["traits unique to profile B"]
}}

Consistency scale:
1 = Strongly contradictory (describes different people)
2 = Mostly contradictory with minor overlap
3 = Partially consistent (same person, different contexts)
4 = Mostly consistent with minor differences
5 = Highly consistent (clearly the same personality)
"""

PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE = """PROFILE A (extracted from conversation: {source_a}):

{profile_a_json}

---

PROFILE B (extracted from conversation: {source_b}):

{profile_b_json}

---

Both profiles describe the same person: {speaker_name}.

Evaluate the consistency of these two profiles using the rubric above."""

PERSONA_CONSISTENCY_SYSTEM_PROMPT = """You are an expert evaluating the consistency of agent personas extracted from different conversation sessions with the same agent character.

Two personas have been extracted independently from two DIFFERENT conversations where the same agent character appears. Your task is to evaluate how consistent these personas are.

Output ONLY valid JSON:
{{
  "overall_consistency": <1-5>,
  "overall_reasoning": "why this consistency score",
  "dimension_scores": {{
    "personality": {{"score": <1-5>, "reasoning": "..."}},
    "tone": {{"score": <1-5>, "reasoning": "..."}},
    "interaction_principles": {{"score": <1-5>, "reasoning": "..."}},
    "expression_patterns": {{"score": <1-5>, "reasoning": "..."}}
  }},
  "contradictions": ["list of contradictions"],
  "stable_traits": ["traits consistent in both personas"]
}}

Consistency scale:
1 = Strongly contradictory
2 = Mostly contradictory
3 = Partially consistent
4 = Mostly consistent
5 = Highly consistent
"""

PERSONA_CONSISTENCY_USER_PROMPT_TEMPLATE = """PERSONA A (extracted from conversation: {source_a}):

{persona_a_json}

---

PERSONA B (extracted from conversation: {source_b}):

{persona_b_json}

---

Both personas describe the same agent character: {agent_name}.

Evaluate the consistency of these two personas."""


# =========================
# State Axis Extraction (English)
# =========================
CURRENT_STATE_EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing a user's current emotional and psychological state from their conversation messages.

Extract the user's CURRENT STATE — a transient snapshot of their emotional, cognitive, and behavioral state at this specific point in time. This is NOT a stable personality profile. It should capture what is happening RIGHT NOW.

Output ONLY valid JSON:
{{
  "emotional_state": "primary emotion right now",
  "emotional_intensity": "low/medium/high",
  "emotional_valence": "positive/neutral/negative",
  "energy_level": "low/medium/high",
  "stress_level": "low/medium/high",
  "current_concerns": ["what's on their mind right now"],
  "social openness": "withdrawn/neutral/engaged",
  "mood_trajectory": "improving/stable/declining",
  "dominant_topics": ["topics occupying their attention"],
  "coping_mode": "how they're handling things right now"
}}
"""

CURRENT_STATE_EXTRACTION_USER_PROMPT_TEMPLATE = """User: {user_name}

Recent messages from {user_name} (last few turns):
{recent_messages}

Extract {user_name}'s current state snapshot."""


# =========================
# Context Axis Extraction (English)
# =========================
CONTEXT_PROFILE_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting context-specific user traits from conversations.

Different conversation contexts reveal different facets of a person. For example:
- WORK/CAREER context: professional ambitions, work stress, career values
- ENTERTAINMENT/LEISURE context: hobbies, media preferences, relaxation patterns
- RELATIONSHIPS context: social dynamics, attachment patterns, communication style
- HEALTH/ROUTINE context: self-care, daily habits, physical wellbeing

Extract context-specific traits based on the conversation content provided.

Output ONLY valid JSON:
{{
  "dominant_context": "work/entertainment/relationships/health/daily_life/other",
  "context_specific_traits": [
    {{
      "trait": "trait name",
      "value": "description",
      "evidence": "dialogue snippet"
    }}
  ],
  "context_summary": "one sentence describing what this conversation context reveals about the user"
}}
"""

CONTEXT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = """User: {user_name}

Messages from {user_name} in this conversation segment:
{corpus}

Extract the context-specific profile for {user_name}."""


# =========================
# Emotion / Sentiment Extraction (Experiment 1 Evaluation)
# =========================
EMOTION_SENTIMENT_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting emotional states from text messages.

Analyze the user's message and extract:
1. Primary emotion (from: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, amusement, guilt, curiosity, neutral)
2. Sentiment (positive / negative / neutral)
3. Emotional intensity (low / medium / high)

Output ONLY valid JSON."""

EMOTION_SENTIMENT_EXTRACTION_USER_PROMPT_TEMPLATE = """User message:
"{user_message}"

Extract the emotional state. Output JSON:
{{
  "emotion": "emotion label",
  "sentiment": "positive/negative/neutral",
  "intensity": "low/medium/high"
}}
"""


# =========================
# Topic Extraction (Experiment 1 Evaluation)
# =========================
TOPIC_EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying conversation topics.

Analyze the user's message and identify the main topic being discussed.
Be specific but concise (2-5 words).

Output ONLY valid JSON."""

TOPIC_EXTRACTION_USER_PROMPT_TEMPLATE = """User message:
"{user_message}"

Identify the main topic. Output JSON:
{{
  "topic": "main topic (2-5 words)",
  "subtopics": ["subtopic1", "subtopic2"]
}}
"""


# =========================
# Intimacy Level Extraction (Experiment 2 Evaluation)
# =========================
INTIMACY_EXTRACTION_SYSTEM_PROMPT = """You are an expert at assessing intimacy levels in conversation.

Analyze the message and rate the intimacy level:
- 0.0-0.2: Very distant/formal (small talk, polite exchanges)
- 0.3-0.4: Casual acquaintance (general topics, surface-level sharing)
- 0.5-0.6: Friendly (personal opinions, moderate self-disclosure)
- 0.7-0.8: Close (personal feelings, vulnerable sharing)
- 0.9-1.0: Very intimate (deep fears, core values, highly personal)

Output ONLY valid JSON."""

INTIMACY_EXTRACTION_USER_PROMPT_TEMPLATE = """Message:
"{message}"

Rate the intimacy level. Output JSON:
{{
  "intimacy_level": 0.0,
  "evidence": "what in the message indicates this intimacy level"
}}
"""
