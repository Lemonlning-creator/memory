"""
Prompt Configuration and Loader

This module provides a centralized way to load prompts in different languages.
To switch between Chinese and English prompts, change the PROMPT_LANGUAGE variable below.

System implementation prompts are in templates.py / templates_en.py.
Evaluation prompts are in eval_templates.py / eval_templates_en.py.
"""

import os

# Configuration: Set the language for prompts.
# Options: "zh" for Chinese, "en" for English.
PROMPT_LANGUAGE = os.getenv("MEMORY_PROMPT_LANGUAGE", "zh").strip().lower()


def load_prompts():
    """Load system prompt templates based on the configured language."""
    if PROMPT_LANGUAGE == "zh":
        from . import templates
        return templates
    elif PROMPT_LANGUAGE == "en":
        from . import templates_en
        return templates_en
    else:
        raise ValueError(f"Unsupported prompt language: {PROMPT_LANGUAGE}. Use 'zh' or 'en'.")


def load_eval_prompts():
    """Load evaluation prompt templates based on the configured language."""
    if PROMPT_LANGUAGE == "zh":
        from . import eval_templates
        return eval_templates
    elif PROMPT_LANGUAGE == "en":
        from . import eval_templates_en
        return eval_templates_en
    else:
        raise ValueError(f"Unsupported prompt language: {PROMPT_LANGUAGE}. Use 'zh' or 'en'.")


def get_prompt(prompt_name: str):
    """
    Get a specific prompt by name.

    Args:
        prompt_name: The name of the prompt variable (e.g., 'EI_EVALUATION_SYSTEM_PROMPT')

    Returns:
        The prompt string
    """
    # Try system prompts first, then eval prompts
    module = load_prompts()
    if hasattr(module, prompt_name):
        return getattr(module, prompt_name)
    eval_module = load_eval_prompts()
    if hasattr(eval_module, prompt_name):
        return getattr(eval_module, prompt_name)
    raise AttributeError(f"Prompt '{prompt_name}' not found in system or eval templates")


# ── System prompts ──────────────────────────────────────────────────
_templates = load_prompts()
_zh_extra_templates = None
if PROMPT_LANGUAGE == "zh":
    from . import templates_zh_extra as _zh_extra_templates


def _template_attr(name: str):
    if hasattr(_templates, name):
        return getattr(_templates, name)
    if _zh_extra_templates is not None and hasattr(_zh_extra_templates, name):
        return getattr(_zh_extra_templates, name)
    raise AttributeError(f"Prompt '{name}' not found for language: {PROMPT_LANGUAGE}")

# Memory prompts
MID_TERM_MEMORY_SYSTEM_PROMPT = _templates.MID_TERM_MEMORY_SYSTEM_PROMPT
MID_TERM_MEMORY_USER_PROMPT_TEMPLATE = _templates.MID_TERM_MEMORY_USER_PROMPT_TEMPLATE
LONG_TERM_MEMORY_SYSTEM_PROMPT = _templates.LONG_TERM_MEMORY_SYSTEM_PROMPT
LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE = _templates.LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE

# Response prompts
DIRECT_RESPONSE_SYSTEM_PROMPT = _templates.DIRECT_RESPONSE_SYSTEM_PROMPT
DIRECT_RESPONSE_USER_PROMPT_TEMPLATE = _templates.DIRECT_RESPONSE_USER_PROMPT_TEMPLATE
BACKGROUND_REASONING_SYSTEM_PROMPT = _templates.BACKGROUND_REASONING_SYSTEM_PROMPT
BACKGROUND_REASONING_USER_PROMPT_TEMPLATE = _templates.BACKGROUND_REASONING_USER_PROMPT_TEMPLATE

# Profile prompts
PROFILE_EVOLUTION_SYSTEM_PROMPT = _templates.PROFILE_EVOLUTION_SYSTEM_PROMPT
PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE = _templates.PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE
USER_PROFILE_ACTIVATION_SYSTEM_PROMPT = _template_attr("USER_PROFILE_ACTIVATION_SYSTEM_PROMPT")
USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE = _template_attr("USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE")

# Persona prompts
PERSONA_SYSTEM_PROMPT = _templates.PERSONA_SYSTEM_PROMPT
PERSONA_USER_PROMPT_TEMPLATE = _templates.PERSONA_USER_PROMPT_TEMPLATE

# Profile / Persona Extraction prompts
PROFILE_EXTRACTION_SYSTEM_PROMPT = _templates.PROFILE_EXTRACTION_SYSTEM_PROMPT
PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
PERSONA_EXTRACTION_SYSTEM_PROMPT = _templates.PERSONA_EXTRACTION_SYSTEM_PROMPT
PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE

# Empathy Alignment Reasoning prompts
EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT = _template_attr("EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT")
EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE = _template_attr("EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE")

# Understanding Feedback prompts (Deep Empathy Updating step)
UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT = _template_attr("UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT")
UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE = _template_attr("UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE")

# Experiment baseline prompts
FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT = _template_attr("FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT")
FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = _template_attr("FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE")
SELF_MODEL_SYSTEM_PROMPT = _template_attr("SELF_MODEL_SYSTEM_PROMPT")
SELF_MODEL_USER_PROMPT_TEMPLATE = _template_attr("SELF_MODEL_USER_PROMPT_TEMPLATE")
PERIODIC_REBUILD_SYSTEM_PROMPT = _template_attr("PERIODIC_REBUILD_SYSTEM_PROMPT")
PERIODIC_REBUILD_USER_PROMPT_TEMPLATE = _template_attr("PERIODIC_REBUILD_USER_PROMPT_TEMPLATE")

# ── Evaluation prompts ──────────────────────────────────────────────
_eval_templates = load_eval_prompts()
from . import eval_templates_en as _eval_templates_en


def _eval_template_attr(name: str):
    if hasattr(_eval_templates, name):
        return getattr(_eval_templates, name)
    if hasattr(_eval_templates_en, name):
        return getattr(_eval_templates_en, name)
    raise AttributeError(f"Evaluation prompt '{name}' not found")

# EI Evaluation prompts
EI_EVALUATION_SYSTEM_PROMPT = _eval_template_attr("EI_EVALUATION_SYSTEM_PROMPT")
EI_EVALUATION_USER_PROMPT_TEMPLATE = _eval_template_attr("EI_EVALUATION_USER_PROMPT_TEMPLATE")

# EPITOME Evaluation prompts
EPITOME_EVALUATION_SYSTEM_PROMPT = _eval_template_attr("EPITOME_EVALUATION_SYSTEM_PROMPT")
EPITOME_EVALUATION_USER_PROMPT_TEMPLATE = _eval_template_attr("EPITOME_EVALUATION_USER_PROMPT_TEMPLATE")

# Evidence Judge prompts
EVIDENCE_JUDGE_SYSTEM_PROMPT = _eval_template_attr("EVIDENCE_JUDGE_SYSTEM_PROMPT")
EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE = _eval_template_attr("EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE")

# Cross-Conversation Consistency prompts
PROFILE_CONSISTENCY_SYSTEM_PROMPT = _eval_template_attr("PROFILE_CONSISTENCY_SYSTEM_PROMPT")
PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE = _eval_template_attr("PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE")
PERSONA_CONSISTENCY_SYSTEM_PROMPT = _eval_template_attr("PERSONA_CONSISTENCY_SYSTEM_PROMPT")
PERSONA_CONSISTENCY_USER_PROMPT_TEMPLATE = _eval_template_attr("PERSONA_CONSISTENCY_USER_PROMPT_TEMPLATE")

# State Axis prompts
CURRENT_STATE_EXTRACTION_SYSTEM_PROMPT = _eval_template_attr("CURRENT_STATE_EXTRACTION_SYSTEM_PROMPT")
CURRENT_STATE_EXTRACTION_USER_PROMPT_TEMPLATE = _eval_template_attr("CURRENT_STATE_EXTRACTION_USER_PROMPT_TEMPLATE")

# Context Axis prompts
CONTEXT_PROFILE_EXTRACTION_SYSTEM_PROMPT = _eval_template_attr("CONTEXT_PROFILE_EXTRACTION_SYSTEM_PROMPT")
CONTEXT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = _eval_template_attr("CONTEXT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE")

# Experiment evaluation prompts
EMOTION_SENTIMENT_EXTRACTION_SYSTEM_PROMPT = _eval_template_attr("EMOTION_SENTIMENT_EXTRACTION_SYSTEM_PROMPT")
EMOTION_SENTIMENT_EXTRACTION_USER_PROMPT_TEMPLATE = _eval_template_attr("EMOTION_SENTIMENT_EXTRACTION_USER_PROMPT_TEMPLATE")
TOPIC_EXTRACTION_SYSTEM_PROMPT = _eval_template_attr("TOPIC_EXTRACTION_SYSTEM_PROMPT")
TOPIC_EXTRACTION_USER_PROMPT_TEMPLATE = _eval_template_attr("TOPIC_EXTRACTION_USER_PROMPT_TEMPLATE")
INTIMACY_EXTRACTION_SYSTEM_PROMPT = _eval_template_attr("INTIMACY_EXTRACTION_SYSTEM_PROMPT")
INTIMACY_EXTRACTION_USER_PROMPT_TEMPLATE = _eval_template_attr("INTIMACY_EXTRACTION_USER_PROMPT_TEMPLATE")
