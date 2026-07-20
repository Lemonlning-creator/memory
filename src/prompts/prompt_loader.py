"""
Prompt Configuration and Loader

This module provides a centralized way to load prompts in different languages.
To switch between Chinese and English prompts, change the PROMPT_LANGUAGE variable below.

System implementation prompts are in templates.py / templates_en.py.
Evaluation prompts are in eval_templates.py / eval_templates_en.py.
"""

# Configuration: Set the language for prompts
# Options: "zh" for Chinese, "en" for English
# Note: Experiments import directly from templates_en.py / eval_templates_en.py
# This loader is used by the frontend agent (agent.py) and defaults to Chinese
PROMPT_LANGUAGE = "zh"


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

# Persona prompts
PERSONA_SYSTEM_PROMPT = _templates.PERSONA_SYSTEM_PROMPT
PERSONA_USER_PROMPT_TEMPLATE = _templates.PERSONA_USER_PROMPT_TEMPLATE

# Profile / Persona Extraction prompts
PROFILE_EXTRACTION_SYSTEM_PROMPT = _templates.PROFILE_EXTRACTION_SYSTEM_PROMPT
PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
PERSONA_EXTRACTION_SYSTEM_PROMPT = _templates.PERSONA_EXTRACTION_SYSTEM_PROMPT
PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE

# Empathy Alignment Reasoning prompts
EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT = _templates.EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT
EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE = _templates.EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE

# Understanding Feedback prompts (Deep Empathy Updating step)
UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT = _templates.UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT
UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE = _templates.UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE

# Experiment baseline prompts
FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT = _templates.FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT
FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
SELF_MODEL_SYSTEM_PROMPT = _templates.SELF_MODEL_SYSTEM_PROMPT
SELF_MODEL_USER_PROMPT_TEMPLATE = _templates.SELF_MODEL_USER_PROMPT_TEMPLATE
PERIODIC_REBUILD_SYSTEM_PROMPT = _templates.PERIODIC_REBUILD_SYSTEM_PROMPT
PERIODIC_REBUILD_USER_PROMPT_TEMPLATE = _templates.PERIODIC_REBUILD_USER_PROMPT_TEMPLATE

# Note: Evaluation prompts (EPITOME, PROFILE_CONSISTENCY, EMOTION_SENTIMENT_EXTRACTION, etc.)
# are only used by experiments and are imported directly from eval_templates_en.py
# They are not loaded here since the frontend doesn't need them
