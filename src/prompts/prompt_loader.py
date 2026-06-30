"""
Prompt Configuration and Loader

This module provides a centralized way to load prompts in different languages.
To switch between Chinese and English prompts, change the PROMPT_LANGUAGE variable below.
"""

# Configuration: Set the language for prompts
# Options: "zh" for Chinese, "en" for English
PROMPT_LANGUAGE = "en"


def load_prompts():
    """
    Load prompts based on the configured language.
    
    Returns:
        module: The prompt template module (either templates or templates_en)
    """
    if PROMPT_LANGUAGE == "zh":
        from . import templates
        return templates
    elif PROMPT_LANGUAGE == "en":
        from . import templates_en
        return templates_en
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
    module = load_prompts()
    return getattr(module, prompt_name)


# Export commonly used prompts for direct import
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

# Evidence prompts
EVIDENCE_JUDGE_SYSTEM_PROMPT = _templates.EVIDENCE_JUDGE_SYSTEM_PROMPT
EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE = _templates.EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE

# EI Evaluation prompts
EI_EVALUATION_SYSTEM_PROMPT = _templates.EI_EVALUATION_SYSTEM_PROMPT
EI_EVALUATION_USER_PROMPT_TEMPLATE = _templates.EI_EVALUATION_USER_PROMPT_TEMPLATE

# Profile Extraction prompts
PROFILE_EXTRACTION_SYSTEM_PROMPT = _templates.PROFILE_EXTRACTION_SYSTEM_PROMPT
PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE

# Persona Extraction prompts
PERSONA_EXTRACTION_SYSTEM_PROMPT = _templates.PERSONA_EXTRACTION_SYSTEM_PROMPT
PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE = _templates.PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE

# Empathy Alignment Reasoning prompts
EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT = _templates.EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT
EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE = _templates.EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE

# EPITOME Evaluation prompts
EPITOME_EVALUATION_SYSTEM_PROMPT = _templates.EPITOME_EVALUATION_SYSTEM_PROMPT
EPITOME_EVALUATION_USER_PROMPT_TEMPLATE = _templates.EPITOME_EVALUATION_USER_PROMPT_TEMPLATE
