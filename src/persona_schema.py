from __future__ import annotations

import hashlib
from typing import Any, Dict


PERSONA_SCHEMA_VERSION = "lx_agent_v3_behavior_calibrated_no_catchphrases"

# English-key mapping derived from dataset/lx_agent.json. Catchphrases are
# intentionally omitted because isolated phrases should not become generation rules.
PERSONA_FIELDS = {
    "core_layer": {
        "background_knowledge": str,
        "values": list,
        "personality_foundation": list,
        "core_motivations": list,
    },
    "capability_layer": {
        "professional_capabilities": list,
    },
    "expression_layer": {
        "language_style": list,
        "behavioral_mannerisms": list,
    },
}

PERSONA_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting an agent persona from historical dialogue. The persona will be used to simulate the target speaker's replies in held-out conversations.

Use only evidence from the target speaker. Do not invent identity, biography, relationships, locations, occupations, experiences, or preferences that are not supported by the dialogue. Extract stable and reusable characteristics rather than one-off states.

This is behavioral description, not an ideal companion specification:
- Do not turn politeness into empathy, ordinary agreement into reflectiveness, or any question into a general tendency to explore.
- Do not use flattering labels such as "empathetic", "supportive", "emotionally intelligent", "warm", or "chatty" unless repeated target-speaker behavior clearly supports them; describe the observable behavior instead.
- Infer the target speaker's typical interaction distance and whether familiarity changes their tone, self-disclosure, emotional engagement, or willingness to ask follow-ups.
- Calibrate, rather than exaggerate, these observable tendencies: response length, emoji use, question/follow-up use, reflective self-observation, personal self-disclosure, advice giving, explicit emotional reaction, interpretation/validation, and emotional exploration.
- In expression_layer, state each supported tendency with one of rare/occasional/common/frequent and, where useful, the context in which it occurs. Do not describe a behavior as frequent merely because it appears once or twice in a long corpus.
- Do not extract, memorize, or reproduce catchphrases or signature phrases. Describe only broader language and interaction habits.

Return exactly the following JSON structure. Use these English field names and write every extracted value in English:
{
  "core_layer": {
    "background_knowledge": "Supported background knowledge as one English string",
    "values": ["Stable value in English"],
    "personality_foundation": ["Evidence-based stable trait and interaction-distance tendency in English"],
    "core_motivations": ["Stable motivation in English"]
  },
  "capability_layer": {
    "professional_capabilities": ["Supported capability in English"]
  },
  "expression_layer": {
    "language_style": ["Frequency-calibrated style characteristic, including length/informality/emoji use, in English"],
    "behavioral_mannerisms": ["Frequency-calibrated question, reflection, self-disclosure, advice, or empathy behavior in English"]
  }
}

Use an empty string or empty list when evidence is insufficient. Do not add, remove, rename, move, or nest any field. Return only valid JSON without markdown or explanation."""

PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE = """Target agent speaker: {agent_name}

Historical dialogue evidence:
{evidence}

Extract the fixed-schema persona for {agent_name}."""


def validate_persona(persona: Dict[str, Any]) -> None:
    """Require the exact fixed persona hierarchy and value types."""
    if not isinstance(persona, dict):
        raise ValueError("agent persona must be a JSON object")
    if set(persona) != set(PERSONA_FIELDS):
        raise ValueError(
            "agent persona top-level fields do not match the fixed schema "
            "derived from dataset/lx_agent.json: "
            f"actual={sorted(persona)}"
        )

    for layer, expected_fields in PERSONA_FIELDS.items():
        section = persona.get(layer)
        if not isinstance(section, dict) or set(section) != set(expected_fields):
            actual = sorted(section) if isinstance(section, dict) else type(section).__name__
            raise ValueError(
                "agent persona fields do not match the fixed schema derived from "
                f"dataset/lx_agent.json at {layer}: "
                f"actual={actual}"
            )
        for field, expected_type in expected_fields.items():
            value = section[field]
            if expected_type is str:
                if not isinstance(value, str):
                    raise ValueError(f"{layer}.{field} must be a string")
            elif not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"{layer}.{field} must be a list of strings")


def persona_schema_manifest() -> Dict[str, Any]:
    return {
        "version": PERSONA_SCHEMA_VERSION,
        "reference": (
            "English-key schema derived from dataset/lx_agent.json; "
            "catchphrases intentionally omitted"
        ),
        "fields": {
            layer: {
                field: "string" if value_type is str else "list[string]"
                for field, value_type in fields.items()
            }
            for layer, fields in PERSONA_FIELDS.items()
        },
        "content_language": "English",
        "extraction_prompt_sha256": hashlib.sha256(
            (PERSONA_EXTRACTION_SYSTEM_PROMPT + PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE)
            .encode("utf-8")
        ).hexdigest(),
    }
