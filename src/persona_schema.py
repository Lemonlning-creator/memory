from __future__ import annotations

import hashlib
from typing import Any, Dict


PERSONA_SCHEMA_VERSION = "lx_agent_v1"

# English-key mapping of the hierarchy and fields in dataset/lx_agent.json.
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
        "catchphrases": list,
        "behavioral_mannerisms": list,
    },
}

PERSONA_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting an agent persona from historical dialogue. The persona will be used to simulate the target speaker's replies.

Use only evidence from the target speaker. Do not invent identity, biography, relationships, locations, occupations, experiences, preferences, or catchphrases that are not supported by the dialogue. Extract stable and reusable characteristics rather than one-off states.

Return exactly the following JSON structure. Use these English field names and write every extracted value in English:
{
  "core_layer": {
    "background_knowledge": "Supported background knowledge as one English string",
    "values": ["Stable value in English"],
    "personality_foundation": ["Stable personality trait in English"],
    "core_motivations": ["Stable motivation in English"]
  },
  "capability_layer": {
    "professional_capabilities": ["Supported capability in English"]
  },
  "expression_layer": {
    "language_style": ["Stable language-style characteristic in English"],
    "catchphrases": ["Repeated or strongly supported expression in English"],
    "behavioral_mannerisms": ["Stable interaction behavior in English"]
  }
}

Use an empty string or empty list when evidence is insufficient. Do not add, remove, rename, move, or nest any field. Return only valid JSON without markdown or explanation."""

PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE = """Target agent speaker: {agent_name}

Historical dialogue evidence:
{evidence}

Extract the fixed-schema persona for {agent_name}."""


def validate_persona(persona: Dict[str, Any]) -> None:
    """Require the exact lx_agent_v1 hierarchy and value types."""
    if not isinstance(persona, dict):
        raise ValueError("agent persona must be a JSON object")
    if set(persona) != set(PERSONA_FIELDS):
        raise ValueError(
            "agent persona top-level fields do not match the English-key mapping "
            "of dataset/lx_agent.json: "
            f"actual={sorted(persona)}"
        )

    for layer, expected_fields in PERSONA_FIELDS.items():
        section = persona.get(layer)
        if not isinstance(section, dict) or set(section) != set(expected_fields):
            actual = sorted(section) if isinstance(section, dict) else type(section).__name__
            raise ValueError(
                "agent persona fields do not match the English-key mapping of "
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
        "reference": "English-key mapping of dataset/lx_agent.json",
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
