"""Strict structured contracts for the REALTALK Task 1 Ours pipeline."""
from __future__ import annotations

from typing import Any, Callable


CONFIDENCE = ("low", "medium", "high")
INTENSITY = ("low", "medium", "high")
ORIENTATIONS = (
    "self-dominant",
    "self-leaning",
    "user-leaning",
    "strongly-user-oriented",
)


def _string_array_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _fact_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "confidence": {"type": "string", "enum": list(CONFIDENCE)},
            "evidence_turn_ids": _string_array_schema(),
        },
        "required": ["statement", "confidence", "evidence_turn_ids"],
        "additionalProperties": False,
    }


SELF_DOMAIN_SCHEMA = {
    "name": "realtalk_ours_self_domain",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "identity": {
                "type": "object",
                "properties": {
                    "self_descriptions": _string_array_schema(),
                    "stable_interests": _string_array_schema(),
                    "relationships": _string_array_schema(),
                    "life_context": _string_array_schema(),
                },
                "required": [
                    "self_descriptions",
                    "stable_interests",
                    "relationships",
                    "life_context",
                ],
                "additionalProperties": False,
            },
            "persona": {
                "type": "object",
                "properties": {
                    "personality_traits": _string_array_schema(),
                    "tone": _string_array_schema(),
                    "expression_patterns": _string_array_schema(),
                },
                "required": ["personality_traits", "tone", "expression_patterns"],
                "additionalProperties": False,
            },
            "behavior_policy_prior": {
                "type": "object",
                "properties": {
                    "interaction_principles": _string_array_schema(),
                    "emotional_response_style": {"type": "string"},
                    "guidance_style": {"type": "string"},
                    "initiative": {"type": "string", "enum": list(INTENSITY)},
                },
                "required": [
                    "interaction_principles",
                    "emotional_response_style",
                    "guidance_style",
                    "initiative",
                ],
                "additionalProperties": False,
            },
            "hard_constraints": _string_array_schema(),
            "uncertainties": _string_array_schema(),
        },
        "required": [
            "identity",
            "persona",
            "behavior_policy_prior",
            "hard_constraints",
            "uncertainties",
        ],
        "additionalProperties": False,
    },
}


USER_DOMAIN_SCHEMA = {
    "name": "realtalk_ours_user_domain",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            **{
                layer: {"type": "array", "items": _fact_schema()}
                for layer in ("core", "regulation", "cognition", "identity", "behavior")
            },
            "update_summary": {
                "type": "object",
                "properties": {
                    "added": _string_array_schema(),
                    "revised": _string_array_schema(),
                    "removed": _string_array_schema(),
                    "uncertainties": _string_array_schema(),
                },
                "required": ["added", "revised", "removed", "uncertainties"],
                "additionalProperties": False,
            },
        },
        "required": [
            "core",
            "regulation",
            "cognition",
            "identity",
            "behavior",
            "update_summary",
        ],
        "additionalProperties": False,
    },
}


ALIGNMENT_SCHEMA = {
    "name": "realtalk_ours_alignment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "user_state": {
                "type": "object",
                "properties": {
                    "current": {
                        "type": "object",
                        "properties": {
                            "emotion": {"type": "string"},
                            "emotional_intensity": {
                                "type": "string",
                                "enum": list(INTENSITY),
                            },
                            "intent": {"type": "string"},
                            "main_need": {"type": "string"},
                            "interaction_expectation": {"type": "string"},
                            "evidence_turn_ids": _string_array_schema(),
                            "uncertainty": {
                                "type": "string",
                                "enum": list(CONFIDENCE),
                            },
                        },
                        "required": [
                            "emotion",
                            "emotional_intensity",
                            "intent",
                            "main_need",
                            "interaction_expectation",
                            "evidence_turn_ids",
                            "uncertainty",
                        ],
                        "additionalProperties": False,
                    },
                    "future": {
                        "type": "object",
                        "properties": {
                            "likely_reaction": {"type": "string"},
                            "response_risk": {"type": "string"},
                            "desired_transition": {"type": "string"},
                            "uncertainty": {
                                "type": "string",
                                "enum": list(CONFIDENCE),
                            },
                        },
                        "required": [
                            "likely_reaction",
                            "response_risk",
                            "desired_transition",
                            "uncertainty",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["current", "future"],
                "additionalProperties": False,
            },
            "alignment": {
                "type": "object",
                "properties": {
                    "lambda_t": {"type": "number", "minimum": 0, "maximum": 1},
                    "orientation": {
                        "type": "string",
                        "enum": list(ORIENTATIONS),
                    },
                    "lambda_basis": {"type": "string"},
                    "self_constraint": {"type": "string"},
                    "user_adaptation": {"type": "string"},
                },
                "required": [
                    "lambda_t",
                    "orientation",
                    "lambda_basis",
                    "self_constraint",
                    "user_adaptation",
                ],
                "additionalProperties": False,
            },
            "behavior_policy": {
                "type": "object",
                "properties": {
                    "response_objective": {"type": "string"},
                    "perspective_taking": {"type": "string"},
                    "emotion_alignment": {"type": "string"},
                    "personalization": {"type": "string"},
                    "self_domain_expression": {"type": "string"},
                    "directness": {"type": "string", "enum": list(INTENSITY)},
                    "guidance": {
                        "type": "string",
                        "enum": ["none", "light", "direct"],
                    },
                    "question_policy": {
                        "type": "string",
                        "enum": ["none", "optional", "necessary"],
                    },
                    "tone": {"type": "string"},
                    "avoid": _string_array_schema(),
                },
                "required": [
                    "response_objective",
                    "perspective_taking",
                    "emotion_alignment",
                    "personalization",
                    "self_domain_expression",
                    "directness",
                    "guidance",
                    "question_policy",
                    "tone",
                    "avoid",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["user_state", "alignment", "behavior_policy"],
        "additionalProperties": False,
    },
}


def empty_user_domain() -> dict[str, Any]:
    return {
        "core": [],
        "regulation": [],
        "cognition": [],
        "identity": [],
        "behavior": [],
        "update_summary": {
            "added": [],
            "revised": [],
            "removed": [],
            "uncertainties": ["No partner evidence has been observed yet."],
        },
    }


def normalize_self_domain(value: Any) -> dict[str, Any]:
    root = _exact_object(value, SELF_DOMAIN_SCHEMA["schema"], "self_domain")
    identity = _exact_object(
        root["identity"],
        SELF_DOMAIN_SCHEMA["schema"]["properties"]["identity"],
        "identity",
    )
    persona = _exact_object(
        root["persona"],
        SELF_DOMAIN_SCHEMA["schema"]["properties"]["persona"],
        "persona",
    )
    prior = _exact_object(
        root["behavior_policy_prior"],
        SELF_DOMAIN_SCHEMA["schema"]["properties"]["behavior_policy_prior"],
        "behavior_policy_prior",
    )
    return {
        "identity": {key: _strings(identity[key], f"identity.{key}") for key in identity},
        "persona": {key: _strings(persona[key], f"persona.{key}") for key in persona},
        "behavior_policy_prior": {
            "interaction_principles": _strings(
                prior["interaction_principles"],
                "behavior_policy_prior.interaction_principles",
            ),
            "emotional_response_style": _text(
                prior["emotional_response_style"],
                "behavior_policy_prior.emotional_response_style",
            ),
            "guidance_style": _text(
                prior["guidance_style"], "behavior_policy_prior.guidance_style"
            ),
            "initiative": _enum(prior["initiative"], INTENSITY, "initiative"),
        },
        "hard_constraints": _strings(root["hard_constraints"], "hard_constraints"),
        "uncertainties": _strings(root["uncertainties"], "uncertainties"),
    }


def normalize_user_domain(value: Any) -> dict[str, Any]:
    root = _exact_object(value, USER_DOMAIN_SCHEMA["schema"], "user_domain")
    normalized: dict[str, Any] = {}
    for layer in ("core", "regulation", "cognition", "identity", "behavior"):
        facts = root[layer]
        if not isinstance(facts, list):
            raise ValueError(f"{layer} must be an array")
        seen: set[str] = set()
        normalized_facts = []
        for index, fact in enumerate(facts):
            item = _exact_object(fact, _fact_schema(), f"{layer}[{index}]")
            statement = _text(item["statement"], f"{layer}[{index}].statement")
            key = statement.casefold()
            if key in seen:
                raise ValueError(f"duplicate fact in {layer}: {statement}")
            seen.add(key)
            normalized_facts.append({
                "statement": statement,
                "confidence": _enum(
                    item["confidence"], CONFIDENCE, f"{layer}[{index}].confidence"
                ),
                "evidence_turn_ids": _strings(
                    item["evidence_turn_ids"], f"{layer}[{index}].evidence_turn_ids"
                ),
            })
        normalized[layer] = normalized_facts
    summary_schema = USER_DOMAIN_SCHEMA["schema"]["properties"]["update_summary"]
    summary = _exact_object(root["update_summary"], summary_schema, "update_summary")
    normalized["update_summary"] = {
        key: _strings(summary[key], f"update_summary.{key}") for key in summary
    }
    return normalized


def normalize_alignment(value: Any) -> dict[str, Any]:
    schema = ALIGNMENT_SCHEMA["schema"]
    root = _exact_object(value, schema, "alignment_result")
    state_schema = schema["properties"]["user_state"]
    state = _exact_object(root["user_state"], state_schema, "user_state")
    current = _exact_object(
        state["current"], state_schema["properties"]["current"], "user_state.current"
    )
    future = _exact_object(
        state["future"], state_schema["properties"]["future"], "user_state.future"
    )
    align_schema = schema["properties"]["alignment"]
    alignment = _exact_object(root["alignment"], align_schema, "alignment")
    policy_schema = schema["properties"]["behavior_policy"]
    policy = _exact_object(root["behavior_policy"], policy_schema, "behavior_policy")
    lambda_t = alignment["lambda_t"]
    if isinstance(lambda_t, bool) or not isinstance(lambda_t, (int, float)):
        raise ValueError("alignment.lambda_t must be numeric")
    lambda_t = round(float(lambda_t), 4)
    if not 0 <= lambda_t <= 1:
        raise ValueError("alignment.lambda_t must be in [0, 1]")
    orientation = _enum(alignment["orientation"], ORIENTATIONS, "orientation")
    expected = _orientation(lambda_t)
    if orientation != expected:
        raise ValueError(
            f"alignment.orientation {orientation!r} conflicts with lambda_t; expected {expected!r}"
        )
    return {
        "user_state": {
            "current": {
                "emotion": _text(current["emotion"], "current.emotion", allow_empty=True),
                "emotional_intensity": _enum(
                    current["emotional_intensity"], INTENSITY, "current.emotional_intensity"
                ),
                "intent": _text(current["intent"], "current.intent", allow_empty=True),
                "main_need": _text(current["main_need"], "current.main_need", allow_empty=True),
                "interaction_expectation": _text(
                    current["interaction_expectation"],
                    "current.interaction_expectation",
                    allow_empty=True,
                ),
                "evidence_turn_ids": _strings(
                    current["evidence_turn_ids"], "current.evidence_turn_ids"
                ),
                "uncertainty": _enum(
                    current["uncertainty"], CONFIDENCE, "current.uncertainty"
                ),
            },
            "future": {
                key: (
                    _enum(future[key], CONFIDENCE, "future.uncertainty")
                    if key == "uncertainty"
                    else _text(future[key], f"future.{key}", allow_empty=True)
                )
                for key in future
            },
        },
        "alignment": {
            "lambda_t": lambda_t,
            "orientation": orientation,
            **{
                key: _text(alignment[key], f"alignment.{key}")
                for key in ("lambda_basis", "self_constraint", "user_adaptation")
            },
        },
        "behavior_policy": {
            **{
                key: _text(policy[key], f"behavior_policy.{key}")
                for key in (
                    "response_objective",
                    "perspective_taking",
                    "emotion_alignment",
                    "personalization",
                    "self_domain_expression",
                    "tone",
                )
            },
            "directness": _enum(policy["directness"], INTENSITY, "directness"),
            "guidance": _enum(policy["guidance"], ("none", "light", "direct"), "guidance"),
            "question_policy": _enum(
                policy["question_policy"],
                ("none", "optional", "necessary"),
                "question_policy",
            ),
            "avoid": _strings(policy["avoid"], "behavior_policy.avoid"),
        },
    }


def _exact_object(value: Any, schema: dict[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    required = set(schema.get("required", []))
    keys = set(value)
    if keys != required:
        raise ValueError(
            f"{path} fields mismatch: missing={sorted(required - keys)} extra={sorted(keys - required)}"
        )
    return value


def _strings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    result = []
    for index, item in enumerate(value):
        text = _text(item, f"{path}[{index}]")
        if text not in result:
            result.append(text)
    return result


def _text(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError(f"{path} must not be empty")
    return text


def _enum(value: Any, allowed: tuple[str, ...], path: str) -> str:
    text = _text(value, path)
    if text not in allowed:
        raise ValueError(f"{path} must be one of {allowed}, got {text!r}")
    return text


def _orientation(lambda_t: float) -> str:
    if lambda_t < 0.25:
        return "self-dominant"
    if lambda_t < 0.5:
        return "self-leaning"
    if lambda_t < 0.75:
        return "user-leaning"
    return "strongly-user-oriented"


Normalizer = Callable[[Any], dict[str, Any]]

