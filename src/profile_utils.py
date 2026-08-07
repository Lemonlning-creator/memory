from __future__ import annotations

from typing import Any, Dict, Mapping

from .profile_schema import PROFILE_FIELDS, PROFILE_LAYERS, create_empty_static_profile, normalize_bare_profile


def state_axis(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility view for runtime-only state consumers.

    Persisted profiles are bare five-layer documents. Legacy callers can still
    read static_profile through this temporary view without writing wrappers.
    """
    if "state_axis" in profile:
        return profile["state_axis"]
    return {"static_profile": profile, "current_state": {}, "projected_state": {}}


def context_axis(profile: Dict[str, Any]) -> Dict[str, Any]:
    if "context_axis" in profile:
        return profile["context_axis"]
    return {"current_context": "", "context_detail": "", "inferred_at_turn": 0}


def flatten_static_profile(static_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return the complete fixed five-layer profile used by runtime prompts.

    Empty summaries and fields are deliberately preserved so the response model
    can distinguish known profile content from profile gaps.
    """
    return normalize_bare_profile(static_profile)


def serialize_static_profile_for_prompt(static_profile: Dict[str, Any]) -> str:
    """Serialize every fixed profile field as grouped text, including empty values."""
    normalized = flatten_static_profile(static_profile)
    lines = []
    for layer in PROFILE_LAYERS:
        lines.append(f"{layer}:")
        section = normalized[layer]
        for field in ("summary", *PROFILE_FIELDS[layer]):
            value = section[field]
            text = "；".join(value) if isinstance(value, list) else value
            lines.append(f"  - {field}: {text}")
    return "\n".join(lines)


def convert_to_flat_profile(static_profile: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    normalized = normalize_bare_profile(static_profile)
    for layer in PROFILE_LAYERS:
        for field in PROFILE_FIELDS[layer]:
            flat[f"{layer}_{field}"] = normalized[layer][field]
    return flat


def convert_from_flat_profile(flat_profile: Dict[str, Any]) -> Dict[str, Any]:
    layered = create_empty_static_profile()
    for layer in PROFILE_LAYERS:
        prefix = f"{layer}_"
        for key, value in flat_profile.items():
            if key.startswith(prefix):
                attribute = key[len(prefix):]
                if attribute in PROFILE_FIELDS[layer]:
                    layered[layer][attribute] = normalize_bare_profile({layer: {attribute: value}})[layer][attribute]
    return layered


def count_profile_attributes(static_profile: Dict[str, Any]) -> int:
    normalized = normalize_bare_profile(static_profile)
    return sum(
        1
        for layer in PROFILE_LAYERS
        for field in PROFILE_FIELDS[layer]
        if normalized[layer][field]
    )


def get_attribute_confidences(static_profile: Dict[str, Any]) -> Dict[str, float]:
    """Legacy experimental metric: persisted bare values have neutral confidence."""
    normalized = normalize_bare_profile(static_profile)
    return {
        f"{layer}.{field}": 0.5
        for layer in PROFILE_LAYERS
        for field in PROFILE_FIELDS[layer]
        if normalized[layer][field]
    }


def migrate_profile(old: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy wrapper/leaf formats to the persisted bare five-layer contract."""
    return normalize_bare_profile(old)


def runtime_profile_from_bare(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the in-memory wrapper used by existing runtime dialogue logic.

    The wrapper is never persisted or returned by the Profile API.
    """
    return {
        "state_axis": {
            "static_profile": normalize_bare_profile(profile),
            "current_state": {},
            "projected_state": {},
        },
        "context_axis": {
            "current_context": "",
            "context_detail": "",
            "inferred_at_turn": 0,
        },
    }


def create_empty_profile() -> Dict[str, Any]:
    """Legacy/runtime constructor retained for experiment and dialogue code."""
    return runtime_profile_from_bare(create_empty_static_profile())
