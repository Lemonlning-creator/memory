from __future__ import annotations

from typing import Any, Dict


def state_axis(profile: Dict[str, Any]) -> Dict[str, Any]:
    return profile["state_axis"]


def context_axis(profile: Dict[str, Any]) -> Dict[str, Any]:
    return profile["context_axis"]


def flatten_static_profile(static_profile: Dict[str, Any]) -> Dict[str, Any]:
    """展平带 memory_ids 的 static_profile 为纯 value dict，供 LLM prompt 使用。"""
    result: Dict[str, Any] = {}
    for section, fields in static_profile.items():
        if section == "basic_info":
            result[section] = fields
            continue
        result[section] = {}
        for key, val in fields.items():
            result[section][key] = val["value"] if isinstance(val, dict) and "value" in val else val
    return result


def convert_to_flat_profile(static_profile: Dict[str, Any]) -> Dict[str, Any]:
    """将五层层次化画像转换为扁平画像（无层级结构）。

    用于实验1（Flat Profile baseline）和实验5（消融 w/o hierarchical structure）。
    将所有层级的属性合并到一个无层级的 dict 中，key 为 "section.attribute" 格式。

    Input: 5-layer profile like:
        {
            "core": {"fears": {"value": "...", "confidence": 0.8, ...}},
            "regulation": {"humor": {"value": "...", ...}},
            ...
        }

    Output: flat profile like:
        {
            "core_fears": {"value": "...", "confidence": 0.8, ...},
            "regulation_humor": {"value": "...", ...},
            ...
        }
    """
    flat: Dict[str, Any] = {}
    for section, fields in static_profile.items():
        if not isinstance(fields, dict):
            flat[section] = fields
            continue
        for key, val in fields.items():
            flat_key = f"{section}_{key}"
            flat[flat_key] = val
    return flat


def convert_from_flat_profile(flat_profile: Dict[str, Any]) -> Dict[str, Any]:
    """将扁平画像还原为五层层次结构（用于存储和兼容性）。

    Input: flat profile like:
        {"core_fears": {"value": "..."}, "regulation_humor": {"value": "..."}}

    Output: 5-layer profile like:
        {"core": {"fears": {"value": "..."}}, "regulation": {"humor": {"value": "..."}}}
    """
    from .epistemic_decay import PROFILE_LAYERS

    layered: Dict[str, Any] = {layer: {} for layer in PROFILE_LAYERS}
    for flat_key, val in flat_profile.items():
        matched = False
        for layer in PROFILE_LAYERS:
            prefix = f"{layer}_"
            if flat_key.startswith(prefix):
                attr_name = flat_key[len(prefix):]
                layered[layer][attr_name] = val
                matched = True
                break
        if not matched:
            # Put unmatched keys into a "meta" section
            layered.setdefault("meta", {})[flat_key] = val
    return layered


def count_profile_attributes(static_profile: Dict[str, Any]) -> int:
    """统计画像中属性数量（用于 completeness 和 entropy 计算）。"""
    count = 0
    for section, fields in static_profile.items():
        if isinstance(fields, dict):
            count += len(fields)
    return count


def get_attribute_confidences(static_profile: Dict[str, Any]) -> Dict[str, float]:
    """提取画像中所有属性的 confidence 值。

    Returns:
        Dict mapping "section.attribute" -> confidence value.
        Attributes without confidence default to 0.5.
    """
    confidences: Dict[str, float] = {}
    for section, fields in static_profile.items():
        if not isinstance(fields, dict):
            continue
        for key, val in fields.items():
            path = f"{section}.{key}"
            if isinstance(val, dict):
                c = val.get("confidence", 0.5)
                if not isinstance(c, (int, float)):
                    c = 0.5
                confidences[path] = float(c)
            else:
                confidences[path] = 0.5
    return confidences


def migrate_profile(old: Dict[str, Any]) -> Dict[str, Any]:
    """将旧格式 user_profile 迁移到三轴结构。"""
    def wrap_leaves(section: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v if isinstance(v, dict) and "value" in v else {"value": v, "memory_ids": []})
            for k, v in section.items()
        }

    old_static = old.get("static_profile", {})
    new_static: Dict[str, Any] = {}
    for section, fields in old_static.items():
        if isinstance(fields, dict):
            new_static[section] = wrap_leaves(fields)
        else:
            new_static[section] = fields

    return {
        "state_axis": {
            "static_profile": new_static,
            "current_state": old.get("current_state", {}),
            "projected_state": old.get("projected_state", {}),
        },
        "context_axis": {
            "current_context": "其他",
            "context_detail": "",
            "inferred_at_turn": 0,
        },
    }


def create_empty_profile() -> Dict[str, Any]:
    return {
        "state_axis": {
            "static_profile": {},
            "current_state": {},
            "projected_state": {},
        },
        "context_axis": {
            "current_context": "",
            "context_detail": "",
            "inferred_at_turn": 0,
        },
    }
