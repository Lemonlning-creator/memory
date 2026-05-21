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
