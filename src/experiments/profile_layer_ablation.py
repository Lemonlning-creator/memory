"""Five-layer user-profile ablation experiment required by todolist.md."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm_client import LLMClient
from ..profile_utils import flatten_static_profile
from ..prompts.prompt_loader import (
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    DIRECT_RESPONSE_USER_PROMPT_TEMPLATE,
    USER_PROFILE_ACTIVATION_SYSTEM_PROMPT,
    USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE,
)
from ..utils import load_json, parse_json, save_json

LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
CASES = (
    ("core", "core_growth", "我最近总觉得自己在原地打转。明明一直在学习，却担心这些努力最后没有沉淀成真正有价值的东西。"),
    ("regulation", "regulation_pressure", "项目突然卡住了，我脑子里一直在反复推演各种失败可能，越想越停不下来。"),
    ("cognition", "cognition_decision", "我有两个方案：一个快但后续维护成本高，一个慢但结构更稳。你帮我按长期影响把它们拆开比较一下。"),
    ("identity", "identity_work", "实验室下周要做阶段汇报。我既要解释算法结果，也得让非技术同事听懂，你觉得该怎么准备？"),
    ("behavior", "behavior_preference", "我想找一套能长期坚持的科研效率工具和工作流，不想要只图新鲜感的推荐。"),
)


def _profile_text(profile: dict[str, Any]) -> str:
    flat = flatten_static_profile(profile)
    lines = [f"- {field}: {value}" for attrs in flat.values() if isinstance(attrs, dict)
             for field, value in attrs.items() if value]
    return "\n".join(lines)


def _response(llm: LLMClient, message: str, profile: dict[str, Any], persona: dict[str, Any]) -> str:
    prompt = DIRECT_RESPONSE_USER_PROMPT_TEMPLATE.format(
        user_input=message,
        static_profile=_profile_text(profile),
        current_state="{}",
        current_context="{}",
        persona_config=json.dumps(persona, ensure_ascii=False),
        relevant_memory="{}",
    )
    return llm.chat(DIRECT_RESPONSE_SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=450)


def _activation(llm: LLMClient, message: str, response: str, profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    candidates = {
        layer: {field: (payload.get("value") if isinstance(payload, dict) else payload)
                for field, payload in fields.items()
                if (payload.get("value") if isinstance(payload, dict) else payload)}
        for layer, fields in profile.items() if layer in LAYERS and isinstance(fields, dict)
    }
    try:
        result = parse_json(llm.chat(
            USER_PROFILE_ACTIVATION_SYSTEM_PROMPT,
            USER_PROFILE_ACTIVATION_USER_PROMPT_TEMPLATE.format(
                user_message=message,
                assistant_response=response,
                current_context="{}",
                user_profile=json.dumps(candidates, ensure_ascii=False),
            ),
            temperature=0.0,
            max_tokens=600,
        ))
    except Exception:
        result = {}
    raw = result.get("activated_profile", {}) if isinstance(result, dict) else {}
    return {layer: raw.get(layer, []) if isinstance(raw.get(layer, []), list) else [] for layer in LAYERS}


def _judge(llm: LLMClient, message: str, response: str) -> dict[str, float]:
    prompt = f"""Evaluate this Chinese companion-agent reply against the user message. Return JSON only.
User message: {message}
Agent reply: {response}
Score each dimension from 1 (poor) to 5 (excellent): relevance, personalization, helpfulness, naturalness.
JSON: {{\"relevance\": number, \"personalization\": number, \"helpfulness\": number, \"naturalness\": number}}"""
    try:
        result = parse_json(llm.chat("You are a strict, neutral dialogue evaluator. Output only JSON.", prompt, temperature=0.0, max_tokens=150))
    except Exception:
        result = {}
    return {name: float(result.get(name, 0)) for name in ("relevance", "personalization", "helpfulness", "naturalness")}


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def run(profile_path: str, persona_path: str, output_dir: str, conditions: tuple[str, ...] | None = None) -> dict[str, Any]:
    llm = LLMClient()
    raw = load_json(profile_path)
    profile = raw.get("state_axis", {}).get("static_profile", raw)
    persona = load_json(persona_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "profile_layer_ablation_results.json"
    existing = load_json(str(results_path)).get("records", []) if results_path.exists() else []
    selected = conditions or ("full_profile", *[f"without_{layer}" for layer in LAYERS])
    records: list[dict[str, Any]] = existing
    completed = {(r.get("case_id"), r.get("condition")) for r in records}
    for target_layer, case_id, message in CASES:
        for condition in selected:
            if (case_id, condition) in completed:
                continue
            working = deepcopy(profile)
            removed = None if condition == "full_profile" else condition.removeprefix("without_")
            if removed:
                working.pop(removed, None)
            response = _response(llm, message, working, persona)
            activation = _activation(llm, message, response, working)
            scores = _judge(llm, message, response)
            records.append({
                "case_id": case_id,
                "target_layer": target_layer,
                "message": message,
                "condition": condition,
                "removed_layer": removed,
                "response": response,
                "activation": activation,
                "scores": scores,
            })
            # Checkpoint each model triplet, so a network interruption does not
            # discard already completed cases.
            save_json(str(results_path), {"records": records})
            print(f"{case_id} | {condition} complete")

    summary: dict[str, Any] = {"num_cases": len(CASES), "conditions": {}}
    baseline_by_case = {r["case_id"]: r for r in records if r["condition"] == "full_profile"}
    for condition in ("full_profile", *[f"without_{layer}" for layer in LAYERS]):
        group = [r for r in records if r["condition"] == condition]
        if not group:
            continue
        summary["conditions"][condition] = {
            "mean_scores": {metric: _mean([r["scores"][metric] for r in group])
                            for metric in ("relevance", "personalization", "helpfulness", "naturalness")},
            "target_layer_activation_rate": round(sum(bool(r["activation"][r["target_layer"]]) for r in group) / len(group), 3),
        }
    layer_effects = {}
    for layer in LAYERS:
        case_id = next(case_id for target, case_id, _ in CASES if target == layer)
        base = baseline_by_case.get(case_id)
        ablated = next((r for r in records if r["case_id"] == case_id and r["condition"] == f"without_{layer}"), None)
        if base and ablated:
            layer_effects[layer] = {
                "baseline_target_activated": bool(base["activation"][layer]),
                "ablated_target_activated": bool(ablated["activation"][layer]),
                "personalization_delta": round(ablated["scores"]["personalization"] - base["scores"]["personalization"], 3),
                "helpfulness_delta": round(ablated["scores"]["helpfulness"] - base["scores"]["helpfulness"], 3),
            }
    summary["targeted_layer_effects"] = layer_effects
    summary["run_at"] = datetime.now().isoformat()
    save_json(str(results_path), {"records": records})
    save_json(str(root / "profile_layer_ablation_summary.json"), summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="dataset/test_user.json")
    parser.add_argument("--persona", default="dataset/test_agent.json")
    parser.add_argument("--output-dir", default="data/profile_layer_ablation")
    parser.add_argument("--conditions", nargs="*", choices=("full_profile", *[f"without_{layer}" for layer in LAYERS]))
    args = parser.parse_args()
    run(args.profile, args.persona, args.output_dir, tuple(args.conditions) if args.conditions else None)
