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


def build_cases(profile: dict[str, Any]) -> tuple[tuple[str, str, str, str], ...]:
    """Build one deterministic layer-targeted input from dataset/test_user.json.

    Each test input is grounded in the highest-confidence attribute in its own
    profile layer, replacing the previous hand-written CASES fixture.
    """
    templates = {
        "core": "我最近常有一种担心：{value}。你会怎么建议我？",
        "regulation": "碰到难题时，我往往会这样：{value}。现在该怎么处理？",
        "cognition": "我处理信息或做决策时的习惯是：{value}。这次我应该怎么推进？",
        "identity": "我的实际情况是：{value}。你给我的建议可以怎样更贴合？",
        "behavior": "我平时的偏好或习惯是：{value}。你会怎么给建议？",
    }
    cases = []
    for layer in LAYERS:
        fields = profile.get(layer, {})
        if not isinstance(fields, dict) or not fields:
            raise ValueError(f"Profile layer '{layer}' is missing or empty.")
        field, payload = max(
            fields.items(),
            key=lambda item: float(item[1].get("confidence", 0)) if isinstance(item[1], dict) else 0.0,
        )
        value = payload.get("value", "") if isinstance(payload, dict) else str(payload)
        value = value.strip().rstrip("。！？!?")
        if not value:
            raise ValueError(f"Profile field '{layer}.{field}' has no value.")
        case_id = f"{layer}_{field}".replace(" ", "_")
        cases.append((layer, case_id, templates[layer].format(value=value), field))
    return tuple(cases)


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
    cases = build_cases(profile)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "profile_layer_ablation_results.json"
    existing = load_json(str(results_path)).get("records", []) if results_path.exists() else []
    current_case_ids = {case_id for _, case_id, _, _ in cases}
    # Do not mix results produced by an older test-case definition with the
    # profile-derived cases used by this version of the experiment.
    existing = [r for r in existing if r.get("case_id") in current_case_ids]
    selected = conditions or ("full_profile", *[f"without_{layer}" for layer in LAYERS])
    records: list[dict[str, Any]] = existing
    completed = {(r.get("case_id"), r.get("condition")) for r in records}
    for target_layer, case_id, message, source_field in cases:
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
                "source_profile_field": source_field,
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

    summary: dict[str, Any] = {"num_cases": len(cases), "conditions": {}}
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
        case_id = next(case_id for target, case_id, _, _ in cases if target == layer)
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
