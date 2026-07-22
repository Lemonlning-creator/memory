"""Single-layer user-profile ablation experiment.

Each natural user input is designed for exactly one profile layer.  The
baseline sees the full profile; its only comparison removes that target layer.
The ablated response is then checked twice for profile-activation leakage.
"""
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

# These inputs deliberately describe situations rather than copy a field value
# from test_user.json.  Each case is intended to need only one target layer.
CASE_SPECS = (
    {
        "layer": "core",
        "field": "desires",
        "case_id": "core_long_term_direction",
        "message": "最近忙了很多学习和项目，但我总觉得东西越积越多、方向却不够清楚。你觉得我下一步该怎么取舍？",
        "design_rationale": "测试长期成长、知识整合与长期价值取向，而非职业或内容偏好。",
    },
    {
        "layer": "regulation",
        "field": "control",
        "case_id": "regulation_overplanning",
        "message": "我想把方案的每个细节都想清楚再开始，结果越规划越难推进。现在卡住了，该怎么破？",
        "design_rationale": "测试规划/控制作为应对方式，不包含职业、兴趣或表达风格线索。",
    },
    {
        "layer": "cognition",
        "field": "expression style",
        "case_id": "cognition_explain_complexity",
        "message": "我要把一个复杂方案讲给别人听，但担心信息太散、对方抓不住重点。你会怎么帮我组织？",
        "design_rationale": "测试结构化表达和信息组织偏好，不指定职业领域。",
    },
    {
        "layer": "identity",
        "field": "occupation",
        "case_id": "identity_technical_report",
        "message": "下周我要做一次技术结果汇报，听众既有懂技术的人也有非技术同事。准备时最应该注意什么？",
        "design_rationale": "测试知识密集型工作与技术沟通的身份背景，不要求调用长期目标或兴趣。",
    },
    {
        "layer": "behavior",
        "field": "content preferences",
        "case_id": "behavior_learning_topic_choice",
        "message": "这周我有一点空档，想选一个既有技术深度、又能持续激发思考的主题系统学一下。你会推荐从什么方向开始，为什么？",
        "design_rationale": "强制模型根据内容偏好选择具体学习主题与路径，避免只做泛化追问。",
    },
)

METRICS = ("target_layer_fit", "target_layer_utilization", "helpfulness", "naturalness")


def build_cases(profile: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate profile coverage and attach the target field as ground truth."""
    cases: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        layer = spec["layer"]
        field = spec["field"]
        payload = profile.get(layer, {}).get(field)
        if payload is None:
            raise ValueError(f"Missing required target field: {layer}.{field}")
        value = payload.get("value") if isinstance(payload, dict) else payload
        if not value:
            raise ValueError(f"Target field has no usable value: {layer}.{field}")
        cases.append({**spec, "target_profile": {layer: {field: payload}}})
    return tuple(cases)


def _profile_text(profile: dict[str, Any]) -> str:
    flat = flatten_static_profile(profile)
    lines = [
        f"- {field}: {value}"
        for attrs in flat.values() if isinstance(attrs, dict)
        for field, value in attrs.items() if value
    ]
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


def _activation(
    llm: LLMClient,
    message: str,
    response: str,
    candidate_profile: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Post-hoc activation judged only against the supplied candidate profile."""
    candidates = {
        layer: {
            field: payload.get("value") if isinstance(payload, dict) else payload
            for field, payload in fields.items()
            if (payload.get("value") if isinstance(payload, dict) else payload)
        }
        for layer, fields in candidate_profile.items()
        if layer in LAYERS and isinstance(fields, dict)
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
    return {
        layer: raw.get(layer, []) if isinstance(raw.get(layer, []), list) else []
        for layer in LAYERS
    }


def _target_is_activated(activation: dict[str, list[dict[str, Any]]], layer: str, field: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("field") == field
        for item in activation.get(layer, [])
    )


def _judge(
    llm: LLMClient,
    message: str,
    response: str,
    target_layer: str,
    target_field: str,
    target_profile: dict[str, Any],
) -> dict[str, float]:
    """Evaluate only the target layer's effect from that user's perspective."""
    prompt = f"""你是用户画像消融实验的严格评审员。请代入拥有下列“目标画像信息”的用户，评估回复是否真正利用了这一个画像层；不要推断或使用其他层的用户信息。

目标层：{target_layer}
目标字段：{target_field}
目标画像信息：
{json.dumps(target_profile, ensure_ascii=False, indent=2)}

用户输入：
{message}

智能体回复：
{response}

逐项给 1–5 的整数分：
1. target_layer_fit：这位用户会认为回复是否贴合目标画像信息和当前情境。
2. target_layer_utilization：回复是否清楚、恰当地把目标字段转化为具体的理解、话题选择或建议；泛化回答不得高分。
3. helpfulness：针对当前输入，回复是否有实际帮助或下一步价值。
4. naturalness：回复是否自然，不过度揣测，适合继续对话。

只输出 JSON：
{{"target_layer_fit": number, "target_layer_utilization": number, "helpfulness": number, "naturalness": number}}"""
    try:
        result = parse_json(llm.chat(
            "You are a strict evaluator. Output only JSON.",
            prompt,
            temperature=0.0,
            max_tokens=180,
        ))
    except Exception:
        result = {}
    return {metric: float(result.get(metric, 0)) for metric in METRICS}


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    """Write a compact report that matches the v2 experiment schema."""
    valid = summary["valid_case_means"]
    rows = []
    for layer in LAYERS:
        item = summary["per_layer"][layer]
        delta = item["ablation_delta"]
        status = "可用" if item["case_usable"] else "不可用（目标层信息泄露）"
        rows.append(
            f"| {layer} | {item['target_field']} | {status} | "
            f"{item['baseline_target_activated']} | "
            f"{item['target_leaked_with_full_reference']} | "
            f"{delta['target_layer_fit']:+.1f} | "
            f"{delta['target_layer_utilization']:+.1f} | "
            f"{delta['helpfulness']:+.1f} |"
        )
    leaked = ", ".join(summary["leaked_cases"]) or "无"
    content = f"""# 五层用户画像单层消融实验报告

**实验版本：** `{summary['experiment_version']}`
**运行时间：** {summary['run_at']}
**用例数：** {summary['num_cases']}；**可用用例：** {summary['usable_cases']}

## 实验设计

每个自然语言输入只绑定一个目标画像层。基线向智能体提供完整画像；消融条件仅删除该用例的目标层，其余层保留。每条消融回复再进行两次激活判定：一次使用消融后的候选画像，另一次使用完整画像作为参考。若后一次重新判出被删目标字段，则说明其他层的信息可推断出目标层，该用例标记为泄露、不可用于因果结论。

评分器只接收当前目标层及其字段，而不接收完整画像，评估目标层贴合度、目标层实际利用度、帮助性与自然度。

## 分层结果

| 层级 | 目标字段 | 用例可用性 | 基线目标激活 | 消融回复泄露 | 贴合度变化 | 利用度变化 | 帮助性变化 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

变化值为“消融 − 基线”；负值表示删除目标层后该指标下降。检测到泄露的用例：{leaked}。

## 仅对可用用例的均值

| 条件 | 目标层贴合度 | 目标层利用度 | 帮助性 | 自然度 |
| --- | ---: | ---: | ---: | ---: |
| 完整画像 | {valid['baseline']['target_layer_fit']:.2f} | {valid['baseline']['target_layer_utilization']:.2f} | {valid['baseline']['helpfulness']:.2f} | {valid['baseline']['naturalness']:.2f} |
| 删除目标层 | {valid['ablated']['target_layer_fit']:.2f} | {valid['ablated']['target_layer_utilization']:.2f} | {valid['ablated']['helpfulness']:.2f} | {valid['ablated']['naturalness']:.2f} |

## 解读规则

- 仅当用例没有泄露、且基线目标字段被激活而消融回复未激活时，才将评分变化解释为该画像层的证据。
- 若发生泄露，应重新设计输入，使其不被其他画像层的语义信息覆盖。
- 每个条件至少重复 3 次并加入人工盲评后，方可作稳定的层级贡献结论。
"""
    (output / "report.md").write_text(content, encoding="utf-8")


def run(profile_path: str, persona_path: str, output_dir: str) -> dict[str, Any]:
    llm = LLMClient()
    raw = load_json(profile_path)
    full_profile = raw.get("state_axis", {}).get("static_profile", raw)
    persona = load_json(persona_path)
    cases = build_cases(full_profile)
    results: list[dict[str, Any]] = []

    for case in cases:
        layer = case["layer"]
        field = case["field"]
        message = case["message"]
        target_profile = case["target_profile"]

        # Only this layer is ablated for this case; all other layers remain
        # available exactly as they are in the full-profile baseline.
        ablated_profile = deepcopy(full_profile)
        ablated_profile.pop(layer, None)

        baseline_response = _response(llm, message, full_profile, persona)
        baseline_activation = _activation(llm, message, baseline_response, full_profile)
        baseline_scores = _judge(llm, message, baseline_response, layer, field, target_profile)

        ablated_response = _response(llm, message, ablated_profile, persona)
        # First judge against what the ablated agent actually received.
        ablated_activation = _activation(llm, message, ablated_response, ablated_profile)
        # Then judge the same reply against the full profile.  If the removed
        # target field appears here, the remaining layers leaked target-layer
        # information and the test case is not isolated.
        full_reference_activation = _activation(llm, message, ablated_response, full_profile)
        ablated_scores = _judge(llm, message, ablated_response, layer, field, target_profile)

        target_leaked = _target_is_activated(full_reference_activation, layer, field)
        results.append({
            "case_id": case["case_id"],
            "target_layer": layer,
            "target_field": field,
            "target_profile": target_profile,
            "message": message,
            "design_rationale": case["design_rationale"],
            "baseline": {
                "agent_profile_scope": "full_profile",
                "response": baseline_response,
                "activation": baseline_activation,
                "target_activated": _target_is_activated(baseline_activation, layer, field),
                "scores": baseline_scores,
            },
            "ablated": {
                "agent_profile_scope": f"full_profile_without_{layer}",
                "response": ablated_response,
                "activation_with_ablated_candidates": ablated_activation,
                "activation_with_full_reference": full_reference_activation,
                "target_activated_with_ablated_candidates": _target_is_activated(ablated_activation, layer, field),
                "target_leaked_with_full_reference": target_leaked,
                "scores": ablated_scores,
            },
            "case_usable": not target_leaked,
        })
        print(f"{case['case_id']} complete | leaked={target_leaked}")

    layer_summary: dict[str, Any] = {}
    for result in results:
        baseline_scores = result["baseline"]["scores"]
        ablated_scores = result["ablated"]["scores"]
        layer_summary[result["target_layer"]] = {
            "case_id": result["case_id"],
            "target_field": result["target_field"],
            "case_usable": result["case_usable"],
            "baseline_target_activated": result["baseline"]["target_activated"],
            "ablated_target_activated": result["ablated"]["target_activated_with_ablated_candidates"],
            "target_leaked_with_full_reference": result["ablated"]["target_leaked_with_full_reference"],
            "baseline_scores": baseline_scores,
            "ablated_scores": ablated_scores,
            "ablation_delta": {
                metric: round(ablated_scores[metric] - baseline_scores[metric], 3)
                for metric in METRICS
            },
        }

    valid_results = [result for result in results if result["case_usable"]]
    summary = {
        "experiment_version": "single_layer_ablation_v2",
        "num_cases": len(results),
        "usable_cases": len(valid_results),
        "leaked_cases": [result["case_id"] for result in results if not result["case_usable"]],
        "per_layer": layer_summary,
        "valid_case_means": {
            "baseline": {
                metric: _mean([r["baseline"]["scores"][metric] for r in valid_results])
                for metric in METRICS
            },
            "ablated": {
                metric: _mean([r["ablated"]["scores"][metric] for r in valid_results])
                for metric in METRICS
            },
        },
        "run_at": datetime.now().isoformat(),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(str(output / "profile_layer_ablation_results.json"), {"records": results})
    save_json(str(output / "profile_layer_ablation_summary.json"), summary)
    _write_report(output, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-layer user-profile ablation experiment")
    parser.add_argument("--profile", default="dataset/test_user.json")
    parser.add_argument("--persona", default="dataset/test_agent.json")
    parser.add_argument("--output-dir", default="data/profile_layer_ablation")
    args = parser.parse_args()
    run(args.profile, args.persona, args.output_dir)
