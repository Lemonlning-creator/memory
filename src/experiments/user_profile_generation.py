from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..agent import StateDrivenCompanionAgent
from ..memory_os_local import MemoryOSLocal
from ..utils import save_json, load_json
from ..llm_client import LLMClient
from .agent_persona_generation import detect_agent_speaker

# =========================
# 画像格式转换：agent 内部格式 → 简洁展示格式
# =========================
# agent 内部格式：
#   {state_axis: {static_profile: {core: {fears: {value: "...", confidence: 0.5, ...}}}}}
#
# 目标格式（参考 dataset/lsy_user.json）：
#   {core: {summary: "...", values: ["...", "..."], motivations: ["...", ...]}}

PROFILE_REFORMAT_SYSTEM_PROMPT = """你是一个用户画像格式化专家。你的任务是将一个结构化的用户画像（JSON）转换为简洁、易读的展示格式。

目标格式要求：
1. 保留5层结构：core、regulation、cognition、identity、behavior
2. 每层包含一个 summary（一句话概括该层画像）
3. 每层的属性转换为字符串数组，key 用简短的描述性名称
4. 过滤掉"None explicitly mentioned"或空的属性
5. 合并相似的属性，避免重复
6. 每个数组包含2-5个条目，每条是简短的一句话

输出纯 JSON，不要任何解释。"""

PROFILE_REFORMAT_USER_PROMPT = """请将以下用户画像转换为简洁展示格式。

参考示例（目标格式）：
{example}

需要转换的画像：
{profile}

输出 JSON："""

# 目标格式示例（从 lsy_user.json 提取）
FORMAT_EXAMPLE = {
    "core": {
        "summary": "一句话概括该层画像",
        "values": ["条目1", "条目2"],
        "motivations": ["条目1", "条目2"],
        "concerns": ["条目1"]
    },
    "regulation": {
        "summary": "一句话概括该层画像",
        "stress_response": ["条目1", "条目2"],
        "conflict_style": ["条目1"],
        "emotion_regulation": ["条目1", "条目2"]
    },
    "cognition": {
        "summary": "一句话概括该层画像",
        "thinking_style": ["条目1", "条目2"],
        "decision_style": ["条目1"]
    },
    "identity": {
        "summary": "一句话概括该层画像",
        "current_stage": ["条目1", "条目2"],
        "professional_identity": ["条目1"]
    },
    "behavior": {
        "summary": "一句话概括该层画像",
        "learning": ["条目1"],
        "interests": ["条目1", "条目2"]
    }
}


def _extract_raw_profile(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """从 agent 内部格式中提取 static_profile，去掉元数据。"""
    static_profile = user_profile.get("state_axis", {}).get("static_profile", {})
    if not static_profile:
        static_profile = user_profile

    raw: Dict[str, Any] = {}
    for layer in ["core", "regulation", "cognition", "identity", "behavior"]:
        layer_data = static_profile.get(layer, {})
        if not isinstance(layer_data, dict):
            continue
        raw[layer] = {}
        for attr_key, attr_val in layer_data.items():
            if isinstance(attr_val, dict):
                value = attr_val.get("value", "")
                confidence = attr_val.get("confidence", 0.5)
                # 过滤掉空值和 "None explicitly mentioned"
                if not value or "none explicitly" in value.lower() or "not explicitly" in value.lower():
                    continue
                raw[layer][attr_key] = {
                    "value": value,
                    "confidence": confidence,
                    "evidence": attr_val.get("evidence", ""),
                }
            elif isinstance(attr_val, str) and attr_val.strip():
                raw[layer][attr_key] = {"value": attr_val}

    return raw


def reformat_profile(llm: LLMClient, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """将 agent 内部画像格式转换为简洁展示格式。"""
    raw_profile = _extract_raw_profile(user_profile)

    if not any(raw_profile.values()):
        print("[Profile Reformat] 画像为空，跳过转换")
        return {}

    # 用 JSON 格式传给 LLM
    profile_json = json.dumps(raw_profile, ensure_ascii=False, indent=2)
    example_json = json.dumps(FORMAT_EXAMPLE, ensure_ascii=False, indent=2)

    user_prompt = PROFILE_REFORMAT_USER_PROMPT.format(
        example=example_json,
        profile=profile_json,
    )

    try:
        raw_response = llm.chat(
            PROFILE_REFORMAT_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.3,
            max_tokens=2000,
        )
        # 解析 JSON
        text = raw_response.strip().strip("```json").strip("```").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        result = json.loads(text)

        # 验证结构
        for layer in ["core", "regulation", "cognition", "identity", "behavior"]:
            if layer not in result:
                result[layer] = {"summary": ""}

        print("[Profile Reformat] 转换完成")
        return result
    except Exception as e:
        print(f"[Profile Reformat] 转换失败: {e}")
        # 降级：直接返回提取的原始格式
        fallback: Dict[str, Any] = {}
        for layer, attrs in raw_profile.items():
            fallback[layer] = {"summary": "", "attributes": {}}
            for k, v in attrs.items():
                fallback[layer]["attributes"][k] = v.get("value", "")
        return fallback


# =========================
# 原有管线逻辑
# =========================

def detect_user_speaker(chat: Dict[str, Any]) -> str:
    names = chat.get("name", {})
    if isinstance(names, dict) and names.get("speaker_1"):
        return str(names["speaker_1"])
    return "default_user"

def session_keys(chat: Dict[str, Any]) -> List[str]:
    keys = [
        key for key, value in chat.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))

def flatten_dialogue(chat: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for key in session_keys(chat):
        messages = chat[key]
        for message_index, message in enumerate(messages, start=1):
            content = str(message.get("clean_text") or "").strip()
            if not content:
                continue
            turns.append({
                "turn_index": len(turns) + 1,
                "session_id": key,
                "message_index": message_index,
                "session_message_count": len(messages),
                "dia_id": message.get("dia_id", ""),
                "speaker": str(message.get("speaker") or "").strip(),
                "content": content,
            })
    return turns

def checkpoint_path_for(memory_path: Path) -> Path:
    return memory_path / "pipeline_checkpoint.json"

def load_checkpoint(memory_path: Path) -> Dict[str, Any] | None:
    path = checkpoint_path_for(memory_path)
    if not path.exists():
        return None
    return load_json(path)

def save_checkpoint(memory_path: Path, payload: Dict[str, Any]) -> None:
    save_json(str(checkpoint_path_for(memory_path)), payload)

def run_profile_pipeline(
    realtalk_path: str | Path,
    profile_path: str | None = None,
    resume: bool = False,
    start_turn: int | None = None,
    replay_window: int = 20,
    output_path: str | None = None,
) -> None:
    source_path = Path(realtalk_path)
    chat = load_json(source_path)
    user_name = detect_user_speaker(chat)
    agent_name = detect_agent_speaker(chat)
    persona_path = f"agent/{agent_name}_persona.json"
    profile_name = user_name + "_" + agent_name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[Profile Generation] start", f"source={source_path}", f"speaker={user_name}")
    agent = StateDrivenCompanionAgent(profile_path=profile_path, persona_path=persona_path, user_name=profile_name)

    memory_path = Path("data") / "realtalk_memory_runs" / f"{profile_name}_{run_id}"

    checkpoint = load_checkpoint(memory_path) if resume else None
    if start_turn is not None:
        replay_start_turn = max(1, start_turn)
        resume_source = "--start-turn"
    elif checkpoint:
        last_completed_turn = int(checkpoint.get("last_completed_turn", 0) or 0)
        replay_start_turn = max(1, last_completed_turn - replay_window + 1)
        resume_source = "checkpoint"
    else:
        replay_start_turn = 1
        resume_source = "fresh"

    agent.memory_manager = MemoryOSLocal(
        persist_path=str(memory_path)
    )
    print(f"[Pipeline] resume={resume} resume_source={resume_source} replay_start_turn={replay_start_turn}")

    target_turns = 0
    current_session = ""
    turns = flatten_dialogue(chat)
    for turn in turns:
        if turn["turn_index"] < replay_start_turn or turn["speaker"] != user_name:
            continue

        if turn["session_id"] != current_session:
            current_session = turn["session_id"]
            print(f"[Pipeline] session key={current_session}")

        agent.observe_dialogue_turn("user", turn["content"])

        target_turns += 1

        save_checkpoint(memory_path, {
            "source_path": str(source_path),
            "user_name": user_name,
            "profile_path": agent.profile_path,
            "memory_path": str(memory_path),
            "last_completed_turn": turn["turn_index"],
            "updated_at": datetime.now().isoformat(),
        })

    finalize_result = agent.finalize_session()

    # 保存 agent 内部格式画像
    save_json(agent.profile_path, agent.user_profile)
    print(f"[Pipeline] agent profile saved path={agent.profile_path}")

    # 转换为简洁展示格式并保存
    print("[Pipeline] reformatting profile to display format...")
    llm = LLMClient()
    display_profile = reformat_profile(llm, agent.user_profile)

    if output_path:
        save_path = Path(output_path)
    else:
        save_path = source_path.parent / f"{user_name.lower().replace(' ', '_')}_profile_display.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(save_path), display_profile)
    print(f"[Pipeline] display profile saved path={save_path}")

    result: Dict[str, Any] = {
        "source_path": str(source_path),
        "user_name": user_name,
        "profile_path": agent.profile_path,
        "display_profile_path": str(save_path),
        "memory_path": str(memory_path),
        "replay_start_turn": replay_start_turn,
        "target_turns": target_turns,
        "finalize_result": finalize_result,
    }

    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a user profile from a REALTALK chat file.")
    parser.add_argument("--realtalk", required=True, help="Path to a REALTALK chat JSON file.")
    parser.add_argument("--profile-path", default="user", help="Defaults to {user_name}_{agent_name}_profile.json.")
    parser.add_argument("--output-path", default=None, help="Output path for the display-format profile.")
    parser.add_argument("--start-turn", type=int, default=None, help="Replay from this 1-based dialogue turn.")
    parser.add_argument("--replay-window", type=int, default=20, help="When resuming from checkpoint, replay this many previous turns.")
    args = parser.parse_args()

    run_profile_pipeline(
        realtalk_path=args.realtalk,
        profile_path=args.profile_path,
        output_path=args.output_path,
        start_turn=args.start_turn,
        replay_window=args.replay_window
    )

if __name__ == "__main__":
    main()
