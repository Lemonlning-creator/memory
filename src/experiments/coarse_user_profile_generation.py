from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..llm_client import LLMClient
from ..utils import load_json, parse_json, save_json
from .agent_persona_generation import detect_agent_speaker, session_keys


PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")

COARSE_PROFILE_SYSTEM_PROMPT = """你是用户画像提取专家。请根据双人对话，为指定的人类用户生成粗粒度用户画像。

画像只包含以下五个最外层维度：
- core：核心恐惧、核心欲望、价值观、依恋倾向和意义来源的总体概括。
- regulation：面对压力、冲突和不确定性时的主要调节与应对方式。
- cognition：表达方式、信息处理、情绪显露、社交距离和决策风格的总体概括。
- identity：用户明确表达或可以谨慎推断的身份、关系、生活条件和所处环境。
- behavior：内容兴趣、消费或娱乐偏好、习惯和长期行为模式的总体概括。

要求：
1. 只分析目标用户，不要把对话伙伴的特征写入用户画像。
2. 每个维度直接写一至两段简洁的中文描述，不要创建任何下级字段。
3. 不要输出 confidence、evidence、value 或 memory_ids 等结构。
4. 只根据对话中稳定、重复或明确表达的信息归纳，不要把一次性的情绪和事件写成稳定特征。
5. 不要编造职业、年龄、家庭、经济状况或心理特征。证据不足时使用审慎措辞，明确表示相关信息有限。
6. 对话伙伴的发言只能帮助理解上下文，不能作为用户特征的证据；画像中的每个判断都必须能由目标用户自己的发言支持。
7. 禁止把“玩了很久”改写成“沉迷”，禁止把停止无效争论直接定性为“顺从”或“回避”。“沉迷”“依赖”“顺从”“回避”“人格”等负面或心理学标签，只有在目标用户明确这样自述且有重复证据时才能使用；否则必须改用“喜欢”“有时”“降低投入”“不再继续争论”“当前表现为”等中性、有限的表述。
8. 只返回合法 JSON，不要输出 Markdown 或解释文字。

必须严格返回以下结构，五个值都必须是字符串：
{
  "core": "一至两段粗略描述",
  "regulation": "一至两段粗略描述",
  "cognition": "一至两段粗略描述",
  "identity": "一至两段粗略描述",
  "behavior": "一至两段粗略描述"
}
"""

COARSE_PROFILE_USER_PROMPT = """目标用户：{user_name}
对话伙伴：{agent_name}

以下输入已经过清洗，只保留说话人名称以及每条消息的 speaker 和 clean_text：
{dialogue}

请生成目标用户 {user_name} 的五层粗粒度用户画像。"""


def detect_user_speaker(chat: Dict[str, Any]) -> str:
    names = chat.get("name", {})
    if isinstance(names, dict) and names.get("speaker_1"):
        return str(names["speaker_1"])
    return "default_user"


def collect_dialogue_messages(chat: Dict[str, Any]) -> List[Dict[str, str]]:
    """按原始顺序收集对话，只保留 speaker 和 clean_text。"""
    messages: List[Dict[str, str]] = []
    for key in session_keys(chat):
        for message in chat[key]:
            speaker = str(message.get("speaker") or "").strip()
            clean_text = str(message.get("clean_text") or "").strip()
            if not speaker or not clean_text:
                continue
            messages.append({"speaker": speaker, "clean_text": clean_text})
    return messages


def _evenly_sample(items: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit <= 0:
        raise ValueError("max_utterances must be greater than 0")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]

    last_index = len(items) - 1
    indices = [round(index * last_index / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def _dialogue_payload(
    names: Dict[str, str],
    messages: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    return {"name": names, "messages": list(messages)}


def _serialized_payload(names: Dict[str, str], messages: Sequence[Dict[str, str]]) -> str:
    return json.dumps(
        _dialogue_payload(names, messages),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_compact_dialogue(
    chat: Dict[str, Any],
    max_utterances: int = 180,
    max_chars: int = 24000,
) -> str:
    """构造只含 name、speaker、clean_text 的紧凑模型输入。"""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    raw_names = chat.get("name", {})
    names = {
        key: str(raw_names[key])
        for key in ("speaker_1", "speaker_2")
        if isinstance(raw_names, dict) and raw_names.get(key)
    }
    messages = _evenly_sample(collect_dialogue_messages(chat), max_utterances)
    if not messages:
        raise ValueError("No valid dialogue messages found")

    # 如果字符预算不足，逐步稀疏采样，同时始终保留最后一条发言。
    while len(messages) > 1 and len(_serialized_payload(names, messages)) > max_chars:
        reduced = messages[::2]
        if reduced[-1] is not messages[-1]:
            reduced.append(messages[-1])
        messages = reduced

    serialized = _serialized_payload(names, messages)
    if len(serialized) <= max_chars:
        return serialized

    # 单条消息本身过长时，仅截断 clean_text，不引入其他字段。
    message = dict(messages[0])
    suffix = "……"
    low, high = 0, len(message["clean_text"])
    while low < high:
        middle = (low + high + 1) // 2
        candidate = dict(message)
        candidate["clean_text"] = message["clean_text"][:middle] + suffix
        if len(_serialized_payload(names, [candidate])) <= max_chars:
            low = middle
        else:
            high = middle - 1
    message["clean_text"] = message["clean_text"][:low] + suffix
    serialized = _serialized_payload(names, [message])
    if len(serialized) > max_chars:
        raise ValueError("max_chars is too small to hold the dialogue structure")
    return serialized


def validate_coarse_profile(profile: Dict[str, Any]) -> Dict[str, str]:
    """校验并按固定顺序返回五层粗略画像。"""
    if not isinstance(profile, dict):
        raise ValueError("LLM output must be a JSON object")

    expected = set(PROFILE_LAYERS)
    actual = set(profile)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Invalid profile layers: missing={missing}, extra={extra}")

    result: Dict[str, str] = {}
    for layer in PROFILE_LAYERS:
        value = profile[layer]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Profile layer '{layer}' must be a non-empty string")
        result[layer] = value.strip()
    return result


def generate_coarse_profile(
    realtalk_path: str | Path,
    output_path: str | Path | None = None,
    config_path: str = "config.ini",
    max_utterances: int = 180,
    max_chars: int = 24000,
) -> Dict[str, Any]:
    source_path = Path(realtalk_path)
    chat = load_json(str(source_path))
    user_name = detect_user_speaker(chat)
    agent_name = detect_agent_speaker(chat)
    dialogue = build_compact_dialogue(
        chat,
        max_utterances=max_utterances,
        max_chars=max_chars,
    )

    print(
        "[Coarse Profile Generation] start",
        f"source={source_path}",
        f"speaker={user_name}",
    )
    llm = LLMClient(config_path)
    raw_result = llm.chat(
        COARSE_PROFILE_SYSTEM_PROMPT,
        COARSE_PROFILE_USER_PROMPT.format(
            user_name=user_name,
            agent_name=agent_name,
            dialogue=dialogue,
        ),
        temperature=0.2,
        max_tokens=1600,
    )
    profile = validate_coarse_profile(parse_json(raw_result))

    if output_path is None:
        safe_user = re.sub(r"[^0-9A-Za-z_\-一-鿿]+", "_", user_name).strip("_")
        safe_agent = re.sub(r"[^0-9A-Za-z_\-一-鿿]+", "_", agent_name).strip("_")
        output_path = Path("user") / f"{safe_user}_{safe_agent}_coarse_profile.json"
    target_path = Path(output_path)
    save_json(str(target_path), profile)
    print(f"[Coarse Profile Generation] saved={target_path}")

    return {
        "source_path": str(source_path),
        "user_name": user_name,
        "agent_name": agent_name,
        "output_path": str(target_path),
        "profile": profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a five-layer coarse user profile from a REALTALK chat JSON file."
    )
    parser.add_argument("--realtalk", required=True, help="Path to a REALTALK chat JSON file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to user/{user_name}_{agent_name}_coarse_profile.json.",
    )
    parser.add_argument("--config", default="config.ini", help="LLM config path.")
    parser.add_argument("--max-utterances", type=int, default=180, help="Max messages sent to LLM.")
    parser.add_argument("--max-chars", type=int, default=24000, help="Max characters in dialogue JSON.")
    args = parser.parse_args()

    generate_coarse_profile(
        realtalk_path=args.realtalk,
        output_path=args.output,
        config_path=args.config,
        max_utterances=args.max_utterances,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
