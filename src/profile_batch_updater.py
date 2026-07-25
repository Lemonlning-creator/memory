from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from dotenv import load_dotenv

from .logger import logger
from .profile_schema import PROFILE_FIELDS, PROFILE_LAYERS, normalize_bare_profile
from .utils import load_json

load_dotenv()

PROFILE_TIMEZONE = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(PROFILE_TIMEZONE).isoformat()


PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
PROFILE_FIELDS: Dict[str, tuple[str, ...]] = {
    "core": ("fears", "desires", "values", "attachment style", "sources of meaning"),
    "regulation": ("avoidance", "control", "people-pleasing", "aggression", "humor", "obsession", "rationalization"),
    "cognition": ("expression style", "information density", "emotional visibility", "social distance", "decision style"),
    "identity": ("occupation", "age", "social relationships", "family", "economy", "devices", "physical environment"),
    "behavior": ("content preferences", "consumption preferences", "entertainment preferences", "habits", "long-term behavior patterns"),
}


def _field_aliases(layer: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for canonical in PROFILE_FIELDS[layer]:
        aliases[canonical] = canonical
        aliases[canonical.replace(" ", "_").replace("-", "_")] = canonical
    return aliases

SYSTEM_PROMPT = """你是陪伴智能体的用户画像更新器。你只根据本批原始对话中的用户消息更新画像，不使用中期或长期记忆。
目标是帮助智能体形成“它正在陪伴的这个人是什么样的人”的稳定认识，而不是复述用户对自己的标签、测试指令或本次画像生成过程。

输出格式：
- 必须输出一个 JSON 对象，不要输出 Markdown 或解释。
- 顶层只能有 layers；layers 必须且只能包含 core、regulation、cognition、identity、behavior。
- 每层必须包含 summary 和 attributes。summary 可以是 null，或包含 value、confidence、evidence_message_ids 的对象；但只要该层 attributes 有任何更新，summary 就必须同步更新且不可为 null。
- attributes 只能使用给定白名单字段。字段名必须逐字复制白名单，包括其中的空格和连字符；不要改成 snake_case。
- 每个 summary 和属性值必须是包含 value、confidence、evidence_message_ids 的对象，不能直接返回字符串或数组。

画像写法：
- 这是陪伴 agent 对眼前这个人的内部认识；最终文字要像已写入画像的直接判断，不要写成“我在观察谁”或“对谁作评估”的报告。
- summary 只归纳同层、已有证据支持的属性，使用一到两句朴素具体的判断句。优先把相邻属性自然合成一句，例如“重视稳定和长期发展，希望在现实基础上持续成长”，不要给人格下定义或交代推断过程。
- 五层 summary 风格严格参照 deployment 分支预设画像：
  core："重视稳定和长期发展，希望在现实基础上持续成长。"
  regulation："面对压力时容易产生焦虑，但通常不会停下行动，而是边担心边推进事情。"
  cognition："思考方式偏务实，关注现实可行性，喜欢从经验和实际效果分析问题。"
  identity："人工智能专业学生，当前正处于毕业与求职并行的重要阶段。"
  behavior："近期行为重心围绕毕业、求职和 AI 工具使用展开，娱乐偏好具有较强的长期连续性。"
- 属性也使用同样的直接句式，例如“重视稳定和长期发展，而非短期收益”“优先选择现实可落地的方案”“倾向使用层次化、结构化的方式组织和表达信息”。
- 禁止过程、观察者和标签化措辞，例如“最终画像呈现为”“本次对话显示”“深层价值收敛为”“人格底色是”“用户表示自己是”“总结来说”“可见其”“说明对方”。不得以“他/她/对方/用户”开头。
- 不要直接搬运用户的自我描述或任务要求；要从自然话题中的选择、反应、偏好、取舍和反复出现的行为中归纳。
- 如果用户直接要求你给自己画像、描述希望生成什么画像、或进行测试验收，这些只是任务指令，不应作为人格证据写入画像。
- 可以综合旧画像与本批新证据更新 summary，但变化必须有本批证据支撑；旧画像只用于理解当前值，不代表本批证据。

五层语义：
- core：深层动机、担忧、价值观、依恋模式和意义来源；描述用户长期在意什么、害怕什么、被什么驱动。
- regulation：压力、冲突、不确定性和情绪波动下的调节方式；描述用户如何控制、回避、讨好、坚持或合理化。
- cognition：表达、信息密度、情绪可见度、社交距离和决策风格；描述用户怎样组织信息、判断和沟通。
- identity：职业/阶段、关系、家庭、经济、设备和环境等相对客观身份与处境；证据不足不要臆测。
- behavior：内容/消费/娱乐偏好、习惯和长期行为模式；描述可观察的偏好和反复行为。

证据规则：
- 属性只记录用户明确表达或可被多条消息直接支持的稳定信息。证据不足、一次性情绪、助手诱导内容、测试脚本内容、推测和矛盾信息不要更新。
- 所有 evidence_message_ids 必须来自本批用户消息。绝对不要复用旧画像里的证据 ID；不要复制旧画像中没有新证据支持的变化。"""

META_LANGUAGE_MARKERS = (
    "最终画像", "本次对话", "这次对话", "本批对话", "本批消息", "用户表示",
    "用户自称", "用户说自己", "总结来说", "画像呈现", "呈现为", "收敛为",
    "人格底色", "核心底色", "作为测试", "测试中",
)


class ProfileUpdateError(ValueError):
    pass


def _reject_meta_language(value: str, path: str) -> None:
    for marker in META_LANGUAGE_MARKERS:
        if marker in value:
            raise ProfileUpdateError(f"{path} contains report-style wording: {marker}")
    if value.lstrip().startswith(("他", "她", "对方")):
        raise ProfileUpdateError(f"{path} must be a direct profile statement")


def _nullable_string_array_schema() -> Dict[str, Any]:
    return {
        "anyOf": [
            {"type": "null"},
            {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        ]
    }


def build_profile_response_format() -> Dict[str, Any]:
    """Strict fixed-field response schema. Evidence is not part of model output."""
    layer_properties: Dict[str, Any] = {}
    for layer in PROFILE_LAYERS:
        layer_properties[layer] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1}]},
                **{field: _nullable_string_array_schema() for field in PROFILE_FIELDS[layer]},
            },
            "required": ["summary", *PROFILE_FIELDS[layer]],
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "profile_patch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "layers": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": layer_properties,
                        "required": list(PROFILE_LAYERS),
                    }
                },
                "required": ["layers"],
            },
        },
    }


def _atomic_save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True)
class RawDialogueTurn:
    message_id: str
    user: str
    created_at: str

    def as_dict(self) -> Dict[str, str]:
        return {"message_id": self.message_id, "user": self.user, "created_at": self.created_at}


def _clean_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileUpdateError(f"{path} must be a non-empty string")
    result = value.strip()
    _reject_meta_language(result, path)
    return result


def _clean_values(value: Any, path: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ProfileUpdateError(f"{path} must be a non-empty string array")
    if len(value) > 12:
        raise ProfileUpdateError(f"{path} has too many values")
    cleaned: List[str] = []
    for index, item in enumerate(value):
        item_value = _clean_string(item, f"{path}[{index}]")
        if len(item_value) > 240:
            raise ProfileUpdateError(f"{path}[{index}] is too long")
        if item_value not in cleaned:
            cleaned.append(item_value)
    return cleaned


def validate_patch(data: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(data, Mapping) or set(data) != {"layers"}:
        raise ProfileUpdateError("patch must contain only layers")
    layers = data.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != set(PROFILE_LAYERS):
        raise ProfileUpdateError("patch must contain exactly the fixed five layers")

    result: Dict[str, Dict[str, Any]] = {}
    for layer in PROFILE_LAYERS:
        section = layers[layer]
        required = {"summary", *PROFILE_FIELDS[layer]}
        if not isinstance(section, Mapping) or set(section) != required:
            raise ProfileUpdateError(f"layers.{layer} has invalid fields")
        normalized: Dict[str, Any] = {}
        summary = section["summary"]
        if summary is not None:
            normalized["summary"] = _clean_string(summary, f"layers.{layer}.summary")
        for field in PROFILE_FIELDS[layer]:
            raw = section[field]
            if raw is not None:
                normalized[field] = _clean_values(raw, f"layers.{layer}.{field}")
        result[layer] = normalized
    return result


def merge_patch(profile: Mapping[str, Any], patch: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Apply only explicit field updates and keep every other stored value."""
    merged = normalize_bare_profile(profile)
    for layer in PROFILE_LAYERS:
        updates = patch.get(layer, {})
        for field in ("summary", *PROFILE_FIELDS[layer]):
            if field in updates:
                merged[layer][field] = copy.deepcopy(updates[field])
    return merged


class KimiProfileExtractor:
    """Independent OpenAI-compatible Profile extractor with task-local retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_attempts: int = 2,
        timeout_seconds: float = 60.0,
        client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PROFILE_API_KEY")
        self.base_url = base_url or os.getenv("PROFILE_BASE_URL", "https://api.moonshot.cn/v1")
        self.model = model or os.getenv("PROFILE_MODEL", "kimi-k2.6")
        self.max_attempts = max(1, max_attempts)
        self.client = client
        if self.client is None and self.api_key:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProfileUpdateError("openai dependency is required for profile extraction") from exc
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_seconds)

    @property
    def available(self) -> bool:
        return self.client is not None

    def extract(self, current_profile: Mapping[str, Any], turns: Iterable[Mapping[str, str]]) -> Dict[str, Dict[str, Any]]:
        if not self.available:
            raise ProfileUpdateError("PROFILE_API_KEY is not configured")
        turn_list = list(turns)
        payload = {
            "field_whitelist": {layer: list(PROFILE_FIELDS[layer]) for layer in PROFILE_LAYERS},
            "current_profile": normalize_bare_profile(current_profile),
            "raw_dialogue_batch": turn_list,
            "output_example": {
                "layers": {
                    layer: {"summary": None, **{field: None for field in PROFILE_FIELDS[layer]}}
                    for layer in PROFILE_LAYERS
                }
            },
        }
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            raw = ""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=3000,
                    response_format=build_profile_response_format(),
                    extra_body={"thinking": {"type": "disabled"}},
                )
                raw = (response.choices[0].message.content or "").strip()
                return validate_patch(json.loads(raw))
            except Exception as exc:
                last_error = exc
                logger.warning("[PROFILE_BATCH] attempt=%s/%s failed: %s", attempt, self.max_attempts, exc)
                if attempt < self.max_attempts:
                    if raw:
                        messages.append({"role": "assistant", "content": raw[:12000]})
                    messages.append({"role": "user", "content": "上次输出校验失败：" + str(exc)[:1200] + "。请仅返回修正后的完整 JSON。"})
        raise ProfileUpdateError(f"profile extraction failed after {self.max_attempts} attempts: {last_error}")


def _validate_item(item: Any, allowed_ids: set[str], path: str) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ProfileUpdateError(f"{path} must be an object")
    if set(item) != {"value", "confidence", "evidence_message_ids"}:
        raise ProfileUpdateError(f"{path} has invalid keys")
    value = item["value"]
    confidence = item["confidence"]
    evidence_ids = item["evidence_message_ids"]
    if not isinstance(value, str) or not value.strip():
        raise ProfileUpdateError(f"{path}.value must be a non-empty string")
    _reject_meta_language(value, path)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
        raise ProfileUpdateError(f"{path}.confidence must be between 0 and 1")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ProfileUpdateError(f"{path}.evidence_message_ids must be a non-empty list")
    if any(not isinstance(message_id, str) or message_id not in allowed_ids for message_id in evidence_ids):
        raise ProfileUpdateError(f"{path} references evidence outside this batch")
    return {
        "value": value.strip(),
        "confidence": round(float(confidence), 3),
        "evidence_message_ids": list(dict.fromkeys(evidence_ids)),
    }


def normalize_patch_field_names(data: Any) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get("layers"), dict):
        return data
    normalized = copy.deepcopy(data)
    for layer, section in normalized["layers"].items():
        if layer not in PROFILE_FIELDS or not isinstance(section, dict):
            continue
        attributes = section.get("attributes")
        if not isinstance(attributes, dict):
            continue
        aliases = _field_aliases(layer)
        normalized_attributes: Dict[str, Any] = {}
        for supplied, value in attributes.items():
            canonical = aliases.get(supplied, supplied)
            if canonical in normalized_attributes:
                raise ProfileUpdateError(f"layers.{layer}.attributes contains duplicate aliases for {canonical}")
            normalized_attributes[canonical] = value
        section["attributes"] = normalized_attributes
    return normalized


def validate_patch(data: Any, allowed_ids: set[str]) -> Dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"layers"}:
        raise ProfileUpdateError("top level must contain only layers")
    layers = data["layers"]
    if not isinstance(layers, dict) or set(layers) != set(PROFILE_LAYERS):
        raise ProfileUpdateError("layers must contain exactly the five profile layers")

    clean: Dict[str, Any] = {"layers": {}}
    for layer in PROFILE_LAYERS:
        section = layers[layer]
        if not isinstance(section, dict) or set(section) != {"summary", "attributes"}:
            raise ProfileUpdateError(f"layers.{layer} must contain summary and attributes")
        summary = section["summary"]
        clean_summary = None if summary is None else _validate_item(summary, allowed_ids, f"layers.{layer}.summary")
        attributes = section["attributes"]
        if not isinstance(attributes, dict):
            raise ProfileUpdateError(f"layers.{layer}.attributes must be an object")
        unknown = set(attributes) - set(PROFILE_FIELDS[layer])
        if unknown:
            raise ProfileUpdateError(f"layers.{layer}.attributes contains unknown fields: {sorted(unknown)}")
        clean_attributes = {
            field: _validate_item(item, allowed_ids, f"layers.{layer}.attributes.{field}")
            for field, item in attributes.items()
        }
        if clean_attributes and clean_summary is None:
            raise ProfileUpdateError(f"layers.{layer}.summary is required when attributes change")
        clean["layers"][layer] = {"summary": clean_summary, "attributes": clean_attributes}
    return clean


def merge_patch(profile: Mapping[str, Any], patch: Mapping[str, Any], turns: Iterable[Mapping[str, str]]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(profile))
    static_profile = result.setdefault("state_axis", {}).setdefault("static_profile", {})
    turn_map = {turn["message_id"]: turn["user"] for turn in turns}

    for layer in PROFILE_LAYERS:
        layer_profile = static_profile.setdefault(layer, {})
        layer_patch = patch["layers"][layer]
        updates = dict(layer_patch["attributes"])
        if layer_patch["summary"] is not None:
            updates["summary"] = layer_patch["summary"]
        for field, item in updates.items():
            evidence_ids = item["evidence_message_ids"]
            evidence = " | ".join(turn_map[message_id][:240] for message_id in evidence_ids)
            old = layer_profile.get(field, {})
            memory_ids = old.get("memory_ids", []) if isinstance(old, dict) else []
            layer_profile[field] = {
                "value": item["value"],
                "confidence": item["confidence"],
                "evidence": evidence,
                "evidence_message_ids": evidence_ids,
                "memory_ids": memory_ids,
                "updated_at": _now_iso(),
            }
    return result


class ProfileBatchUpdater:
    """Persistent raw-dialogue queue. Only this Profile path performs noise judgement."""

    def __init__(
        self,
        profile_path: str,
        extractor: Optional[KimiProfileExtractor] = None,
        min_user_messages: Optional[int] = None,
        max_wait_seconds: Optional[int] = None,
        on_profile_updated: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.profile_path = Path(profile_path)
        self.queue_path = self.profile_path.with_suffix(self.profile_path.suffix + ".pending.json")
        self.extractor = extractor or KimiProfileExtractor()
        message_threshold = min_user_messages if min_user_messages is not None else int(os.getenv("PROFILE_BATCH_MESSAGES", "8"))
        wait_threshold = max_wait_seconds if max_wait_seconds is not None else int(os.getenv("PROFILE_BATCH_SECONDS", "900"))
        self.min_user_messages = max(1, message_threshold)
        self.max_wait_seconds = max(1, wait_threshold)
        self.on_profile_updated = on_profile_updated
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._running = False
        self._timer: Optional[threading.Timer] = None
        self._schedule_existing_queue()

    def _load_queue(self) -> Dict[str, Any]:
        data = load_json(str(self.queue_path)) if self.queue_path.exists() else {}
        turns = data.get("turns", []) if isinstance(data, dict) else []
        return {"first_enqueued_at": data.get("first_enqueued_at") if isinstance(data, dict) else None, "turns": turns}

    def _save_queue(self, queue: Mapping[str, Any]) -> None:
        _atomic_save_json(self.queue_path, queue)

    def submit_turn(self, user: str) -> str:
        now = _now_iso()
        turn = RawDialogueTurn(uuid.uuid4().hex, user.strip(), now)
        with self._lock:
            queue = self._load_queue()
            if not queue["turns"]:
                queue["first_enqueued_at"] = now
            queue["turns"].append(turn.as_dict())
            self._save_queue(queue)
            if len(queue["turns"]) >= self.min_user_messages:
                self._start_worker_locked()
            else:
                self._schedule_timer_locked(queue)
        return turn.message_id

    def _schedule_existing_queue(self) -> None:
        with self._lock:
            queue = self._load_queue()
            if queue["turns"]:
                if len(queue["turns"]) >= self.min_user_messages:
                    self._start_worker_locked()
                else:
                    self._schedule_timer_locked(queue)

    def _schedule_timer_locked(self, queue: Mapping[str, Any]) -> None:
        if self._timer and self._timer.is_alive():
            return
        first = queue.get("first_enqueued_at")
        elapsed = 0.0
        if isinstance(first, str):
            try:
                elapsed = max(0.0, time.time() - datetime.fromisoformat(first).timestamp())
            except ValueError:
                pass
        self._timer = threading.Timer(max(0.05, self.max_wait_seconds - elapsed), self._timer_elapsed)
        self._timer.daemon = True
        self._timer.start()

    def _timer_elapsed(self) -> None:
        with self._lock:
            self._timer = None
            self._start_worker_locked()

    def _start_worker_locked(self) -> None:
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        succeeded = False
        try:
            succeeded = self.process_pending()
        finally:
            with self._lock:
                self._running = False
                queue = self._load_queue()
                if succeeded and queue["turns"]:
                    if len(queue["turns"]) >= self.min_user_messages:
                        self._start_worker_locked()
                    else:
                        self._schedule_timer_locked(queue)

    def process_pending(self) -> bool:
        with self._process_lock:
            return self._process_pending()

    def _process_pending(self) -> bool:
        with self._lock:
            queue = self._load_queue()
            turns = list(queue["turns"])
        if not turns:
            return False
        if not self.extractor.available:
            logger.warning("[PROFILE_BATCH] pending turns kept because PROFILE_API_KEY is not configured")
            return False
        consumed_ids = {turn["message_id"] for turn in turns}
        try:
            current = load_json(str(self.profile_path)) if self.profile_path.exists() else {}
            patch = self.extractor.extract(normalize_bare_profile(current), turns)
            merged = merge_patch(current, patch)
            _atomic_save_json(self.profile_path, merged)
        except Exception as exc:
            logger.exception("[PROFILE_BATCH] batch failed; pending turns retained: %s", exc)
            return False
        with self._lock:
            latest = self._load_queue()
            remaining = [turn for turn in latest["turns"] if turn.get("message_id") not in consumed_ids]
            self._save_queue({"first_enqueued_at": remaining[0]["created_at"] if remaining else None, "turns": remaining})
        if self.on_profile_updated:
            self.on_profile_updated(merged)
        logger.info("[PROFILE_BATCH] processed %s raw dialogue turns", len(turns))
        return True
