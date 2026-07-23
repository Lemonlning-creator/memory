from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .logger import logger
from .utils import load_json

PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
PROFILE_FIELDS: Dict[str, tuple[str, ...]] = {
    "core": ("fears", "desires", "values", "attachment style", "sources of meaning"),
    "regulation": ("avoidance", "control", "people-pleasing", "aggression", "humor", "obsession", "rationalization"),
    "cognition": ("expression style", "information density", "emotional visibility", "social distance", "decision style"),
    "identity": ("occupation", "age", "social relationships", "family", "economy", "devices", "physical environment"),
    "behavior": ("content preferences", "consumption preferences", "entertainment preferences", "habits", "long-term behavior patterns"),
}

SYSTEM_PROMPT = """你是用户画像更新器。你只根据本批原始对话中的用户消息更新画像，不使用中期或长期记忆。
必须输出一个 JSON 对象，不要输出 Markdown 或解释。顶层只能有 layers；layers 必须且只能包含 core、regulation、cognition、identity、behavior。
每层必须包含 summary 和 attributes。summary 可以是 null，或包含 value、confidence、evidence_message_ids 的对象；但只要该层 attributes 有任何更新，summary 就必须同步更新且不可为 null。attributes 只能使用给定白名单字段。
summary 用一句话概括该层当前画像，可综合旧画像与本批新证据；属性只记录用户明确表达或可被直接支持的稳定信息。证据不足、一次性情绪、助手诱导内容、推测和矛盾信息不要更新。
所有 evidence_message_ids 必须来自本批用户消息。不要复制旧画像中没有新证据支持的变化。"""


class ProfileUpdateError(ValueError):
    pass


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
        return {
            "message_id": self.message_id,
            "user": self.user,
            "created_at": self.created_at,
        }


class KimiProfileExtractor:
    """Independent OpenAI-compatible profile extractor with task-local correction retries."""

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

    def extract(self, current_profile: Mapping[str, Any], turns: Iterable[Mapping[str, str]]) -> Dict[str, Any]:
        if not self.available:
            raise ProfileUpdateError("PROFILE_API_KEY is not configured")

        turn_list = list(turns)
        allowed_ids = {turn["message_id"] for turn in turn_list}
        payload = {
            "field_whitelist": {key: list(value) for key, value in PROFILE_FIELDS.items()},
            "current_static_profile": current_profile,
            "raw_dialogue_batch": turn_list,
            "output_example": {
                "layers": {
                    layer: {"summary": None, "attributes": {}}
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
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = json.loads(raw)
                return validate_patch(parsed, allowed_ids)
            except Exception as exc:
                last_error = exc
                logger.warning("[PROFILE_BATCH] attempt=%s/%s failed: %s", attempt, self.max_attempts, exc)
                if attempt < self.max_attempts:
                    if raw:
                        messages.append({"role": "assistant", "content": raw[:12000]})
                    messages.append({
                        "role": "user",
                        "content": "上次输出校验失败：" + str(exc)[:1200] + "。请仅返回修正后的完整 JSON。",
                    })
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
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
    return result


class ProfileBatchUpdater:
    """Small persistent raw-dialogue queue; triggers by count or age and removes turns only after success."""

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
        now = datetime.now(timezone.utc).isoformat()
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
        delay = max(0.05, self.max_wait_seconds - elapsed)
        self._timer = threading.Timer(delay, self._timer_elapsed)
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
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

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
            static_profile = current.get("state_axis", {}).get("static_profile", {})
            patch = self.extractor.extract(static_profile, turns)
            merged = merge_patch(current, patch, turns)
            _atomic_save_json(self.profile_path, merged)
        except Exception as exc:
            logger.exception("[PROFILE_BATCH] batch failed; pending turns retained: %s", exc)
            return False

        with self._lock:
            latest = self._load_queue()
            remaining = [turn for turn in latest["turns"] if turn.get("message_id") not in consumed_ids]
            self._save_queue({
                "first_enqueued_at": remaining[0]["created_at"] if remaining else None,
                "turns": remaining,
            })
        if self.on_profile_updated:
            self.on_profile_updated(merged)
        logger.info("[PROFILE_BATCH] updated profile from %s raw dialogue turns", len(turns))
        return True
