from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from dotenv import load_dotenv
from openai import OpenAI

from ..llm_client import LLMClient
from ..utils import parse_json, load_json, save_json

load_dotenv()
from .agent_persona_generation import detect_agent_speaker
from .user_profile_generation import detect_user_speaker
from ..prompts.templates import (
    EVIDENCE_JUDGE_SYSTEM_PROMPT,
    EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE,
)

EVIDENCE_TYPE_WEIGHTS = {
    "utterance": 0.6,
    "turn_window": 0.9,
    "event_summary": 0.7,
    "qa_evidence": 1.0,
}

@dataclass
class RetrievedEvidence:
    evidence: Dict[str, Any]
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0

def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            text = flatten_text(item)
            if text:
                parts.append(text)
        return "；".join(parts)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            key_text = flatten_text(key)
            value_text = flatten_text(item)
            if key_text and value_text:
                parts.append(f"{key_text}: {value_text}")
            elif value_text:
                parts.append(value_text)
            elif key_text:
                parts.append(key_text)
        return "；".join(parts)
    return str(value)

def session_keys(chat: Dict[str, Any]) -> List[str]:
    keys = [
        key for key, value in chat.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))

def format_turn(message: Dict[str, Any]) -> str:
    speaker = message.get("speaker", "")
    text = message.get("clean_text", "")
    return f"[{speaker}] {text}".strip()

def value_memory_ids(value: Any) -> List[str]:
    if isinstance(value, dict) and isinstance(value.get("memory_ids"), list):
        return [str(item) for item in value["memory_ids"]]
    return []

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def normalize_id(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", text).strip("_")

class EvidencePoolBuilder:
    def __init__(
        self,
        window_size: int = 2,
        include_event_summary: bool = False,
        include_qa_evidence: bool = False,
    ):
        self.window_size = window_size
        self.include_event_summary = include_event_summary
        self.include_qa_evidence = include_qa_evidence

    def build_from_file(self, file_path: str | Path) -> List[Dict[str, Any]]:
        source_path = Path(file_path)
        chat = load_json(source_path)
        user_name = detect_user_speaker(chat)
        agent_name = detect_agent_speaker(chat)
        evidence: List[Dict[str, Any]] = []
        dia_index: Dict[str, Dict[str, Any]] = {}

        for key in session_keys(chat):
            messages = chat[key]
            evidence.extend(self._build_session_evidence(
                session_id=key,
                messages=messages,
                user_name=user_name,
                agent_name=agent_name,
            ))
            if self.include_event_summary:
                evidence.extend(self._build_event_summary_evidence(
                    chat=chat,
                    session_id=key
                ))
            if self.include_qa_evidence:
                for message in messages:
                    if message.get("dia_id"):
                        dia_index[message["dia_id"]] = message
        if self.include_qa_evidence:
            evidence.extend(self._build_qa_evidence(chat, dia_index=dia_index))
        return evidence
    
    def _build_session_evidence(
        self,
        session_id: str,
        messages: Sequence[Dict[str, Any]],
        user_name: str,
        agent_name: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, message in enumerate(messages):
            if message.get("speaker") != user_name:
                continue
            dia_id = message.get("dia_id", f"{session_id}:{idx}")
            text = message.get("clean_text", "")
            rows.append({
                "evidence_id": f"{session_id}__{normalize_id(dia_id)}__utterance",
                "session_id": session_id,
                "dia_id": dia_id,
                "user_name": user_name,
                "agent_name": agent_name,
                "evidence_type": "utterance",
                "text": text,
                "time": message.get("date_time", "")
            })

            start = max(0, idx - self.window_size)
            end = min(len(messages), idx + self.window_size + 1)
            window = list(messages[start:end])
            window_text = "\n".join(format_turn(item) for item in window if item.get("clean_text"))
            rows.append({
                "evidence_id": f"{session_id}__{normalize_id(dia_id)}__window",
                "session_id": session_id,
                "dia_id": dia_id,
                "user_name": user_name,
                "agent_name": agent_name,
                "evidence_type": "turn_window",
                "text": window_text,
                "time_range": [
                    item.get("date_time", "") for item in window
                    if item.get("date_time")
                ]
            })
        return rows

    def _build_event_summary_evidence(
        self,
        chat: Dict[str, Any],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        event_key = f"events_{session_id}"
        events = chat.get(event_key)
        if not isinstance(events, dict):
            return []
        rows: List[Dict[str, Any]] = []
        for agent_key, event_list in events.items():
            if not isinstance(event_list, list):
                continue
            speaker = "agent_a" if agent_key == "agent_a" else "agent_b"
            for idx, event in enumerate(event_list):
                if not isinstance(event, dict):
                    continue
                text = flatten_text(event.get("sub-event") or event)
                if not text:
                    continue
                rows.append({
                    "evidence_id": f"{session_id}__{agent_key}_{idx}__event",
                    "evidence_type": "event_summary",
                    "speaker": speaker,
                    "text": text,
                    "event": event
                })
        return rows

    def _build_qa_evidence(
        self,
        chat: Dict[str, Any],
        dia_index: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        qa_items = chat.get("qa", [])
        if not isinstance(qa_items, list):
            return []
        rows: List[Dict[str, Any]] = []
        for idx, item in enumerate(qa_items):
            if not isinstance(item, dict):
                continue
            evidence_ids = [str(eid) for eid in item.get("evidence", [])]
            turns = [dia_index[eid] for eid in evidence_ids if eid in dia_index]
            text = (
                f"Question: {item.get('question', '')}\n"
                f"Answer: {item.get('answer', '')}\n"
                f"Evidence:\n" + "\n".join(format_turn(turn) for turn in turns)
            )
            rows.append({
                "evidence_id": f"qa_{idx}",
                "evidence_type": "qa_evidence",
                "text": text
            })
        return rows

class ProfileClaimExtractor:
    def __init__(self, atomize: bool = True):
        self.atomize = atomize

    def extract(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        static_profile = profile.get("state_axis", {}).get("static_profile", {})
        claims: List[Dict[str, Any]] = []
        self._walk(static_profile, [], claims)
        return claims

    def _walk(self, node: Any, path: List[str], claims: List[Dict[str, Any]]) -> None:
        if not isinstance(node, dict):
            self._append_claim(path, node, [], claims)
            return
        if "value" in node:
            self._append_claim(path, node.get("value"), value_memory_ids(node), claims)
            return
        for key, value in node.items():
            self._walk(value, path + [str(key)], claims)

    def _append_claim(
        self,
        path: List[str],
        value: Any,
        memory_ids: List[str],
        claims: List[Dict[str, Any]],
    ) -> None:
        text_value = flatten_text(value)
        if not text_value:
            return
        label = ".".join(path)
        parent_claim_id = f"claim_{len(claims) + 1:04d}"
        atom_values = self._atomize_value(value) if self.atomize else [text_value]
        for atom_index, atom in enumerate(atom_values, start=1):
            atom = atom.strip()
            if not atom:
                continue
            claim_id = parent_claim_id if len(atom_values) == 1 else f"{parent_claim_id}_{atom_index:02d}"
            claims.append({
                "claim_id": claim_id,
                "parent_claim_id": parent_claim_id,
                "path": path,
                "label": label,
                "value": atom,
                "source_value": value,
                "claim": f"{label}: {atom}",
                "memory_ids": memory_ids
            })

    def _atomize_value(self, value: Any) -> List[str]:
        if isinstance(value, list):
            atoms: List[str] = []
            for item in value:
                atoms.extend(self._atomize_value(item))
            return atoms
        text = flatten_text(value)
        separators = r"[；;。.\n\r]+"
        parts = [part.strip(" ，,、\t") for part in re.split(separators, text)]
        atoms = [part for part in parts if part]
        return atoms or ([text] if text else [])

class ProfileEvidenceEvaluator:
    def __init__(
        self,
        llm: LLMClient | None = None,
        config_path: str = "config.ini",
        top_k: int = 8,
        min_retrieval_score: float = 0.8,
        semantic_threshold: float = 0.2,
        use_embedding_retrieval: bool = True,
    ):
        self.llm = llm or LLMClient(config_path)
        self.config_path = config_path
        self.top_k = top_k
        self.min_retrieval_score = min_retrieval_score
        self.semantic_threshold = semantic_threshold
        self.use_embedding_retrieval = use_embedding_retrieval
        self._embedding_cache: Dict[str, List[float]] = {}

        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        api_config = config["API"]
        self.embedding_model_name = api_config.get("embedding_model", fallback="text-embedding-v4")
        self.embedding_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

    def retrieve(self, claim: Dict[str, Any], evidence_pool: Sequence[Dict[str, Any]]) -> List[RetrievedEvidence]:
        claim_text = claim.get("claim", "")
        retrieved: List[RetrievedEvidence] = []
        claim_embedding = self._embed_cached(
            f"claim:{claim.get('claim_id', claim_text)}",
            claim_text,
        ) if self.use_embedding_retrieval else None
        for evidence in evidence_pool:
            lexical_score = self._score_evidence(claim_text, evidence)
            semantic_score = 0.0
            if claim_embedding is not None:
                evidence_text = self._evidence_embedding_text(evidence)
                evidence_embedding = self._embed_cached(
                    f"evidence:{evidence.get('evidence_id', evidence_text)}",
                    evidence_text,
                )
                semantic_score = cosine_similarity(claim_embedding, evidence_embedding)
            combined_score = max(semantic_score, lexical_score) if self.use_embedding_retrieval else lexical_score
            if (
                semantic_score >= self.semantic_threshold
                or lexical_score >= self.min_retrieval_score
            ):
                retrieved.append(RetrievedEvidence(
                    evidence=evidence,
                    score=combined_score,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                ))
        return sorted(retrieved, key=lambda item: item.score, reverse=True)[: self.top_k]

    def _embed_text(self, text: str) -> List[float]:
        text = text.strip()
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=text,
                )
                embedding = list(response.data[0].embedding)
                return embedding
            except Exception as exc:
                last_error = exc
                wait_seconds = attempt * 3
                if attempt < 3:
                    time.sleep(wait_seconds)
        raise last_error

    def _embed_cached(self, cache_key: str, text: str) -> List[float]:
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self._embed_text(text)
        return self._embedding_cache[cache_key]

    def _evidence_embedding_text(self, evidence: Dict[str, Any]) -> str:
        target_text = flatten_text(evidence.get("target_text", ""))
        context_text = flatten_text(evidence.get("text", ""))
        return "\n".join(part for part in [target_text, context_text] if part)

    def judge_claim(self, claim: Dict[str, Any], retrieved: Sequence[RetrievedEvidence]) -> Dict[str, Any]:
        if not retrieved:
            return {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "support_level": "证据不足",
                "score": 0,
                "stability": "低",
                "abstraction_risk": "明显",
                "hallucination_risk": "可能存在",
                "supporting_evidence_ids": [],
                "counter_evidence_ids": [],
                "reason": "规则召回阶段没有找到达到阈值的候选证据。",
            }

        evidence_payload = [
            {
                "evidence_id": item.evidence["evidence_id"],
                "score": round(item.score, 3),
                "semantic_score": round(item.semantic_score, 3),
                "lexical_score": round(item.lexical_score, 3),
                "evidence_type": item.evidence.get("evidence_type"),
                "text": item.evidence.get("text", ""),
                "session_id": item.evidence.get("session_id", ""),
            }
            for item in retrieved
        ]
        prompt = EVIDENCE_JUDGE_USER_PROMPT_TEMPLATE.format(
            claim=json.dumps(claim, ensure_ascii=False, indent=2),
            evidence=json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        )
        raw = self.llm.chat(EVIDENCE_JUDGE_SYSTEM_PROMPT, prompt, temperature=0.1)
        result = parse_json(raw)
        result["claim_id"] = claim["claim_id"]
        result["claim"] = claim["claim"]
        result["retrieved_evidence"] = evidence_payload
        return result

    def evaluate(
        self,
        claims: Sequence[Dict[str, Any]],
        evidence_pool: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for claim in claims:
            retrieved = self.retrieve(claim, evidence_pool)
            result = self.judge_claim(claim, retrieved)
            results.append(result)
        return results

    def _score_evidence(
        self,
        claim_text: str,
        evidence: Dict[str, Any],
    ) -> float:
        score = EVIDENCE_TYPE_WEIGHTS.get(evidence.get("evidence_type", ""), 0.0)

        if self._is_style_claim(claim_text) and evidence.get("evidence_type") in {"utterance", "turn_window"}:
            score += 1.5
        if self._is_ability_or_preference_claim(claim_text) and evidence.get("evidence_type") in {"turn_window"}:
            score += 0.8

        length = len(flatten_text(evidence.get("text", "")))
        if 60 <= length <= 500:
            score += 0.4
        return score

    def _is_style_claim(self, claim_text: str) -> bool:
        return any(word in claim_text for word in ["表达", "语言", "口头禅", "语气", "风格", "幽默"])

    def _is_ability_or_preference_claim(self, claim_text: str) -> bool:
        return any(word in claim_text for word in ["能力", "偏好", "喜欢", "习惯", "目标", "兴趣", "模式"])

def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "claim_count": 0,
            "supported_count": 0,
            "partial_supported_count": 0,
            "unsupported_count": 0,
            "insufficient_count": 0,
            "evidence_support_rate": 0.0,
            "average_score": 0.0,
        }
    support_levels = [str(item.get("support_level", "")) for item in results]
    scores = [float(item.get("score", 0) or 0) for item in results]
    supported_count = support_levels.count("支持")
    partial_supported_count = support_levels.count("部分支持")
    return {
        "claim_count": total,
        "supported_count": supported_count,
        "partial_supported_count": partial_supported_count,
        "unsupported_count": support_levels.count("不支持"),
        "insufficient_count": support_levels.count("证据不足"),
        "evidence_support_rate": round((supported_count + partial_supported_count) / total, 4),
        "average_score": round(sum(scores) / total, 4),
    }


def build_profile_evidence_report(
    realtalk_path: str | Path,
    profile_path: str | Path | None,
    output_dir: str | Path,
    user_name: str = "default_user",
    top_k: int = 8,
    min_retrieval_score: float = 0.8,
    semantic_threshold: float = 0.2,
    use_embedding_retrieval: bool = True,
) -> Dict[str, Any]:
    
    evidence_pool = EvidencePoolBuilder().build_from_file(realtalk_path)

    resolved_profile_path = Path(profile_path) if profile_path else Path("user") / f"{user_name}_profile.json"
    profile = load_json(resolved_profile_path)
    claims = ProfileClaimExtractor().extract(profile)
    parent_claim_count = len({claim.get("parent_claim_id", claim["claim_id"]) for claim in claims})

    output = Path(output_dir)
    write_jsonl(output / "evidence_pool.jsonl", evidence_pool)
    save_json(output / "claims.json", claims)

    evaluator = ProfileEvidenceEvaluator(
        llm=LLMClient(),
        top_k=top_k,
        min_retrieval_score=min_retrieval_score,
        semantic_threshold=semantic_threshold,
        use_embedding_retrieval=use_embedding_retrieval
    )
    results = evaluator.evaluate(claims, evidence_pool)
    summary = summarize_results(results)
    report = {
        "realtalk_path": str(realtalk_path),
        "profile_path": str(resolved_profile_path),
        "user_name": user_name,
        "evidence_count": len(evidence_pool),
        "claim_count": len(claims),
        "parent_claim_count": parent_claim_count,
        "results": results,
    }
    save_json(output / "evaluation_results.json", report)
    save_json(output / "summary.json", summary)
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate profile claim evidence support on REALTALK data.")
    parser.add_argument("--realtalk", required=True, help="REALTALK json file or data directory.")
    parser.add_argument("--profile", default=None, help="Profile JSON path. Defaults to user/{user_name}_profile.json.")
    parser.add_argument("--user-name", default="default_user", help="User name used when --profile is omitted.")
    parser.add_argument("--output", default="data/profile_evidence_eval", help="Output directory.")
    parser.add_argument("--top-k", type=int, default=8, help="Top retrieved evidence per claim.")
    parser.add_argument("--min-score", type=float, default=0.8, help="Minimum rule retrieval score.")
    parser.add_argument("--semantic-threshold", type=float, default=0.2, help="Minimum embedding cosine similarity.")
    parser.add_argument("--no-embedding-retrieval", action="store_true", help="Disable embedding semantic retrieval.")
    args = parser.parse_args()

    build_profile_evidence_report(
        realtalk_path=args.realtalk,
        profile_path=args.profile,
        output_dir=args.output,
        user_name=args.user_name,
        top_k=args.top_k,
        min_retrieval_score=args.min_score,
        semantic_threshold=args.semantic_threshold,
        use_embedding_retrieval=not args.no_embedding_retrieval,
    )


if __name__ == "__main__":
    main()
