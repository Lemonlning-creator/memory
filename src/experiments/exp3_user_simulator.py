"""Strict hidden-profile construction, user simulation, and Exp3-B metrics."""
from __future__ import annotations

from dataclasses import dataclass
import json
from statistics import mean, pstdev
from typing import Any, Dict, Mapping, Sequence

from ..llm_client import LLMClient
from ..utils import parse_json


HIDDEN_CLAIM_EXTRACTION_SYSTEM = """You extract stable user-profile claims from
held-out user utterances. Every claim must be directly supported by one or more
provided evidence IDs. Do not infer facts from assistant utterances or external
knowledge. Copy evidence IDs exactly. Return only valid JSON."""

HIDDEN_CLAIM_EXTRACTION_TEMPLATE = """USER: {user_name}
ALLOWED PROFILE PATHS (closed set):
{profile_paths}

HELD-OUT USER EVIDENCE:
{evidence}

Extract atomic candidate claims. A claim contains one stable fact only. Exclude
momentary feelings unless they express a recurring pattern. Every path MUST be
copied exactly from ALLOWED PROFILE PATHS. If no allowed path fits, omit the
claim; never create a new path.

Return exactly:
{{"claims":[{{"candidate_id":"E001","path":"layer.field","text":"claim",
"evidence_ids":["existing evidence id"],"stability":"stable"}}]}}
candidate_id values must be sequential E001, E002, ... . stability is either
stable or transient."""

HIDDEN_CLAIM_AUDIT_SYSTEM = """You audit semantic differences between an initial
user profile and a full user profile. Work at atomic-claim level, not field-name
level. A full claim is:
- known: semantically already present in the initial profile;
- new: absent from the initial profile;
- refinement: compatible, meaningful specificity beyond an initial claim;
- contradiction: incompatible with the initial profile;
Evidence support is a separate axis: copy matching held-out evidence claim IDs
when they exist, otherwise leave matched_evidence_claim_ids empty. Never change
the novelty relation merely because evidence is absent.
Hard consistency rules:
1. new requires an empty matched_initial_claim_ids list.
2. refinement requires at least one matched_initial_claim_id.
3. known requires at least one matched_initial_claim_id.
4. contradiction requires at least one matched_initial_claim_id.
Return one audit row for every full-profile claim, in input order. Copy all IDs
exactly and return only valid JSON."""

HIDDEN_CLAIM_AUDIT_TEMPLATE = """INITIAL PROFILE CLAIMS:
{initial_claims}

FULL PROFILE CLAIMS:
{full_claims}

HELD-OUT EVIDENCE CLAIMS:
{evidence_claims}

Return exactly:
{{"audit":[{{"full_claim_id":"P001","relation":"known|new|refinement|contradiction",
"matched_initial_claim_ids":[],"matched_evidence_claim_ids":[]}}]}}"""

DISCLOSURE_CONTROLLER_SYSTEM = """You are the private disclosure controller for
a realistic user simulator. Decide whether the user's next reply may reveal at
most two still-hidden claims. Relevance to the companion's latest message is
required. Respect sensitivity, trust, fatigue, and repeated questioning. A
generic prompt must not trigger unrelated disclosure. Return only valid JSON;
never draft the visible reply."""

DISCLOSURE_CONTROLLER_TEMPLATE = """AVAILABLE, NOT-YET-DISCLOSED TARGET CLAIMS:
{hidden_claims}

NUMBER OF CLAIMS ALREADY DISCLOSED: {disclosed_count}
Never repeat previously disclosed information.

STATE: trust={trust}, fatigue={fatigue}

RECENT CONVERSATION:
{conversation}

COMPANION'S LATEST MESSAGE:
{agent_message}

Return exactly:
{{"decision":"disclose|withhold|refuse|none","allowed_claim_ids":[],
"disclosure_depth":"none|partial|full","perceived_burden":0,
"rationale":"brief private reason","next_trust":0.0,"next_fatigue":0.0}}
perceived_burden is the integer 0, 1, or 2. next_trust and next_fatigue are
numbers in [0,1]."""

REPLY_RENDERER_SYSTEM = """You render one natural user reply. You only know the
known profile, style examples, recent conversation, and any claims explicitly
authorized for this turn. Never invent biography or expose claim IDs, the
simulation, dataset, or private annotations. Do not turn the reply into a list
of facts. Return only valid JSON."""

REPLY_RENDERER_TEMPLATE = """USER: {user_name}
KNOWN INITIAL PROFILE CLAIMS:
{known_claims}

CONTENT-FREE SURFACE STYLE GUIDANCE:
{style_examples}

RECENT CONVERSATION:
{conversation}

COMPANION'S LATEST MESSAGE:
{agent_message}

DISCLOSURE DECISION: {decision}
AUTHORIZED CLAIMS FOR THIS TURN (the only hidden facts you may reveal):
{allowed_claims}

Return exactly:
{{"user_reply":"visible reply only","evidenced_claim_ids":[]}}
evidenced_claim_ids must list only authorized claims directly expressed in the
visible reply."""

OPENER_TEMPLATE = """USER: {user_name}
KNOWN INITIAL PROFILE CLAIMS:
{known_claims}

CONTENT-FREE SURFACE STYLE GUIDANCE:
{style_examples}

Start a short, ordinary conversation. Do not disclose any hidden target claim
and do not introduce yourself by listing personal facts.

Return exactly: {{"user_reply":"visible opening message only",
"evidenced_claim_ids":[]}}"""

PROFILE_DISCOVERY_JUDGE_SYSTEM_PROMPT = """Evaluate semantic entailment between
atomic user-profile claims. A hidden claim is supported only when the candidate
profile expresses the same stable fact with compatible specificity. A final
claim is new only when it is not entailed by the initial profile. Do not use
keyword overlap or external knowledge. Copy IDs exactly and return valid JSON."""

PROFILE_DISCOVERY_JUDGE_TEMPLATE = """HIDDEN TARGET CLAIMS:
{hidden_claims}

INITIAL PROFILE CLAIMS:
{initial_claims}

FINAL PROFILE CLAIMS:
{final_claims}

Return exactly:
{{"hidden_supported_by_initial":[],"hidden_supported_by_final":[],
"final_claims_new_vs_initial":[],"new_final_claims_supported_by_hidden":[],
"notes":"brief"}}
Every array must contain ID strings only, for example ["H001", "H004"]. Never
put claim objects, paths, explanations, null, or booleans in an ID array."""

PROFILE_COVERAGE_JUDGE_TEMPLATE = """HIDDEN TARGET CLAIMS:
{hidden_claims}

CANDIDATE PROFILE CLAIMS:
{candidate_claims}

Return exactly: {{"supported_hidden_claim_ids":[],"notes":"brief"}}
supported_hidden_claim_ids must contain ID strings only, never claim objects."""

PROFILE_NOVELTY_JUDGE_TEMPLATE = """INITIAL PROFILE CLAIMS:
{initial_claims}

FINAL PROFILE CLAIMS:
{final_claims}

List final-profile claims that are not semantically entailed by the initial
profile. Return exactly: {{"new_final_claim_ids":[],"notes":"brief"}}
new_final_claim_ids may contain F-prefixed ID strings from FINAL PROFILE CLAIMS
only. Never return I-prefixed IDs or claim objects."""

PROFILE_SUPPORT_JUDGE_TEMPLATE = """HIDDEN TARGET CLAIMS:
{hidden_claims}

NEW FINAL PROFILE CLAIMS:
{new_final_claims}

List new final claims semantically entailed by the hidden targets. Return
exactly: {{"supported_final_claim_ids":[],"notes":"brief"}}
supported_final_claim_ids may contain F-prefixed ID strings from NEW FINAL
PROFILE CLAIMS only. Never return H-prefixed IDs or claim objects."""


@dataclass(frozen=True)
class ProfileClaim:
    claim_id: str
    path: str
    text: str

    def as_dict(self) -> Dict[str, str]:
        return {"id": self.claim_id, "path": self.path, "text": self.text}


@dataclass(frozen=True)
class HiddenClaim(ProfileClaim):
    novelty: str
    evidence_ids: tuple[str, ...]
    evidence_texts: tuple[str, ...]
    sensitivity: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.claim_id,
            "path": self.path,
            "text": self.text,
            "novelty": self.novelty,
            "evidence_ids": list(self.evidence_ids),
            "evidence_texts": list(self.evidence_texts),
            "sensitivity": self.sensitivity,
        }


def _require_exact_keys(
    payload: Mapping[str, Any], required: set[str], context: str
) -> None:
    actual = set(payload)
    if actual != required:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(required - actual)}, "
            f"unexpected={sorted(actual - required)}"
        )


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _require_string_list(
    value: Any, context: str, *, unique: bool = True
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{context} contains duplicate IDs")
    return value


def _static_profile(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    if "state_axis" not in profile:
        return profile
    state = profile["state_axis"]
    if not isinstance(state, Mapping) or "static_profile" not in state:
        raise ValueError("runtime profile must contain state_axis.static_profile")
    static = state["static_profile"]
    if not isinstance(static, Mapping):
        raise ValueError("state_axis.static_profile must be an object")
    return static


def _leaf_values(raw: Any, path: str) -> list[str]:
    value = raw
    if isinstance(raw, Mapping):
        if "value" not in raw:
            raise ValueError(f"profile leaf {path} is an object without a value field")
        value = raw["value"]
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"profile leaf {path} list must contain only strings")
        return [item.strip() for item in value if item.strip()]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    raise ValueError(f"unsupported profile leaf type at {path}: {type(value).__name__}")


def atomic_profile_claims(
    profile: Mapping[str, Any], prefix: str, include_summaries: bool = False
) -> list[ProfileClaim]:
    """Convert a profile into deterministic atomic claims; reject malformed leaves."""
    static = _static_profile(profile)
    claims: list[ProfileClaim] = []
    counter = 1
    for layer, fields in static.items():
        if not isinstance(layer, str) or not isinstance(fields, Mapping):
            raise ValueError("static profile must map layer names to field objects")
        for field, raw_value in fields.items():
            if not isinstance(field, str):
                raise ValueError("profile field names must be strings")
            if field == "summary" and not include_summaries:
                continue
            path = f"{layer}.{field}"
            for value in _leaf_values(raw_value, path):
                if value.lower() in {"unknown", "none", "n/a"}:
                    continue
                claims.append(ProfileClaim(f"{prefix}{counter:03d}", path, value))
                counter += 1
    return claims


def format_claims(claims: Sequence[ProfileClaim]) -> str:
    return json.dumps([claim.as_dict() for claim in claims], ensure_ascii=False, indent=2)


def format_renderable_claims(claims: Sequence[ProfileClaim]) -> str:
    """Expose only claim semantics to the renderer, never evidence annotations."""
    return json.dumps(
        [{"id": claim.claim_id, "path": claim.path, "text": claim.text} for claim in claims],
        ensure_ascii=False,
        indent=2,
    )


def format_conversation(turns: Sequence[Mapping[str, str]], max_turns: int = 12) -> str:
    recent = turns[-max_turns:]
    if not recent:
        return "(no previous turns)"
    lines = []
    for index, turn in enumerate(recent):
        _require_exact_keys(turn, {"speaker", "text"}, f"conversation turn {index}")
        lines.append(f"{_require_text(turn['speaker'], 'speaker')}: "
                     f"{_require_text(turn['text'], 'text')}")
    return "\n".join(lines)


def extract_heldout_evidence_claims(
    llm: LLMClient,
    user_name: str,
    evidence: Mapping[str, str],
    profile_paths: Sequence[str],
) -> list[Dict[str, Any]]:
    if not evidence:
        raise ValueError("held-out user evidence cannot be empty")
    if any(not isinstance(key, str) or not isinstance(value, str) or not value.strip()
           for key, value in evidence.items()):
        raise ValueError("held-out evidence must map string IDs to non-empty text")
    raw = llm.chat(
        HIDDEN_CLAIM_EXTRACTION_SYSTEM,
        HIDDEN_CLAIM_EXTRACTION_TEMPLATE.format(
            user_name=user_name,
            profile_paths=json.dumps(list(profile_paths), ensure_ascii=False),
            evidence=json.dumps(evidence, ensure_ascii=False, indent=2),
        ),
        temperature=0.3,
        max_tokens=8000,
    )
    payload = parse_json(raw)
    _require_exact_keys(payload, {"claims"}, "held-out claim extraction")
    if not isinstance(payload["claims"], list):
        raise ValueError("held-out claim extraction claims must be a list")
    rows: list[Dict[str, Any]] = []
    valid_evidence_ids = set(evidence)
    valid_paths = set(profile_paths)
    for index, row in enumerate(payload["claims"], 1):
        if not isinstance(row, Mapping):
            raise ValueError("held-out claim row must be an object")
        _require_exact_keys(
            row, {"candidate_id", "path", "text", "evidence_ids", "stability"},
            f"held-out claim {index}",
        )
        expected_id = f"E{index:03d}"
        if row["candidate_id"] != expected_id:
            raise ValueError(f"held-out claim IDs must be sequential; expected {expected_id}")
        evidence_ids = _require_string_list(row["evidence_ids"], f"{expected_id}.evidence_ids")
        if not evidence_ids:
            raise ValueError(f"{expected_id} must cite at least one held-out evidence ID")
        unknown = set(evidence_ids) - valid_evidence_ids
        if unknown:
            raise ValueError(f"{expected_id} cites unknown evidence IDs {sorted(unknown)}")
        stability = row["stability"]
        if stability not in {"stable", "transient"}:
            raise ValueError(f"{expected_id}.stability must be stable or transient")
        path = _require_text(row["path"], f"{expected_id}.path")
        if path not in valid_paths:
            raise ValueError(f"{expected_id} uses unknown profile path {path}")
        rows.append({
            "candidate_id": expected_id,
            "path": path,
            "text": _require_text(row["text"], f"{expected_id}.text"),
            "evidence_ids": evidence_ids,
            "stability": stability,
        })
    return rows


def _build_hidden_claim_manifest_once(
    llm: LLMClient,
    user_name: str,
    initial_profile: Mapping[str, Any],
    full_profile: Mapping[str, Any],
    evidence: Mapping[str, str],
) -> Dict[str, Any]:
    """Construct hidden targets as the evidence-backed semantic difference P* minus P0."""
    initial = atomic_profile_claims(initial_profile, "I")
    full = atomic_profile_claims(full_profile, "P")
    if not full:
        raise ValueError("full profile contains no atomic claims")
    paths = sorted({claim.path for claim in initial + full})
    evidence_claims = extract_heldout_evidence_claims(llm, user_name, evidence, paths)
    audit_rows: list[Any] = []
    chunk_size = 8
    for chunk_start in range(0, len(full), chunk_size):
        chunk = full[chunk_start:chunk_start + chunk_size]
        raw = llm.chat(
            HIDDEN_CLAIM_AUDIT_SYSTEM,
            HIDDEN_CLAIM_AUDIT_TEMPLATE.format(
                initial_claims=format_claims(initial),
                full_claims=format_claims(chunk),
                evidence_claims=json.dumps(
                    evidence_claims, ensure_ascii=False, indent=2
                ),
            ),
            temperature=0.3,
            max_tokens=8000,
        )
        chunk_payload = parse_json(raw)
        _require_exact_keys(chunk_payload, {"audit"}, "hidden claim audit chunk")
        if (
            not isinstance(chunk_payload["audit"], list)
            or len(chunk_payload["audit"]) != len(chunk)
        ):
            raise ValueError(
                "hidden claim audit chunk must contain exactly one row per full claim"
            )
        actual_ids = [
            row["full_claim_id"] if isinstance(row, Mapping) and "full_claim_id" in row else None
            for row in chunk_payload["audit"]
        ]
        expected_ids = [claim.claim_id for claim in chunk]
        if actual_ids != expected_ids:
            raise ValueError(
                f"hidden claim audit chunk ID/order mismatch; expected={expected_ids}, "
                f"actual={actual_ids}"
            )
        audit_rows.extend(chunk_payload["audit"])
    payload = {"audit": audit_rows}
    initial_ids = {claim.claim_id for claim in initial}
    evidence_by_id = {row["candidate_id"]: row for row in evidence_claims}
    allowed_relations = {"known", "new", "refinement", "contradiction"}
    audit: list[Dict[str, Any]] = []
    hidden: list[HiddenClaim] = []
    for index, (claim, row) in enumerate(zip(full, payload["audit"]), 1):
        if not isinstance(row, Mapping):
            raise ValueError("hidden claim audit row must be an object")
        _require_exact_keys(
            row,
            {"full_claim_id", "relation", "matched_initial_claim_ids",
             "matched_evidence_claim_ids"},
            f"hidden claim audit row {index}",
        )
        if row["full_claim_id"] != claim.claim_id:
            raise ValueError("hidden claim audit rows must preserve full-claim order and IDs")
        relation = row["relation"]
        if relation not in allowed_relations:
            raise ValueError(f"invalid relation for {claim.claim_id}: {relation}")
        matched_initial = _require_string_list(
            row["matched_initial_claim_ids"], f"{claim.claim_id}.matched_initial_claim_ids"
        )
        matched_evidence = _require_string_list(
            row["matched_evidence_claim_ids"], f"{claim.claim_id}.matched_evidence_claim_ids"
        )
        if set(matched_initial) - initial_ids:
            raise ValueError(f"{claim.claim_id} cites unknown initial claim IDs")
        if set(matched_evidence) - set(evidence_by_id):
            raise ValueError(f"{claim.claim_id} cites unknown evidence claim IDs")
        if relation == "known" and not matched_initial:
            raise ValueError(f"{claim.claim_id} is known but matches no initial claim")
        if relation == "new" and matched_initial:
            raise ValueError(f"{claim.claim_id} is new but matches an initial claim")
        if relation in {"refinement", "contradiction"} and not matched_initial:
            raise ValueError(f"{claim.claim_id} is {relation} but matches no initial claim")
        evidence_ids: list[str] = []
        for evidence_claim_id in matched_evidence:
            candidate = evidence_by_id[evidence_claim_id]
            if candidate["stability"] != "stable":
                continue
            for evidence_id in candidate["evidence_ids"]:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        layer = claim.path.split(".", 1)[0]
        sensitivity_by_layer = {
            "core": "high",
            "identity": "high",
            "regulation": "medium",
            "cognition": "medium",
            "behavior": "low",
        }
        if layer not in sensitivity_by_layer:
            raise ValueError(f"cannot assign sensitivity for unknown profile layer {layer}")
        sensitivity = sensitivity_by_layer[layer]
        audit_row = {
            "full_claim": claim.as_dict(),
            "relation": relation,
            "matched_initial_claim_ids": matched_initial,
            "matched_evidence_claim_ids": matched_evidence,
            "sensitivity": sensitivity,
            "evidence_supported": bool(evidence_ids),
            "eligible_as_hidden_target": relation in {"new", "refinement"} and bool(evidence_ids),
        }
        audit.append(audit_row)
        if audit_row["eligible_as_hidden_target"]:
            hidden.append(HiddenClaim(
                claim_id=f"H{len(hidden) + 1:03d}",
                path=claim.path,
                text=claim.text,
                novelty=relation,
                evidence_ids=tuple(evidence_ids),
                evidence_texts=tuple(evidence[evidence_id] for evidence_id in evidence_ids),
                sensitivity=sensitivity,
            ))
    if not hidden:
        raise ValueError("semantic audit produced no evidence-backed hidden target claims")
    return {
        "initial_claims": [claim.as_dict() for claim in initial],
        "full_claims": [claim.as_dict() for claim in full],
        "heldout_evidence_claims": evidence_claims,
        "sensitivity_policy": {
            "core": "high", "identity": "high", "regulation": "medium",
            "cognition": "medium", "behavior": "low",
        },
        "audit": audit,
        "hidden_claims": [claim.as_dict() for claim in hidden],
    }


def build_hidden_claim_manifest(
    llm: LLMClient,
    user_name: str,
    initial_profile: Mapping[str, Any],
    full_profile: Mapping[str, Any],
    evidence: Mapping[str, str],
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Retry generation while preserving the exact hidden-claim validation rules."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    last_error: ValueError | json.JSONDecodeError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _build_hidden_claim_manifest_once(
                llm, user_name, initial_profile, full_profile, evidence
            )
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(
                f"[Exp3 hidden-claim schema retry] attempt={attempt}/{max_attempts} "
                f"error={exc}"
            )
    assert last_error is not None
    raise RuntimeError(
        f"hidden-claim construction failed strict validation after {max_attempts} attempts"
    ) from last_error


def hidden_claims_from_manifest(manifest: Mapping[str, Any]) -> list[HiddenClaim]:
    if "hidden_claims" not in manifest or not isinstance(manifest["hidden_claims"], list):
        raise ValueError("hidden claim manifest must contain hidden_claims list")
    claims: list[HiddenClaim] = []
    for index, row in enumerate(manifest["hidden_claims"], 1):
        if not isinstance(row, Mapping):
            raise ValueError("hidden claim must be an object")
        _require_exact_keys(
            row, {"id", "path", "text", "novelty", "evidence_ids", "evidence_texts", "sensitivity"},
            f"hidden claim {index}",
        )
        expected = f"H{index:03d}"
        if row["id"] != expected:
            raise ValueError(f"hidden claim IDs must be sequential; expected {expected}")
        novelty = row["novelty"]
        sensitivity = row["sensitivity"]
        if novelty not in {"new", "refinement"}:
            raise ValueError(f"{expected}.novelty must be new or refinement")
        if sensitivity not in {"low", "medium", "high"}:
            raise ValueError(f"invalid {expected}.sensitivity")
        evidence_ids = _require_string_list(row["evidence_ids"], f"{expected}.evidence_ids")
        evidence_texts = _require_string_list(
            row["evidence_texts"], f"{expected}.evidence_texts", unique=False
        )
        if not evidence_ids or len(evidence_ids) != len(evidence_texts):
            raise ValueError(f"{expected} must contain aligned non-empty evidence IDs and texts")
        claims.append(HiddenClaim(
            expected,
            _require_text(row["path"], f"{expected}.path"),
            _require_text(row["text"], f"{expected}.text"),
            novelty,
            tuple(evidence_ids),
            tuple(evidence_texts),
            sensitivity,
        ))
    if not claims:
        raise ValueError("hidden claim manifest contains no targets")
    return claims


def _validate_renderer_payload(payload: Mapping[str, Any], allowed: set[str]) -> Dict[str, Any]:
    _require_exact_keys(payload, {"user_reply", "evidenced_claim_ids"}, "reply renderer")
    reply = _require_text(payload["user_reply"], "reply renderer user_reply")
    evidenced = _require_string_list(payload["evidenced_claim_ids"], "evidenced_claim_ids")
    unknown = set(evidenced) - allowed
    if unknown:
        raise ValueError(f"reply renderer used unauthorized claim IDs: {sorted(unknown)}")
    return {"user_reply": reply, "evidenced_claim_ids": evidenced}


def validate_simulator_payload(payload: Mapping[str, Any], allowed_claim_ids: set[str]) -> Dict[str, Any]:
    """Validate the public simulator turn schema without defaults or coercion."""
    required = {
        "user_reply", "revealed_claim_ids", "disclosure_strength",
        "withheld_or_refused", "perceived_burden", "burden_reason",
        "disclosure_decision", "disclosure_depth", "trust", "fatigue",
    }
    _require_exact_keys(payload, required, "simulator payload")
    reply = _require_text(payload["user_reply"], "simulator user_reply")
    revealed = _require_string_list(payload["revealed_claim_ids"], "revealed_claim_ids")
    if set(revealed) - allowed_claim_ids:
        raise ValueError(f"simulator fabricated hidden claim IDs: {sorted(set(revealed) - allowed_claim_ids)}")
    strengths = payload["disclosure_strength"]
    if not isinstance(strengths, Mapping) or set(strengths) != set(revealed):
        raise ValueError("disclosure_strength keys must exactly equal revealed_claim_ids")
    normalized_strengths: Dict[str, float] = {}
    for claim_id, value in strengths.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid disclosure strength for {claim_id}")
        normalized_strengths[claim_id] = float(value)
    if not isinstance(payload["withheld_or_refused"], bool):
        raise ValueError("withheld_or_refused must be boolean")
    burden = payload["perceived_burden"]
    if not isinstance(burden, int) or isinstance(burden, bool) or burden not in {0, 1, 2}:
        raise ValueError("perceived_burden must be integer 0, 1, or 2")
    decision = payload["disclosure_decision"]
    depth = payload["disclosure_depth"]
    if decision not in {"disclose", "withhold", "refuse", "none", "opening"}:
        raise ValueError("invalid disclosure_decision")
    if depth not in {"none", "partial", "full"}:
        raise ValueError("invalid disclosure_depth")
    expected_withheld = decision in {"withhold", "refuse"}
    if payload["withheld_or_refused"] is not expected_withheld:
        raise ValueError("withheld_or_refused is inconsistent with disclosure_decision")
    if decision == "disclose" and (not revealed or depth == "none"):
        raise ValueError("disclose payload requires evidenced claims and non-none depth")
    if decision != "disclose" and (revealed or depth != "none"):
        raise ValueError("non-disclose payload cannot contain revealed claims or disclosure depth")
    trust = payload["trust"]
    fatigue = payload["fatigue"]
    for name, value in (("trust", trust), ("fatigue", fatigue)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a number in [0,1]")
    return {
        "user_reply": reply,
        "revealed_claim_ids": revealed,
        "disclosure_strength": normalized_strengths,
        "withheld_or_refused": payload["withheld_or_refused"],
        "perceived_burden": burden,
        "burden_reason": _require_text(payload["burden_reason"], "burden_reason"),
        "disclosure_decision": decision,
        "disclosure_depth": depth,
        "trust": float(trust),
        "fatigue": float(fatigue),
    }


class HiddenProfileUserSimulator:
    """Two-stage simulator: private disclosure controller, then limited renderer."""

    def __init__(
        self,
        llm: LLMClient,
        user_name: str,
        initial_profile: Mapping[str, Any],
        hidden_claims: Sequence[HiddenClaim],
        style_examples: Sequence[str],
        seed_index: int,
    ) -> None:
        if not hidden_claims:
            raise ValueError("simulator requires at least one hidden target claim")
        if any(not isinstance(value, str) or not value.strip() for value in style_examples):
            raise ValueError("style examples must be non-empty strings")
        self.llm = llm
        self.user_name = _require_text(user_name, "user_name")
        self.known_claims = atomic_profile_claims(initial_profile, "K")
        self.hidden_claims = list(hidden_claims)
        self.claim_by_id = {claim.claim_id: claim for claim in hidden_claims}
        if len(self.claim_by_id) != len(hidden_claims):
            raise ValueError("hidden target claim IDs must be unique")
        self.style_examples = list(style_examples)
        self.seed_index = seed_index
        self.disclosed_ids: set[str] = set()
        self.trust = 0.35
        self.fatigue = 0.0

    def _renderer_context(self) -> Dict[str, str]:
        return {
            "user_name": self.user_name,
            "known_claims": json.dumps(
                [{"path": claim.path, "text": claim.text} for claim in self.known_claims],
                ensure_ascii=False,
                indent=2,
            ),
            "style_examples": json.dumps(self.style_examples[:12], ensure_ascii=False, indent=2),
        }

    def _render_strict(
        self,
        prompt: str,
        allowed_claim_ids: set[str],
        max_attempts: int = 5,
    ) -> Dict[str, Any]:
        last_error: ValueError | json.JSONDecodeError | None = None
        correction = ""
        for attempt in range(1, max_attempts + 1):
            raw = self.llm.chat(
                REPLY_RENDERER_SYSTEM,
                prompt + correction,
                temperature=0.3,
                max_tokens=400,
            )
            try:
                rendered = _validate_renderer_payload(
                    parse_json(raw), allowed_claim_ids
                )
                if allowed_claim_ids and not rendered["evidenced_claim_ids"]:
                    raise ValueError(
                        "renderer must evidence at least one authorized claim "
                        "when disclosure was approved"
                    )
                return rendered
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                correction = (
                    "\n\nYOUR PREVIOUS OUTPUT FAILED STRICT VALIDATION: "
                    f"{exc}. Regenerate the complete JSON object. "
                    "Do not repair or explain the previous output. Return exactly "
                    '{"user_reply":"visible reply only","evidenced_claim_ids":[]} '
                    "with the appropriate authorized hidden IDs in the list."
                )
                print(
                    f"[Exp3 reply-renderer schema retry] attempt={attempt}/{max_attempts} "
                    f"error={exc}"
                )
        assert last_error is not None
        raise RuntimeError(
            f"reply renderer failed strict validation after {max_attempts} attempts"
        ) from last_error

    def opening_turn(self) -> Dict[str, Any]:
        rendered = self._render_strict(
            OPENER_TEMPLATE.format(**self._renderer_context()),
            set(),
        )
        return validate_simulator_payload({
            "user_reply": rendered["user_reply"],
            "revealed_claim_ids": [],
            "disclosure_strength": {},
            "withheld_or_refused": False,
            "perceived_burden": 0,
            "burden_reason": "ordinary opening without hidden-profile disclosure",
            "disclosure_decision": "opening",
            "disclosure_depth": "none",
            "trust": self.trust,
            "fatigue": self.fatigue,
        }, set(self.claim_by_id))

    def _validate_controller(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        required = {
            "decision", "allowed_claim_ids", "disclosure_depth", "perceived_burden",
            "rationale", "next_trust", "next_fatigue",
        }
        _require_exact_keys(payload, required, "disclosure controller")
        decision = payload["decision"]
        depth = payload["disclosure_depth"]
        if decision not in {"disclose", "withhold", "refuse", "none"}:
            raise ValueError("invalid disclosure controller decision")
        if depth not in {"none", "partial", "full"}:
            raise ValueError("invalid disclosure depth")
        ids = _require_string_list(payload["allowed_claim_ids"], "allowed_claim_ids")
        if len(ids) > 2:
            raise ValueError("disclosure controller authorized more than two claims")
        unavailable = set(ids) - (set(self.claim_by_id) - self.disclosed_ids)
        if unavailable:
            raise ValueError(f"controller selected unknown/already disclosed claims: {sorted(unavailable)}")
        if decision == "disclose" and (not ids or depth == "none"):
            raise ValueError("disclose decision requires claims and non-none depth")
        if decision != "disclose" and (ids or depth != "none"):
            raise ValueError("non-disclose decision cannot authorize claims or disclosure depth")
        burden = payload["perceived_burden"]
        if not isinstance(burden, int) or isinstance(burden, bool) or burden not in {0, 1, 2}:
            raise ValueError("controller perceived_burden must be integer 0, 1, or 2")
        next_trust = payload["next_trust"]
        next_fatigue = payload["next_fatigue"]
        for name, value in (("next_trust", next_trust), ("next_fatigue", next_fatigue)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be a number in [0,1]")
        return {
            "decision": decision,
            "allowed_claim_ids": ids,
            "disclosure_depth": depth,
            "perceived_burden": burden,
            "rationale": _require_text(payload["rationale"], "controller rationale"),
            "next_trust": float(next_trust),
            "next_fatigue": float(next_fatigue),
        }

    def respond(self, conversation: Sequence[Mapping[str, str]], agent_message: str) -> Dict[str, Any]:
        agent_message = _require_text(agent_message, "agent_message")
        controller_prompt = DISCLOSURE_CONTROLLER_TEMPLATE.format(
            hidden_claims=format_renderable_claims([
                claim for claim in self.hidden_claims
                if claim.claim_id not in self.disclosed_ids
            ]),
            disclosed_count=len(self.disclosed_ids),
            trust=self.trust,
            fatigue=self.fatigue,
            conversation=format_conversation(conversation),
            agent_message=agent_message,
        )
        controller: Dict[str, Any] | None = None
        controller_error: ValueError | json.JSONDecodeError | None = None
        correction = ""
        for attempt in range(1, 4):
            raw_controller = self.llm.chat(
                DISCLOSURE_CONTROLLER_SYSTEM,
                controller_prompt + correction,
                temperature=0.0,
                max_tokens=500,
            )
            try:
                controller = self._validate_controller(parse_json(raw_controller))
                break
            except (ValueError, json.JSONDecodeError) as exc:
                controller_error = exc
                correction = (
                    "\n\nYOUR PREVIOUS OUTPUT FAILED STRICT VALIDATION: "
                    f"{exc}. Regenerate the complete JSON object."
                )
                print(
                    f"[Exp3 disclosure-controller schema retry] attempt={attempt}/3 "
                    f"error={exc}"
                )
        if controller is None:
            assert controller_error is not None
            raise RuntimeError(
                "disclosure controller failed strict validation after 3 attempts"
            ) from controller_error
        allowed = set(controller["allowed_claim_ids"])
        rendered = self._render_strict(
            REPLY_RENDERER_TEMPLATE.format(
                **self._renderer_context(),
                conversation=format_conversation(conversation),
                agent_message=agent_message,
                decision=controller["decision"],
                allowed_claims=format_renderable_claims(
                    [self.claim_by_id[value] for value in controller["allowed_claim_ids"]]
                ),
            ),
            allowed,
        )
        revealed = rendered["evidenced_claim_ids"]
        self.disclosed_ids.update(revealed)
        self.trust = controller["next_trust"]
        self.fatigue = controller["next_fatigue"]
        strength_value = 1.0 if controller["disclosure_depth"] == "full" else 0.5
        return validate_simulator_payload({
            "user_reply": rendered["user_reply"],
            "revealed_claim_ids": revealed,
            "disclosure_strength": {claim_id: strength_value for claim_id in revealed},
            "withheld_or_refused": controller["decision"] in {"withhold", "refuse"},
            "perceived_burden": controller["perceived_burden"],
            "burden_reason": controller["rationale"],
            "disclosure_decision": controller["decision"],
            "disclosure_depth": controller["disclosure_depth"],
            "trust": self.trust,
            "fatigue": self.fatigue,
        }, set(self.claim_by_id))


def _validated_id_set(payload: Mapping[str, Any], field: str, allowed: set[str]) -> set[str]:
    if field not in payload:
        raise ValueError(f"profile discovery judge omitted required field {field}")
    values = _require_string_list(payload[field], f"profile discovery judge {field}")
    result = set(values)
    unknown = result - allowed
    if unknown:
        raise ValueError(f"profile discovery judge invented IDs in {field}: {sorted(unknown)}")
    return result


def f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def evaluate_hidden_coverage(
    judge: LLMClient,
    hidden_claims: Sequence[HiddenClaim],
    candidate_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    if not hidden_claims:
        raise ValueError("hidden claim set cannot be empty")
    candidate = atomic_profile_claims(candidate_profile, "C")
    prompt = PROFILE_COVERAGE_JUDGE_TEMPLATE.format(
        hidden_claims=format_claims(hidden_claims),
        candidate_claims=format_claims(candidate),
    )
    last_error: ValueError | json.JSONDecodeError | None = None
    correction = ""
    for attempt in range(1, 4):
        raw = judge.chat(
            PROFILE_DISCOVERY_JUDGE_SYSTEM_PROMPT,
            prompt + correction,
            temperature=0.0,
            max_tokens=1200,
        )
        try:
            payload = parse_json(raw)
            _require_exact_keys(
                payload, {"supported_hidden_claim_ids", "notes"},
                "profile coverage judge",
            )
            if not isinstance(payload["notes"], str):
                raise ValueError("profile coverage judge notes must be a string")
            supported = _validated_id_set(
                payload, "supported_hidden_claim_ids",
                {claim.claim_id for claim in hidden_claims},
            )
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            correction = (
                "\n\nPREVIOUS OUTPUT FAILED STRICT VALIDATION: "
                f"{exc}. Regenerate the complete JSON object; arrays contain ID "
                "strings only."
            )
            print(
                f"[Exp3 coverage-judge schema retry] attempt={attempt}/3 error={exc}"
            )
    else:
        assert last_error is not None
        raise RuntimeError(
            "profile coverage judge failed strict validation after 3 attempts"
        ) from last_error
    return {"supported_hidden_claim_ids": sorted(supported), "notes": payload["notes"]}


def evaluate_profile_discovery(
    judge: LLMClient,
    hidden_claims: Sequence[HiddenClaim],
    initial_profile: Mapping[str, Any],
    final_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    if not hidden_claims:
        raise ValueError("hidden claim set cannot be empty")
    initial = atomic_profile_claims(initial_profile, "I")
    final = atomic_profile_claims(final_profile, "F")
    initial_coverage_result = evaluate_hidden_coverage(
        judge, hidden_claims, initial_profile
    )
    final_coverage_result = evaluate_hidden_coverage(
        judge, hidden_claims, final_profile
    )
    hidden_initial = set(initial_coverage_result["supported_hidden_claim_ids"])
    hidden_final = set(final_coverage_result["supported_hidden_claim_ids"])
    final_ids = {claim.claim_id for claim in final}

    novelty_prompt = PROFILE_NOVELTY_JUDGE_TEMPLATE.format(
        initial_claims=format_claims(initial),
        final_claims=format_claims(final),
    )
    last_error: ValueError | json.JSONDecodeError | None = None
    correction = ""
    for attempt in range(1, 4):
        raw = judge.chat(
            PROFILE_DISCOVERY_JUDGE_SYSTEM_PROMPT,
            novelty_prompt + correction,
            temperature=0.0,
            max_tokens=1200,
        )
        try:
            payload = parse_json(raw)
            _require_exact_keys(
                payload, {"new_final_claim_ids", "notes"},
                "profile novelty judge",
            )
            if not isinstance(payload["notes"], str):
                raise ValueError("profile novelty judge notes must be a string")
            new_final = _validated_id_set(
                payload, "new_final_claim_ids", final_ids
            )
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            correction = (
                "\n\nPREVIOUS OUTPUT FAILED STRICT VALIDATION: "
                f"{exc}. Regenerate the complete JSON object using F-prefixed "
                "ID strings only."
            )
            print(
                f"[Exp3 novelty-judge schema retry] attempt={attempt}/3 error={exc}"
            )
    else:
        assert last_error is not None
        raise RuntimeError(
            "profile novelty judge failed strict validation after 3 attempts"
        ) from last_error

    new_final_claims = [claim for claim in final if claim.claim_id in new_final]
    if new_final_claims:
        support_prompt = PROFILE_SUPPORT_JUDGE_TEMPLATE.format(
            hidden_claims=format_claims(hidden_claims),
            new_final_claims=format_claims(new_final_claims),
        )
        last_error = None
        correction = ""
        for attempt in range(1, 4):
            raw = judge.chat(
                PROFILE_DISCOVERY_JUDGE_SYSTEM_PROMPT,
                support_prompt + correction,
                temperature=0.0,
                max_tokens=1200,
            )
            try:
                support_payload = parse_json(raw)
                _require_exact_keys(
                    support_payload, {"supported_final_claim_ids", "notes"},
                    "profile support judge",
                )
                if not isinstance(support_payload["notes"], str):
                    raise ValueError("profile support judge notes must be a string")
                supported_new = _validated_id_set(
                    support_payload, "supported_final_claim_ids", new_final
                )
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                correction = (
                    "\n\nPREVIOUS OUTPUT FAILED STRICT VALIDATION: "
                    f"{exc}. Regenerate using only the listed F-prefixed IDs."
                )
                print(
                    f"[Exp3 support-judge schema retry] attempt={attempt}/3 error={exc}"
                )
        else:
            assert last_error is not None
            raise RuntimeError(
                "profile support judge failed strict validation after 3 attempts"
            ) from last_error
        support_notes = support_payload["notes"]
    else:
        supported_new = set()
        support_notes = "No new final-profile claims were identified."
    target_count = len(hidden_claims)
    initial_coverage = len(hidden_initial) / target_count
    final_coverage = len(hidden_final) / target_count
    novel_precision = len(supported_new) / len(new_final) if new_final else None
    unsupported_rate = len(new_final - supported_new) / len(new_final) if new_final else None
    return {
        "hidden_claim_count": target_count,
        "initial_claim_count": len(initial),
        "final_claim_count": len(final),
        "initial_hidden_coverage": round(initial_coverage, 6),
        "final_hidden_coverage": round(final_coverage, 6),
        "hidden_coverage_gain": round(final_coverage - initial_coverage, 6),
        "novel_final_claim_count": len(new_final),
        "correct_novel_claim_count": len(supported_new),
        "novel_claim_precision": None if novel_precision is None else round(novel_precision, 6),
        "unsupported_novel_claim_rate": None if unsupported_rate is None else round(unsupported_rate, 6),
        "newly_learned_hidden_claim_ids": sorted(hidden_final - hidden_initial),
        "unsupported_novel_final_claim_ids": sorted(new_final - supported_new),
        "judge_annotations": {
            "hidden_supported_by_initial": sorted(hidden_initial),
            "hidden_supported_by_final": sorted(hidden_final),
            "final_claims_new_vs_initial": sorted(new_final),
            "new_final_claims_supported_by_hidden": sorted(supported_new),
            "notes": {
                "initial_coverage": initial_coverage_result["notes"],
                "final_coverage": final_coverage_result["notes"],
                "novelty": payload["notes"],
                "support": support_notes,
            },
        },
    }


def aggregate_discovery_results(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty exploration results")
    metrics = (
        "initial_hidden_coverage", "final_hidden_coverage", "hidden_coverage_gain",
        "elicitation_rate", "end_to_end_discovery_rate", "discovery_efficiency",
        "coverage_auc", "mean_user_burden", "refusal_rate", "exploration_question_rate",
    )
    aggregated: Dict[str, Any] = {
        "run_count": len(rows),
        "case_count": len({str(row["case_id"]) for row in rows}),
    }
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        aggregated[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    optional_metrics = ("uptake_rate", "novel_claim_precision", "unsupported_novel_claim_rate")
    for metric in optional_metrics:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        aggregated[metric] = None if not values else {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "defined_run_count": len(values),
        }
    return aggregated


__all__ = [
    "HiddenClaim", "HiddenProfileUserSimulator", "ProfileClaim",
    "aggregate_discovery_results", "atomic_profile_claims",
    "build_hidden_claim_manifest", "evaluate_hidden_coverage",
    "evaluate_profile_discovery", "f1_score", "format_claims",
    "hidden_claims_from_manifest", "validate_simulator_payload",
]
