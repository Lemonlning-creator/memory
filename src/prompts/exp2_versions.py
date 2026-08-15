from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict


# Core branch (core/v18-scores-only): the version zoo from the prompt sweep
# (v1-v17, v19-v28) has been removed. The only retained configuration is V18:
#   - v18_reflective_grounding_scores_only  -> the result-audited winning
#     configuration and the single runtime default (scores_only state policy);
#   - v18_reflective_grounding_joint_gate   -> kept only because the controlled
#     ablation runner validates frozen source directories against the prompt
#     version recorded in their prediction rows.
DEFAULT_EXP2_PROMPT_VERSION = "v18_reflective_grounding_scores_only"
EVALUATION_PROMPT_VERSION = "realtalk_table2_eval_v1"

CURRENT_STATE_FIELDS = {
    "emotional_state": str,
    "emotional_intensity": str,
    "emotional_valence": str,
    "energy_level": str,
    "stress_level": str,
    "current_concerns": list,
    "social_openness": str,
    "mood_trajectory": str,
    "dominant_topics": list,
    "coping_mode": str,
}
PROJECTED_STATE_FIELDS = {
    "projected_trend": str,
    "projected_with_empathy": str,
    "risk_of_misalignment": str,
}


def validate_state_update(state_update: Any) -> tuple[bool, str]:
    """Validate the exact fixed state schema emitted by v2/v3 alignment."""
    if not isinstance(state_update, dict):
        return False, "state_update must be an object"
    expected_root = {"current_state", "projected_state"}
    if set(state_update) != expected_root:
        return False, (
            "state_update fields must be exactly "
            f"{sorted(expected_root)}; actual={sorted(state_update)}"
        )

    for section_name, schema in (
        ("current_state", CURRENT_STATE_FIELDS),
        ("projected_state", PROJECTED_STATE_FIELDS),
    ):
        section = state_update.get(section_name)
        if not isinstance(section, dict):
            return False, f"{section_name} must be an object"
        if set(section) != set(schema):
            return False, (
                f"{section_name} fields must be exactly {sorted(schema)}; "
                f"actual={sorted(section)}"
            )
        for field, expected_type in schema.items():
            value = section[field]
            if expected_type is str:
                if not isinstance(value, str) or not value.strip():
                    return False, f"{section_name}.{field} must be a non-empty string"
            elif not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                return False, f"{section_name}.{field} must be a list of non-empty strings"

    return True, ""


@dataclass(frozen=True)
class Exp2PromptBundle:
    version: str
    response_system: str
    response_user: str
    alignment_system: str
    alignment_user: str
    updates_user_state: bool
    description: str
    response_state_policy: str = "empathy_state"

    @property
    def fingerprint(self) -> str:
        payload = "\n---PROMPT---\n".join((
            self.response_system,
            self.response_user,
            self.alignment_system,
            self.alignment_user,
        ))
        # Preserve every existing version's historical fingerprint. New runtime
        # response-state policies are fingerprinted only when they opt out of
        # the legacy empathy-state payload.
        if self.response_state_policy != "empathy_state":
            payload += f"\n---RESPONSE-STATE-POLICY---\n{self.response_state_policy}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def manifest(self) -> Dict[str, str | bool]:
        return {
            "version": self.version,
            "sha256": self.fingerprint,
            "updates_user_state": self.updates_user_state,
            "response_state_policy": self.response_state_policy,
            "description": self.description,
        }


V5_ALIGNMENT_SYSTEM_PROMPT = """Maintain the target agent's working understanding of the user for the following interaction. Analyze rather than generate a reply.

First infer relationship_distance as unfamiliar, casual, familiar, or close from demonstrated interaction behavior. Repeated sessions and detailed stored information do not alone imply closeness; when uncertain, choose the more distant level.

Then infer the user's current state from the latest message and recent dialogue. Stable profile evidence is secondary, and preceding state is weak historical context. Do not manufacture emotion, distress, or an underlying need when the turn is ordinary conversation.

Calibrate emotional_reaction, interpretation, and exploration independently from 0 to 2. Start at 0. Increase a value only when the current turn warrants that act, the relationship distance permits it, and the target persona commonly performs it in comparable situations. More empathy is not inherently better. For unfamiliar or casual relationships, strong scores require explicit evidence. A content question is not emotional exploration.

Return only valid JSON with exactly this structure:
{
  "understanding": {
    "relationship_distance": "unfamiliar/casual/familiar/close",
    "current_emotion": "primary evidenced emotion or neutral/unknown",
    "emotional_intensity": "low/medium/high",
    "underlying_need": "evidenced conversational need or unknown",
    "profile_evidence": ["relevant activated profile evidence"],
    "persona_constraint": "frequency and relationship constraint relevant to the next interaction"
  },
  "prediction": {
    "projected_trend": "cautious likely next-turn direction",
    "risk_of_misalignment": "main risk of excessive or insufficient engagement"
  },
  "empathy_state": {
    "emotional_reaction": 0,
    "interpretation": 0,
    "exploration": 0,
    "activated_tone": "relationship- and persona-calibrated tone",
    "response_guidance": "one concise recommendation for the next interaction"
  },
  "state_update": {
    "current_state": {
      "emotional_state": "primary emotion currently evidenced by the user message",
      "emotional_intensity": "low/medium/high",
      "emotional_valence": "positive/neutral/negative",
      "energy_level": "low/medium/high/unknown",
      "stress_level": "low/medium/high/unknown",
      "current_concerns": ["concern active in this turn"],
      "social_openness": "withdrawn/neutral/engaged/unknown",
      "mood_trajectory": "improving/stable/declining/unknown",
      "dominant_topics": ["topic active in this turn"],
      "coping_mode": "currently evidenced coping mode or unknown"
    },
    "projected_state": {
      "projected_trend": "cautious likely next-turn direction",
      "projected_with_empathy": "cautious likely direction with an appropriately calibrated reply",
      "risk_of_misalignment": "main relationship, tone, or empathy risk"
    }
  }
}

Use empty lists and "unknown" when evidence is insufficient. Do not add, remove, or rename fields."""

V5_ALIGNMENT_USER_PROMPT = """RECENT DIALOGUE:
{recent_context}

CURRENT OBSERVED USER MESSAGE:
{user_message}

EXPLICIT USER PROFILE:
{user_profile}

TARGET AGENT'S FREQUENCY-CALIBRATED PERSONA:
{agent_persona}

PRECEDING USER STATE:
{current_state}

EPISTEMIC OMEGA:
{epistemic_omega}

Infer relationship distance, current understanding, restrained next-turn alignment, and the exact fixed-schema state update."""


# The V6-V10 response-prompt sweep shared this input layout and the V5
# alignment prompt so that only final-response policy changes between variants.
PROMPT_SWEEP_RESPONSE_USER_PROMPT = """LATEST USER MESSAGE:
{user_input}

RECENT REAL DIALOGUE AND RETRIEVED EVIDENCE:
{relevant_memory}

TARGET AGENT PERSONA:
{persona_config}

EXPLICIT USER PROFILE:
{static_profile}

PRECEDING COMPLETED USER STATE:
{current_state}

CURRENT PROFILE CONTEXT:
{current_context}

PREVIOUS-TURN EMPATHY STATE:
{previous_empathy_state}

Write only the target agent's single most plausible next message."""


V7_RESPONSE_SYSTEM_PROMPT = """Write the target conversation partner's next message by reproducing their observable surface style in a real-world dialogue. Do not improve, polish, or professionalize the way this person normally writes.

This variant tests RECENT-STYLE IMITATION at medium strength.

Style evidence is ordered as follows:
1. recent real messages from the target speaker in the supplied dialogue;
2. expression_layer.language_style and expression_layer.behavioral_mannerisms;
3. the remaining persona only for stable factual boundaries.

Match the recent target messages' typical length, sentence complexity, informality, directness, question frequency, enthusiasm, and depth of explanation. If their messages are plain, rough, repetitive, abbreviated, or lightly ungrammatical, do not turn them into polished prose. Do not introduce literary metaphors, thematic analysis, therapeutic phrasing, or sophisticated vocabulary that the target speaker has not been using.

Stay with the latest active topic or direct question. Reuse ordinary vocabulary from the latest user message and recent target messages when natural, without copying a previous message verbatim. Make the same number of conversational moves the target usually makes; default to one.

Background knowledge is not proof of personal experience. Do not claim that the speaker watched, read, visited, owned, remembered, or recently did something unless that exact kind of fact is supported. Do not expose stored user-profile facts merely to appear personalized.

Match emotional tone without turning friendliness or topic enthusiasm into empathy. Ask, reflect, validate, or explore only at the rate visible in recent target messages. Output only the final reply."""


V18_RESPONSE_SYSTEM_PROMPT = V7_RESPONSE_SYSTEM_PROMPT + """

Before writing, silently choose exactly one primary response-act pattern:

A. REFLECTIVE. Use one natural self-observation only when the active topic is the target speaker's own feeling, motive, realization, decision, change, or recurring pattern. A fact, preference, opinion, explanation, or phrase such as "I think" or "I feel" is not reflective without awareness of an internal motive or pattern.

B. GROUNDING. Ask at most one concise clarification, confirmation, or directly connected follow-up about a specific detail the user already raised. Do not use a question merely to keep the conversation active, and do not use generic reciprocal, unrelated, stacked, or counselor-style questions.

C. ORDINARY. When neither A nor B is clearly supported, use the direct answer, acknowledgement, opinion, reaction, or supported self-disclosure this speaker would normally give. The reply may end without a question.

Do not combine A and B unless a genuine ambiguity makes both indispensable. If evidence for an optional act is weak, choose C. Do not mention this internal choice. Output only the final reply."""


_BUNDLES = {
    "v18_reflective_grounding_joint_gate": Exp2PromptBundle(
        version="v18_reflective_grounding_joint_gate",
        response_system=V18_RESPONSE_SYSTEM_PROMPT,
        response_user=PROMPT_SWEEP_RESPONSE_USER_PROMPT,
        alignment_system=V5_ALIGNMENT_SYSTEM_PROMPT,
        alignment_user=V5_ALIGNMENT_USER_PROMPT,
        updates_user_state=True,
        description=(
            "V7 surface-style policy with one joint reflective/grounding/ordinary "
            "response-act gate; full fixed user profile, original fixed persona, "
            "V5 alignment, and all other generation inputs unchanged."
        ),
    ),
    "v18_reflective_grounding_scores_only": Exp2PromptBundle(
        version="v18_reflective_grounding_scores_only",
        response_system=V18_RESPONSE_SYSTEM_PROMPT,
        response_user=PROMPT_SWEEP_RESPONSE_USER_PROMPT,
        alignment_system=V5_ALIGNMENT_SYSTEM_PROMPT,
        alignment_user=V5_ALIGNMENT_USER_PROMPT,
        updates_user_state=True,
        description=(
            "The result-audited V18 response/alignment prompts with the response "
            "state restricted to emotional_reaction, interpretation, and "
            "exploration. This is the direct, single-pass form of the controlled "
            "scores_only condition."
        ),
        response_state_policy="scores_only",
    ),
}


def exp2_prompt_versions() -> tuple[str, ...]:
    return tuple(_BUNDLES)


def get_exp2_prompt_bundle(version: str) -> Exp2PromptBundle:
    try:
        return _BUNDLES[version]
    except KeyError as exc:
        raise ValueError(
            f"unknown Exp2 prompt version {version!r}; choose one of {exp2_prompt_versions()}"
        ) from exc
