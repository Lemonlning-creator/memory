from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import select_realtalk_splits
from src.experiments.operation_checkpoint import OperationCheckpoint
from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_ours import (
    EXPECTED_FULL_TARGETS,
    EXPECTED_MODEL,
    RealTalkOursConfig,
    _generation_self_domain,
    _has_durable_user_domain_evidence,
    _normalize_alignment_for_context,
    _prepare_dataset,
    _structured_call,
    run_realtalk_ours,
)
from src.experiments.realtalk_ours_schemas import (
    SELF_DOMAIN_SCHEMA,
    normalize_alignment,
    normalize_self_domain,
)


def _self_domain() -> dict:
    return {
        "identity": {
            "self_descriptions": [],
            "stable_interests": ["likes music"],
            "relationships": [],
            "life_context": [],
        },
        "persona": {
            "personality_traits": ["casual"],
            "tone": ["brief"],
            "expression_patterns": ["uses informal wording"],
        },
        "behavior_policy_prior": {
            "interaction_principles": ["respond directly"],
            "emotional_response_style": "restrained",
            "guidance_style": "rarely gives advice",
            "initiative": "medium",
        },
        "hard_constraints": ["do not invent personal events"],
        "uncertainties": [],
    }


def _user_domain(evidence_id: str) -> dict:
    return {
        "core": [],
        "regulation": [],
        "cognition": [],
        "identity": [{
            "statement": "The partner discusses daily experiences.",
            "confidence": "low",
            "evidence_turn_ids": [evidence_id],
        }],
        "behavior": [],
        "update_summary": {
            "added": ["daily experiences"],
            "revised": [],
            "removed": [],
            "uncertainties": [],
        },
    }


def _alignment(evidence_id: str | None) -> dict:
    no_evidence = evidence_id is None
    return {
        "user_state": {
            "current": {
                "emotion": "" if no_evidence else "neutral",
                "emotional_intensity": "low",
                "intent": "" if no_evidence else "continue the conversation",
                "main_need": "",
                "interaction_expectation": "" if no_evidence else "a natural reply",
                "evidence_turn_ids": [evidence_id] if evidence_id else [],
                "uncertainty": "high",
            },
            "future": {
                "likely_reaction": "" if no_evidence else "may continue the topic",
                "response_risk": "" if no_evidence else "overly polished wording would be out of character",
                "desired_transition": "" if no_evidence else "a natural continuation",
                "uncertainty": "high",
            },
        },
        "alignment": {
            "lambda_t": 0.0 if no_evidence else 0.6,
            "orientation": "self-dominant" if no_evidence else "user-leaning",
            "lambda_basis": "no partner evidence" if no_evidence else "some partner context is visible",
            "self_constraint": "retain the target's casual style",
            "user_adaptation": "none" if no_evidence else "address the latest topic",
        },
        "behavior_policy": {
            "response_objective": "continue naturally",
            "perspective_taking": "acknowledge the partner's topic",
            "emotion_alignment": "do not exaggerate emotion",
            "personalization": "none" if no_evidence else "use only visible context",
            "self_domain_expression": "use casual concise wording",
            "directness": "medium",
            "guidance": "none",
            "question_policy": "optional",
            "tone": "casual",
            "avoid": ["generic assistant language"],
        },
    }


class FakeBackend:
    model = EXPECTED_MODEL
    base_url = "https://example.invalid/v1"
    enable_thinking = False

    def __init__(self, malformed_self_once: bool = False) -> None:
        self.calls = []
        self.malformed_self_once = malformed_self_once
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "network_attempts": 0,
            "network_retries": 0,
        }

    def available_models(self):
        return [EXPECTED_MODEL]

    def chat(
        self,
        system_prompt,
        user_prompt,
        *,
        temperature,
        max_tokens,
        top_p=0.9,
        response_schema=None,
    ):
        schema_name = response_schema["name"] if response_schema else None
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "schema": schema_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        })
        if schema_name == "realtalk_ours_self_domain":
            prior_self_calls = sum(
                call["schema"] == schema_name for call in self.calls
            )
            if self.malformed_self_once and prior_self_calls == 1:
                content = "not-json"
            else:
                content = json.dumps(_self_domain())
        elif schema_name == "realtalk_ours_user_domain":
            block = user_prompt.split("NEWLY OBSERVED PARTNER TURNS:", 1)[1]
            evidence = re.search(r"\[([^\]]+)\]", block).group(1)
            content = json.dumps(_user_domain(evidence))
        elif schema_name == "realtalk_ours_alignment":
            ids = re.findall(r"\[([^\]]+:turn_\d+)\]", user_prompt)
            content = json.dumps(_alignment(ids[-1] if ids else None))
        else:
            content = "READY" if max_tokens == 8 else "Emi: FAKE_GENERATED"
        self.token_usage["calls"] += 1
        self.token_usage["network_attempts"] += 1
        self.token_usage["prompt_tokens"] += 10
        self.token_usage["completion_tokens"] += 5
        return ChatResult(
            content=content,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_seconds=0.01,
            attempts=1,
        )


class FakeLabels:
    def annotate(self, text):
        return {
            "emotion": "joy" if "FAKE_GENERATED" in text else "sadness",
            "sentiment": "positive" if "FAKE_GENERATED" in text else "negative",
            "intimacy": 0.8 if "FAKE_GENERATED" in text else 0.3,
        }

    def metadata(self):
        return {"provider": "fake-pinned-labels"}


class RealTalkOursTests(unittest.TestCase):
    def test_public_table8_reconstruction_has_expected_519_targets(self):
        config = RealTalkOursConfig(compute_local_metrics=False)
        splits = select_realtalk_splits(config.dataset_dir)
        manifest, prepared = _prepare_dataset(config, splits)
        self.assertEqual(manifest["total_targets"], EXPECTED_FULL_TARGETS)
        self.assertEqual(len(prepared), 10)
        self.assertEqual(
            [(item["speaker"], item["partner"]) for item in prepared],
            [
                ("Emi", "elise"),
                ("Nicolas", "Vanessa"),
                ("Kevin", "elise"),
                ("Akib", "Muhhamed"),
                ("Muhhamed", "Akib"),
                ("Nebraas", "Vanessa"),
                ("Paola", "Kevin"),
                ("Vanessa", "Nicolas"),
                ("elise", "Emi"),
                ("Fahim Khan", "Akib"),
            ],
        )

    def test_strict_schemas_reject_extra_fields_and_lambda_mismatch(self):
        value = _self_domain()
        value["extra"] = True
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            normalize_self_domain(value)

        alignment = _alignment("session_1:turn_1")
        alignment["alignment"]["orientation"] = "self-dominant"
        with self.assertRaisesRegex(ValueError, "conflicts with lambda_t"):
            normalize_alignment(alignment)

    def test_no_partner_evidence_is_deterministically_cold_started(self):
        model_value = _alignment("session_1:turn_1")
        normalized = _normalize_alignment_for_context(
            normalize_alignment(model_value), set()
        )
        self.assertEqual(normalized["alignment"]["lambda_t"], 0.0)
        self.assertEqual(normalized["alignment"]["orientation"], "self-dominant")
        self.assertEqual(normalized["user_state"]["current"]["emotion"], "")
        self.assertEqual(
            normalized["behavior_policy"]["personalization"], "none"
        )

    def test_alignment_policy_cannot_turn_profile_facts_into_message_content(self):
        normalized = _normalize_alignment_for_context(
            normalize_alignment(_alignment("session_1:turn_1")),
            {"session_1:turn_1"},
        )
        expression = normalized["behavior_policy"]["self_domain_expression"]
        self.assertIn("tone, phrasing, initiative", expression)
        self.assertIn(
            "unsupported first-person factual details",
            normalized["behavior_policy"]["avoid"],
        )

    def test_generation_view_hides_identity_facts_but_preserves_style(self):
        view = _generation_self_domain(_self_domain())
        self.assertNotIn("identity", view)
        self.assertNotIn("likes music", json.dumps(view))
        self.assertEqual(view["persona"]["tone"], ["brief"])
        self.assertIn("hard_constraints", view)

    def test_generic_greetings_do_not_update_stable_user_domain(self):
        for text in ("Hi!", "Hey there, how are you?", "What's up?"):
            self.assertFalse(_has_durable_user_domain_evidence(text))
        self.assertTrue(
            _has_durable_user_domain_evidence(
                "I started a new nursing job downtown last week."
            )
        )

    def test_structured_call_repairs_format_only_and_caches_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = OperationCheckpoint(root / "checkpoint.json", "sig")
            backend = FakeBackend(malformed_self_once=True)
            result = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key="self_domain:emi",
                system_prompt="system",
                user_prompt="original evidence",
                schema=SELF_DOMAIN_SCHEMA,
                normalizer=normalize_self_domain,
                max_tokens=1800,
                max_attempts=3,
                raw_audit=root / "raw.jsonl",
            )
            self.assertEqual(result["data"], _self_domain())
            self.assertEqual(result["audit"]["logical_attempts"], 2)
            self.assertIn("FORMAT REPAIR REQUIRED", backend.calls[1]["user"])
            self.assertIn("Do not add new evidence", backend.calls[1]["user"])
            calls = len(backend.calls)
            resumed = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key="self_domain:emi",
                system_prompt="changed but cached",
                user_prompt="changed but cached",
                schema=SELF_DOMAIN_SCHEMA,
                normalizer=normalize_self_domain,
                max_tokens=1800,
                max_attempts=3,
                raw_audit=root / "raw.jsonl",
            )
            self.assertEqual(resumed, result)
            self.assertEqual(len(backend.calls), calls)

    def test_small_run_is_causal_resumable_and_never_marks_pipeline_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = RealTalkOursConfig(
                output_dir=str(output),
                speaker_filter=("Emi",),
                max_eval_points_per_speaker=2,
                compute_local_metrics=False,
            )
            backend = FakeBackend()
            summary = run_realtalk_ours(config, backend=backend)
            self.assertTrue(summary["generation_complete"])
            self.assertEqual(summary["records"], 2)
            self.assertTrue((output / "GENERATION_COMPLETE").exists())
            self.assertFalse((output / "PIPELINE_COMPLETE").exists())
            self.assertTrue((output / "GPT_EVALUATION_PENDING.json").exists())

            generation_calls = [
                call for call in backend.calls
                if call["schema"] is None and call["max_tokens"] == 300
            ]
            self.assertEqual(len(generation_calls), 2)
            self.assertNotIn("FAKE_GENERATED", generation_calls[1]["user"])
            predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [item["generated_message"] for item in predictions],
                ["FAKE_GENERATED", "FAKE_GENERATED"],
            )

            resumed_backend = FakeBackend()
            resumed = run_realtalk_ours(config, backend=resumed_backend)
            self.assertEqual(resumed["records"], 2)
            self.assertEqual(resumed_backend.calls, [])

    def test_local_five_metric_stage_writes_partial_table_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = RealTalkOursConfig(
                output_dir=str(output),
                speaker_filter=("Emi",),
                max_eval_points_per_speaker=2,
            )
            summary = run_realtalk_ours(
                config,
                backend=FakeBackend(),
                label_evaluator=FakeLabels(),
                bertscore_fn=lambda references, candidates: [0.75] * len(references),
            )
            self.assertEqual(summary["local_metrics"]["message_count"], 2)
            macro = summary["local_metrics"]["speaker_macro"]
            self.assertEqual(macro["bertscore_f1"]["mean"], 0.75)
            self.assertEqual(macro["sentiment_accuracy"]["mean"], 0.0)
            self.assertEqual(macro["intimacy_absolute_difference"]["mean"], 0.5)
            self.assertTrue((output / "LOCAL_METRICS_COMPLETE").exists())
            self.assertFalse((output / "PIPELINE_COMPLETE").exists())
            report = (output / "REPORT_PARTIAL.md").read_text(encoding="utf-8")
            self.assertIn("pending gpt-4o-mini", report)


if __name__ == "__main__":
    unittest.main()
