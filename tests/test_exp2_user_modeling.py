import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.user_modeling.runner import (
    FINE_TUNED,
    OURS,
    ZERO_SHOT,
    Exp2UserModelingConfig,
    _normalize_explicit_profile,
    _normalize_target_profile,
    run_user_modeling_evaluation,
)
from src.experiments.user_modeling.schemas import (
    normalize_current_state,
    normalize_empathy,
    normalize_grounding,
    normalize_reflectiveness,
)
from src.experiments.exp2_generation import bertscore_runtime_metadata


class FakeLLM:
    model = "fake-user-modeling"
    max_retries = 6

    def __init__(self):
        self.calls = []
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }

    def chat(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        self.token_usage["prompt_tokens"] += 10
        self.token_usage["completion_tokens"] += 2
        self.token_usage["calls"] += 1
        schema = (kwargs.get("response_schema") or {}).get("name")
        if schema == "exp2_current_user_understanding":
            return json.dumps({
                "emotion": "joy",
                "sentiment": "positive",
                "topic": "work",
            })
        if schema == "exp2_reference_topic":
            return json.dumps({"topic": "work"})
        if schema == "exp2_realtalk_reflectiveness":
            return json.dumps({"reflective": True})
        if schema == "exp2_realtalk_grounding":
            return json.dumps({"grounding": False})
        if schema == "exp2_realtalk_epitome_empathy":
            return json.dumps({
                "emotional_reaction": 1,
                "interpretation": 1,
                "exploration": 0,
            })
        if system_prompt.startswith("You are Emi. Continue"):
            return "I feel good about the work update."
        return json.dumps({
            layer: {
                "item": {
                    "value": f"{layer}-value",
                    "confidence": 0.8,
                    "evidence": "Emi: train work update 1",
                }
            }
            for layer in ("core", "regulation", "cognition", "identity", "behavior")
        })


class FakeLabels:
    def annotate(self, _text):
        return {
            "emotion": "joy",
            "sentiment": "positive",
            "intimacy": 0.5,
        }

    def metadata(self):
        return {"provider": "fake-pinned-realtalk"}


def _chat(first_speaker, second_speaker, prefix):
    chat = {"name": {"speaker_1": first_speaker, "speaker_2": second_speaker}}
    for index in range(1, 4):
        chat[f"session_{index}"] = [
            {
                "speaker": first_speaker,
                "clean_text": f"{prefix} work update {index}",
                "dia_id": f"{prefix}-u{index}",
            },
            {
                "speaker": second_speaker,
                "clean_text": f"{prefix} partner reply {index}",
                "dia_id": f"{prefix}-p{index}",
            },
        ]
    return chat


class RevisedExp2Tests(unittest.TestCase):
    def test_bertscore_uses_fixed_inference_only_english_configuration(self):
        metadata = bertscore_runtime_metadata()
        self.assertEqual(metadata["model"], "roberta-large")
        self.assertEqual(metadata["num_layers"], 17)
        self.assertEqual(metadata["language"], "en")
        self.assertFalse(metadata["idf"])
        self.assertFalse(metadata["rescale_with_baseline"])
        self.assertFalse(metadata["requires_training"])

    def test_current_schema_is_strict(self):
        state = {
            "emotion": "joy",
            "sentiment": "positive",
            "topic": "work",
        }
        self.assertEqual(normalize_current_state(state), state)
        with self.assertRaisesRegex(ValueError, "must contain"):
            normalize_current_state({**state, "extra": True})
        self.assertEqual(
            normalize_reflectiveness({"reflective": True}),
            {"reflective": True},
        )
        self.assertEqual(
            normalize_grounding({"grounding": False}),
            {"grounding": False},
        )
        self.assertEqual(
            normalize_empathy({
                "emotional_reaction": 1,
                "interpretation": 2,
                "exploration": 0,
            }),
            {
                "emotional_reaction": 1,
                "interpretation": 2,
                "exploration": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "in \\[0, 2\\]"):
            normalize_empathy({
                "emotional_reaction": 3,
                "interpretation": 0,
                "exploration": 0,
            })

    def test_profile_shape_validation_rejects_missing_layers(self):
        with self.assertRaisesRegex(ValueError, "missing layers"):
            _normalize_explicit_profile({"core": {}})

    def test_target_profile_drops_partner_evidence(self):
        profile = {
            layer: {}
            for layer in ("core", "regulation", "cognition", "identity", "behavior")
        }
        profile["core"] = {
            "target_trait": {
                "value": "likes hiking",
                "confidence": 0.8,
                "evidence": "Emi: I enjoy hiking.",
            },
            "partner_trait": {
                "value": "lives in New York",
                "confidence": 0.9,
                "evidence": "Paola: I moved to New York.",
            },
            "partner_reflection": {
                "value": "fears relocation",
                "confidence": 0.8,
                "evidence": "Emi: It must have been hard for you to move.",
            },
        }
        profile["cognition"]["questioning_style"] = {
            "value": "asks supportive questions",
            "confidence": 0.8,
            "evidence": "Emi: How are you coping with the move?",
        }
        normalized = _normalize_target_profile(
            profile,
            "Emi",
            ["I enjoy hiking.", "How are you coping with the move?"],
        )
        self.assertIn("target_trait", normalized["core"])
        self.assertNotIn("partner_trait", normalized["core"])
        self.assertNotIn("partner_reflection", normalized["core"])
        self.assertIn("questioning_style", normalized["cognition"])

    def test_two_tracks_are_causal_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "Chat_4_Emi_Paola.json").write_text(
                json.dumps(_chat("Emi", "Paola", "train")),
                encoding="utf-8",
            )
            (dataset / "Chat_1_Emi_Elise.json").write_text(
                json.dumps(_chat("Emi", "Elise", "test")),
                encoding="utf-8",
            )
            config = Exp2UserModelingConfig(
                dataset_dir=str(dataset),
                output_dir=str(root / "output"),
                speaker_filter=["Emi"],
                max_eval_points_per_speaker=2,
                compute_bertscore=True,
            )
            llm = FakeLLM()
            evaluator = FakeLabels()
            with patch(
                "src.experiments.user_modeling.runner.compute_bertscore_f1",
                return_value=[0.8, 0.8, 0.8, 0.8],
            ) as bertscore:
                summary = run_user_modeling_evaluation(
                    config, llm=llm, label_evaluator=evaluator
                )
                first_call_count = len(llm.calls)
                resumed = run_user_modeling_evaluation(
                    config, llm=llm, label_evaluator=evaluator
                )
                self.assertEqual(bertscore.call_count, 1)

            self.assertEqual(summary["num_eval_points"], 2)
            self.assertEqual(resumed["num_eval_points"], 2)
            self.assertEqual(len(llm.calls), first_call_count)
            self.assertEqual(
                set(summary["current_understanding"]),
                {ZERO_SHOT, OURS},
            )
            self.assertEqual(
                set(summary["future_understanding"]),
                {ZERO_SHOT, OURS},
            )
            self.assertNotIn(FINE_TUNED, summary["future_understanding"])
            future_scores = summary["future_understanding"][OURS][
                "speaker_macro"
            ]
            self.assertEqual(future_scores["bertscore_f1"], 0.8)
            self.assertIn("reflectiveness_accuracy", future_scores)
            self.assertIn("grounding_accuracy", future_scores)
            self.assertIn("empathy_absolute_difference", future_scores)
            self.assertEqual(
                [item["session_count"] for item in summary["profile_evolution"]["Emi"]],
                [1, 2, 3],
            )
            profile_calls = [
                system_prompt
                for system_prompt, _, _ in llm.calls
                if "REALTALK TARGET-SPEAKER BINDING" in system_prompt
            ]
            self.assertEqual(len(profile_calls), 3)
            self.assertIn(
                'speaker label is exactly "Emi:"',
                profile_calls[0],
            )
            self.assertIn(
                "the partner's identity, experiences, preferences, or traits",
                profile_calls[0],
            )
            self.assertIn(
                "distinguish self-disclosure from partner-directed",
                profile_calls[0],
            )

            current_calls = [
                user_prompt
                for _, user_prompt, kwargs in llm.calls
                if (kwargs.get("response_schema") or {}).get("name")
                == "exp2_current_user_understanding"
            ]
            continuation_calls = [
                (system_prompt, user_prompt)
                for system_prompt, user_prompt, kwargs in llm.calls
                if system_prompt.startswith("You are Emi. Continue")
                and kwargs.get("response_schema") is None
            ]
            self.assertEqual(len(current_calls), 4)
            self.assertEqual(len(continuation_calls), 4)
            metric_schema_calls = [
                (kwargs.get("response_schema") or {}).get("name")
                for _, _, kwargs in llm.calls
                if (kwargs.get("response_schema") or {}).get("name", "").startswith(
                    "exp2_realtalk_"
                )
            ]
            self.assertEqual(
                metric_schema_calls.count("exp2_realtalk_reflectiveness"),
                6,
            )
            self.assertEqual(
                metric_schema_calls.count("exp2_realtalk_grounding"),
                6,
            )
            self.assertEqual(
                metric_schema_calls.count("exp2_realtalk_epitome_empathy"),
                6,
            )
            empathy_prompts = [
                user_prompt
                for _, user_prompt, kwargs in llm.calls
                if (kwargs.get("response_schema") or {}).get("name")
                == "exp2_realtalk_epitome_empathy"
            ]
            self.assertIn(
                "current speaker's own self-disclosure cannot count",
                empathy_prompts[0],
            )
            self.assertIn(
                "No prior partner message exists",
                empathy_prompts[0],
            )

            # Current-state inference observes the target.
            self.assertIn("test work update 1", current_calls[0])
            # Future generation never sees the target it must generate.
            self.assertNotIn("test work update 1", continuation_calls[0][1])
            self.assertNotIn("test work update 1", continuation_calls[1][1])
            # At the second point, the first completed exchange is legitimate history.
            self.assertIn("test work update 1", continuation_calls[2][1])
            self.assertIn("test partner reply 1", continuation_calls[2][1])
            self.assertNotIn("test work update 2", continuation_calls[2][1])
            self.assertNotIn("test work update 2", continuation_calls[3][1])

            baseline_prompt = continuation_calls[0][1]
            ours_prompt = continuation_calls[1][1]
            self.assertNotIn("FIVE-LAYER USER PROFILE", baseline_prompt)
            self.assertIn("FIVE-LAYER USER PROFILE", ours_prompt)
            self.assertEqual(baseline_prompt, "Emi")
            self.assertTrue(ours_prompt.endswith("Emi"))
            self.assertIn("END PRIVATE PROFILE.", ours_prompt)
            self.assertIn(
                "No current-partner conversation history is available",
                ours_prompt,
            )
            self.assertIn(
                "Output only the message, not the speaker name.",
                continuation_calls[0][0],
            )
            self.assertIn(
                "private background, not as shared history",
                continuation_calls[1][0],
            )
            self.assertIn(
                "never address yourself as Emi",
                continuation_calls[1][0],
            )

            output = Path(config.output_dir)
            lines = (output / "results.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                [json.loads(line)["message_level_index"] for line in lines],
                [0, 1],
            )
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "run_manifest.json").exists())
            self.assertTrue((output / "profile_evolution.json").exists())


if __name__ == "__main__":
    unittest.main()
