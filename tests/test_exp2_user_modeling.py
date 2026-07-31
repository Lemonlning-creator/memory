import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.user_modeling.runner import (
    FINE_TUNED,
    OURS,
    ZERO_SHOT,
    Exp2UserModelingConfig,
    run_user_modeling_evaluation,
)
from src.experiments.user_modeling.schemas import normalize_current_state


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
        if system_prompt.startswith("You are Emi. Continue"):
            return "I feel good about the work update."
        return json.dumps({
            layer: {
                "item": {
                    "value": f"{layer}-value",
                    "confidence": 0.8,
                    "evidence": "training conversation",
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
    def test_current_schema_is_strict(self):
        state = {
            "emotion": "joy",
            "sentiment": "positive",
            "topic": "work",
        }
        self.assertEqual(normalize_current_state(state), state)
        with self.assertRaisesRegex(ValueError, "must contain"):
            normalize_current_state({**state, "extra": True})

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
            )
            llm = FakeLLM()
            evaluator = FakeLabels()
            summary = run_user_modeling_evaluation(
                config, llm=llm, label_evaluator=evaluator
            )
            first_call_count = len(llm.calls)
            resumed = run_user_modeling_evaluation(
                config, llm=llm, label_evaluator=evaluator
            )

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

            output = Path(config.output_dir)
            lines = (output / "results.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
