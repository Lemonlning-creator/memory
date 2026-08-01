import unittest
import os

from src.experiments.profile_quality_benchmark import (
    PERSONAS,
    TURNS_PER_PERSON,
    _turn_plan,
    aggregate_metrics,
    compute_metrics,
    disable_proxy_environment,
    flatten_profile_claims,
)
from src.profile_schema import PROFILE_FIELDS, PROFILE_LAYERS, create_empty_static_profile


class ProfileQualityBenchmarkTests(unittest.TestCase):
    def test_benchmark_disables_inherited_proxy_environment(self):
        names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
        original = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = "http://127.0.0.1:10809"
            disable_proxy_environment()
            self.assertTrue(all(name not in os.environ for name in names))
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_personas_cover_fixed_schema_and_fifty_turns(self):
        self.assertEqual(len(PERSONAS), 5)
        for persona in PERSONAS:
            self.assertEqual(len(persona["facts"]), 15)
            self.assertEqual(len(persona["corrections"]), 3)
            self.assertEqual(len(_turn_plan(persona)), TURNS_PER_PERSON)
            self.assertEqual(
                {fact["fact_id"] for fact in persona["facts"]},
                {f"{layer}.{field}" for layer in PROFILE_LAYERS for field in PROFILE_FIELDS[layer]},
            )

    def test_claim_flattening_excludes_summaries(self):
        profile = create_empty_static_profile()
        profile["core"]["summary"] = "不应成为独立评分条目。"
        profile["core"]["values"] = ["重视可靠性。", "重视长期积累。"]
        claims = flatten_profile_claims(profile)
        self.assertEqual([item["claim"] for item in claims], ["重视可靠性。", "重视长期积累。"]) 

    def test_metric_formulas(self):
        profile = create_empty_static_profile()
        for layer in PROFILE_LAYERS:
            profile[layer][PROFILE_FIELDS[layer][0]] = [f"{layer} claim"]
        dataset = {
            "persona_id": "p",
            "ground_truth": {"facts": [{"fact_id": "f1"}, {"fact_id": "f2"}], "corrections": [{"correction_id": "c1"}, {"correction_id": "c2"}]},
        }
        result = {"final_profile": profile, "calls": [{"latency_seconds": 2.0}, {"latency_seconds": 4.0}]}
        judgement = {
            "claim_assessments": [
                {"verdict": "supported"}, {"verdict": "partially_supported"},
                {"verdict": "unsupported"}, {"verdict": "contradicted"}, {"verdict": "supported"},
            ],
            "fact_assessments": [{"status": "captured"}, {"status": "partial"}],
            "correction_assessments": [{"status": "handled"}, {"status": "failed"}],
        }
        metrics = compute_metrics(dataset, result, judgement)
        self.assertEqual(metrics["profile_accuracy_percent"], 50.0)
        self.assertEqual(metrics["hallucination_percent"], 40.0)
        self.assertEqual(metrics["key_fact_recall_percent"], 75.0)
        self.assertEqual(metrics["correction_handling_percent"], 50.0)
        self.assertEqual(metrics["five_layer_completeness_percent"], 100.0)
        self.assertEqual(metrics["profile_average_latency_seconds"], 3.0)

    def test_aggregate_metrics(self):
        base = {
            "profile_accuracy_percent": 80,
            "hallucination_percent": 10,
            "key_fact_recall_percent": 70,
            "correction_handling_percent": 60,
            "five_layer_completeness_percent": 100,
            "fixed_field_coverage_percent": 80,
            "profile_average_latency_seconds": 5,
            "profile_api_calls": 7,
        }
        summary = aggregate_metrics([{"persona_id": "a", **base}, {"persona_id": "b", **base}])
        self.assertEqual(summary["sample_people"], 2)
        self.assertEqual(summary["dialogue_turns"], 100)
        self.assertEqual(summary["total_profile_api_calls"], 14)


if __name__ == "__main__":
    unittest.main()
