from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from src.agent import StateDrivenCompanionAgent
from src.experiments.exp2_user_modeling import ExperimentCase, TABLE2_METRICS
from src.experiments.exp3_ablation_study import (
    CONDITIONS,
    EXPLICIT_TRAIN_RATIO,
    ONLINE_TRAIN_RATIO,
    Exp3Agent,
    build_parser,
    condition_prompt_bundle,
    condition_root,
    dataset_comparison_payload,
    run_dataset_replay,
    run_exploration_simulation,
    scenario_path,
    select_conditions,
    select_tracks,
    surface_style_guidance,
)
from src.experiments.exp3_user_simulator import (
    HiddenClaim,
    HiddenProfileUserSimulator,
    aggregate_discovery_results,
    atomic_profile_claims,
    build_hidden_claim_manifest,
    validate_simulator_payload,
)
from src.prompts.exp2_versions import get_exp2_prompt_bundle


class _FakeMemory:
    def __init__(self, *args, **kwargs) -> None:
        self.short_term_memory = []

    def append_stm(self, role, content):
        self.short_term_memory.append({"role": role, "content": content})

    def retrieve_relevant_memory(self, user_input):
        return {"recent_messages": list(self.short_term_memory), "mid_term_summaries": []}


class _FakeTracker:
    def __init__(self) -> None:
        self.interaction_count = 0

    def increment(self):
        self.interaction_count += 1

    def compute(self, profile):
        return 0.4


class _FakeAgent:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.memory_manager = _FakeMemory()
        self.epistemic_tracker = _FakeTracker()
        self.user_profile = {
            "state_axis": {
                "static_profile": {"core": {}},
                "current_state": {},
                "projected_state": {},
            },
            "context_axis": {},
        }
        self.last_empathy_state = {}
        self.last_prediction = {}
        self.last_agent_response = ""
        self.instances.append(self)

    def _run_memory_steps(self):
        return None

    def finalize_session(self):
        return {"flushed_mid_term_ids": [], "long_term_memory_id": None}


class _FakeSimulator:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def respond(self, conversation, agent_message):
        return {
            "user_reply": "simulated next user",
            "revealed_claim_ids": [],
            "disclosure_strength": {},
            "withheld_or_refused": False,
            "perceived_burden": 0,
            "burden_reason": "ordinary response",
            "disclosure_decision": "none",
            "disclosure_depth": "none",
            "trust": 0.4,
            "fatigue": 0.0,
        }


class Exp3ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAgent.instances.clear()

    def workspace_tmp(self) -> Path:
        path = Path("tmp") / f"exp3_test_{uuid4().hex}"
        path.mkdir(parents=True)
        return path

    def test_cli_has_component_specific_split_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.phase, "plan")
        self.assertEqual(args.explicit_train_ratio, EXPLICIT_TRAIN_RATIO)
        self.assertEqual(args.online_train_ratio, ONLINE_TRAIN_RATIO)
        self.assertEqual(args.sim_rounds, 20)

    def test_default_exploration_is_capability_only_without_invented_comparator(self) -> None:
        selected = select_conditions(select_tracks([]), [])
        self.assertEqual([len(selected[track]) for track in select_tracks([])], [2, 1, 2])
        self.assertEqual(
            [condition.key for values in selected.values() for condition in values],
            [
                "explicit_user_modeling", "wo_explicit_user_modeling",
                "adaptive_exploration", "bayesian_online", "static_profile",
            ],
        )

    def test_condition_cannot_be_selected_from_wrong_track(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not belong"):
            select_conditions(["explicit"], ["bayesian_online"])

    def test_explicit_and_bayesian_controls_hold_other_switches_constant(self) -> None:
        explicit = CONDITIONS["explicit_user_modeling"]
        no_explicit = CONDITIONS["wo_explicit_user_modeling"]
        self.assertEqual(explicit.exploration_mode, no_explicit.exploration_mode)
        self.assertEqual(explicit.update_mode, no_explicit.update_mode)
        self.assertNotEqual(explicit.uses_explicit_profile, no_explicit.uses_explicit_profile)

        online = CONDITIONS["bayesian_online"]
        static = CONDITIONS["static_profile"]
        self.assertEqual(online.modeling_mode, static.modeling_mode)
        self.assertEqual(online.exploration_mode, static.exploration_mode)
        self.assertNotEqual(online.update_mode, static.update_mode)

    def test_no_explicit_prompt_and_context_mask_profile(self) -> None:
        condition = CONDITIONS["wo_explicit_user_modeling"]
        base = get_exp2_prompt_bundle("v3_realtalk_aligned")
        masked = condition_prompt_bundle(base, condition)
        self.assertNotEqual(masked.fingerprint, base.fingerprint)

        agent = object.__new__(Exp3Agent)
        agent.condition = condition
        with patch.object(
            StateDrivenCompanionAgent,
            "_prompt_context",
            return_value={"static_profile": "secret"},
        ):
            context = agent._prompt_context("hello", {})
        self.assertNotIn("secret", context["static_profile"])

    def test_scenario_path_is_shared_across_exploration_conditions(self) -> None:
        case = ExperimentCase(
            case_id="case", dataset_path="unused", user_speaker="U", agent_speaker="A",
            train_sessions=("session_1",), test_sessions=("session_2",), dataset_sha256="h",
        )
        path = scenario_path(self.workspace_tmp(), case, 2)
        self.assertNotIn("adaptive_exploration", str(path))
        self.assertNotIn("fixed_exploration", str(path))
        self.assertTrue(str(path).endswith("seed_002.json"))

    def test_dataset_replay_writes_reference_not_generated_reply_to_memory(self) -> None:
        chat = {
            "name": {"speaker_1": "User", "speaker_2": "Agent"},
            "session_1": [],
            "session_2": [
                {"speaker": "User", "clean_text": "hello", "dia_id": "D2:1"},
                {"speaker": "Agent", "clean_text": "real reply", "dia_id": "D2:2"},
            ],
        }
        case = ExperimentCase(
            case_id="case", dataset_path="unused.json", user_speaker="User",
            agent_speaker="Agent", train_sessions=("session_1",),
            test_sessions=("session_2",), dataset_sha256="hash",
        )
        condition = CONDITIONS["static_profile"]
        tmp = self.workspace_tmp()
        root = condition_root(tmp, condition)
        assets = root / "cases" / "case" / "assets"
        assets.mkdir(parents=True)
        (assets / "agent_persona.json").write_text("{}", encoding="utf-8")
        (assets / "user_profile.json").write_text("{}", encoding="utf-8")
        (assets / "user_profile_runtime.json").write_text("{}", encoding="utf-8")
        with (
            patch("src.experiments.exp3_ablation_study.Exp3Agent", _FakeAgent),
            patch("src.experiments.exp3_ablation_study.MemoryOSLocal", _FakeMemory),
            patch("src.experiments.exp3_ablation_study.load_json", return_value=chat),
            patch(
                "src.experiments.exp3_ablation_study._generate_with_parallel_alignment",
                return_value=(
                    "generated reply",
                    {"understanding": {}, "prediction": {}, "exploration": {}},
                    {},
                    {},
                ),
            ),
        ):
            count = run_dataset_replay(
                case, tmp, condition, "config.ini", "v3_realtalk_aligned", 0.5
            )
        self.assertEqual(count, 1)
        memory = _FakeAgent.instances[0].memory_manager.short_term_memory
        self.assertIn({"role": "assistant", "content": "real reply"}, memory)
        self.assertNotIn({"role": "assistant", "content": "generated reply"}, memory)

    def test_exploration_uses_generated_agent_reply_as_interactive_history(self) -> None:
        case = ExperimentCase(
            case_id="case", dataset_path="unused.json", user_speaker="User",
            agent_speaker="Agent", train_sessions=("session_1",),
            test_sessions=("session_2",), dataset_sha256="hash",
        )
        condition = CONDITIONS["adaptive_exploration"]
        tmp = self.workspace_tmp()
        assets = condition_root(tmp, condition) / "cases" / "case" / "assets"
        assets.mkdir(parents=True)
        runtime = {
            "state_axis": {"static_profile": {"core": {}}, "current_state": {}, "projected_state": {}},
            "context_axis": {},
        }
        (assets / "agent_persona.json").write_text("{}", encoding="utf-8")
        (assets / "user_profile.json").write_text("{}", encoding="utf-8")
        (assets / "user_profile_runtime.json").write_text(
            json.dumps(runtime), encoding="utf-8"
        )
        scenario = {
            "user_reply": "shared opener", "revealed_claim_ids": [],
            "disclosure_strength": {}, "withheld_or_refused": False,
            "perceived_burden": 0, "burden_reason": "ordinary opening",
            "disclosure_decision": "opening", "disclosure_depth": "none",
            "trust": 0.35, "fatigue": 0.0,
        }
        with (
            patch("src.experiments.exp3_ablation_study.Exp3Agent", _FakeAgent),
            patch("src.experiments.exp3_ablation_study.MemoryOSLocal", _FakeMemory),
            patch("src.experiments.exp3_ablation_study.LLMClient", return_value=object()),
            patch(
                "src.experiments.exp3_ablation_study.HiddenProfileUserSimulator",
                _FakeSimulator,
            ),
            patch(
                "src.experiments.exp3_ablation_study._simulator_assets",
                return_value=([
                    HiddenClaim(
                        "H001", "core.values", "hidden", "new",
                        ("D1",), ("hidden evidence",), "low",
                    )
                ], {"style_examples": []}),
            ),
            patch(
                "src.experiments.exp3_ablation_study.ensure_scenario",
                return_value=scenario,
            ),
            patch(
                "src.experiments.exp3_ablation_study._generate_with_parallel_alignment",
                return_value=(
                    "generated interactive reply",
                    {"prediction": {}, "exploration": {}},
                    {},
                    {},
                ),
            ),
        ):
            generated = run_exploration_simulation(
                case, tmp, condition, "config.ini", "v3_realtalk_aligned",
                0.5, seed_index=1, max_rounds=2,
            )
        self.assertEqual(generated, 2)
        memory = _FakeAgent.instances[0].memory_manager.short_term_memory
        self.assertEqual(
            [row for row in memory if row["role"] == "assistant"],
            [
                {"role": "assistant", "content": "generated interactive reply"},
                {"role": "assistant", "content": "generated interactive reply"},
            ],
        )

    def test_dataset_comparison_is_paired_and_bayesian_is_temporally_segmented(self) -> None:
        conditions = [CONDITIONS["bayesian_online"], CONDITIONS["static_profile"]]
        metric_stats = {
            metric: {"mean": 1.0, "std": 0.0} for metric in TABLE2_METRICS
        }
        aggregate = {
            "example_count": 1, "speaker_count": 1, "ours": metric_stats,
        }
        reference_score = {
            "example_id": "e1",
            **{metric: 1.0 for metric in TABLE2_METRICS},
        }
        candidate_score = {
            "example_id": "e1",
            **{metric: 0.5 for metric in TABLE2_METRICS},
        }
        payload = dataset_comparison_payload(
            {condition.key: aggregate for condition in conditions},
            conditions,
            {
                "bayesian_online": [reference_score],
                "static_profile": [candidate_score],
            },
            {"e1": "late"},
            {condition.key: {} for condition in conditions},
        )
        self.assertTrue(payload["temporal_segmentation_required"])
        self.assertEqual(payload["evaluation_scope"], "comparative")
        comparison = payload["paired_comparisons"][0]
        self.assertIn("late", comparison["early_middle_late_mean_paired_degradation"])
        self.assertEqual(comparison["per_example"][0]["example_id"], "e1")

    def test_style_guidance_contains_no_raw_user_content(self) -> None:
        secret = "I secretly love obscure astronomy documentaries!"
        guidance = surface_style_guidance([secret, "Really?"])
        serialized = " ".join(guidance)
        self.assertNotIn("astronomy", serialized)
        self.assertIn("mean_words_per_message", serialized)


class Exp3SimulatorTests(unittest.TestCase):
    def test_atomic_claims_exclude_redundant_summaries(self) -> None:
        profile = {
            "core": {
                "summary": "summary",
                "values": ["values fairness", "values stability"],
            },
            "behavior": {"interests": ["likes films"]},
        }
        claims = atomic_profile_claims(profile, "H")
        self.assertEqual([claim.claim_id for claim in claims], ["H001", "H002", "H003"])
        self.assertNotIn("summary", [claim.text for claim in claims])

    def test_simulator_rejects_invented_claim_ids_and_invalid_burden(self) -> None:
        payload = {
            "user_reply": "hi", "revealed_claim_ids": [],
            "disclosure_strength": {}, "withheld_or_refused": False,
            "perceived_burden": 0, "burden_reason": "ordinary response",
            "disclosure_decision": "none", "disclosure_depth": "none",
            "trust": 0.4, "fatigue": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "fabricated"):
            validate_simulator_payload(
                dict(payload, revealed_claim_ids=["H999"], disclosure_strength={"H999": 1.0}),
                {"H001"},
            )
        with self.assertRaisesRegex(ValueError, "perceived_burden"):
            validate_simulator_payload(
                dict(payload, perceived_burden=3),
                {"H001"},
            )

    def test_simulator_rejects_missing_fields_instead_of_defaulting(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            validate_simulator_payload({"user_reply": "hi"}, {"H001"})

    def test_discovery_aggregation_reports_mean_and_std(self) -> None:
        base = {
            "case_id": "c",
            "initial_hidden_coverage": 0.0, "final_hidden_coverage": 0.7,
            "hidden_coverage_gain": 0.7, "elicitation_rate": 0.8,
            "uptake_rate": 0.75, "end_to_end_discovery_rate": 0.7,
            "novel_claim_precision": 0.8, "unsupported_novel_claim_rate": 0.2,
            "discovery_efficiency": 0.02, "coverage_auc": 0.4,
            "mean_user_burden": 0.2, "refusal_rate": 0.1,
            "exploration_question_rate": 0.5,
        }
        second = dict(base, case_id="d", final_hidden_coverage=0.9)
        aggregate = aggregate_discovery_results([base, second])
        self.assertEqual(aggregate["case_count"], 2)
        self.assertAlmostEqual(aggregate["final_hidden_coverage"]["mean"], 0.8)
        self.assertAlmostEqual(aggregate["final_hidden_coverage"]["std"], 0.1)

    def test_two_stage_renderer_never_sees_unapproved_hidden_claim(self) -> None:
        class FakeLLM:
            def __init__(self):
                self.prompts = []

            def chat(self, system, prompt, **kwargs):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return json.dumps({
                        "decision": "disclose", "allowed_claim_ids": ["H001"],
                        "disclosure_depth": "full", "perceived_burden": 0,
                        "rationale": "relevant question", "next_trust": 0.5,
                        "next_fatigue": 0.0,
                    })
                return json.dumps({
                    "user_reply": "I do like quiet films.",
                    "evidenced_claim_ids": ["H001"],
                })

        llm = FakeLLM()
        simulator = HiddenProfileUserSimulator(
            llm=llm,
            user_name="User",
            initial_profile={"core": {"values": ["kindness"]}},
            hidden_claims=[
                HiddenClaim("H001", "behavior.interests", "likes quiet films", "new", ("D1",), ("films",), "low"),
                HiddenClaim("H002", "core.fears", "fears abandonment", "new", ("D2",), ("fear",), "high"),
            ],
            style_examples=[], seed_index=1,
        )
        result = simulator.respond(
            [{"speaker": "Agent", "text": "What films do you enjoy?"}],
            "What films do you enjoy?",
        )
        self.assertEqual(result["revealed_claim_ids"], ["H001"])
        self.assertIn("likes quiet films", llm.prompts[1])
        self.assertNotIn("fears abandonment", llm.prompts[1])

    def test_hidden_targets_require_semantic_novelty_and_heldout_evidence(self) -> None:
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def chat(self, system, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"claims": [{
                        "candidate_id": "E001", "path": "behavior.interests",
                        "text": "likes quiet films", "evidence_ids": ["D2"],
                        "stability": "stable",
                    }]})
                return json.dumps({"audit": [
                    {
                        "full_claim_id": "P001", "relation": "known",
                        "matched_initial_claim_ids": ["I001"],
                        "matched_evidence_claim_ids": [],
                    },
                    {
                        "full_claim_id": "P002", "relation": "new",
                        "matched_initial_claim_ids": [],
                        "matched_evidence_claim_ids": ["E001"],
                    },
                ]})

        manifest = build_hidden_claim_manifest(
            FakeLLM(), "User",
            {"core": {"values": ["kindness"]}},
            {
                "core": {"values": ["kindness"]},
                "behavior": {"interests": ["likes quiet films"]},
            },
            {"D2": "I usually prefer quiet films."},
        )
        self.assertEqual(len(manifest["hidden_claims"]), 1)
        self.assertEqual(manifest["hidden_claims"][0]["id"], "H001")
        self.assertEqual(manifest["hidden_claims"][0]["evidence_ids"], ["D2"])


if __name__ == "__main__":
    unittest.main()
