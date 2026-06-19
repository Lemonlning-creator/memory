# Experiments

Experiment entry points live here and should be run from the project root.

## 1. Profile Generation

Generate a user profile from REALTALK dialogue.

```powershell
python -m src.experiments.profile_generation_pipeline --realtalk "D:\Postgraduate\Code\test\REALTALK\data\Chat_1_Emi_Elise.json"
```

## 2. Evidence Support Evaluation

Evaluate whether profile claims are supported by original dialogue evidence.

```powershell
python -m src.experiments.evidence_support_eval --realtalk "D:\Postgraduate\Code\test\REALTALK\data\Chat_1_Emi_Elise.json" --profile "user\Emi_Kate_profile.json" --user-name Emi_Kate --output "data\profile_evidence_eval\Emi_Kate_embedding"
```

## 3. Experiment 1: Profile And Memory Gain For Agent Reply

This experiment compares two conditions on the same selected samples:

```text
baseline            = history + latest target-user input
with_profile_memory = history + latest target-user input + agent speaker persona + target-user profile + retrieved mid-term memory
```

The model role-plays the second speaker and generates the next reply to the first speaker. The human second-speaker reply is used as reference for LLM judging.

The sample filter keeps cases where the first speaker's latest turn expresses emotion, plans, interests, confusion, or needs. It filters out cases where the first speaker is mainly asking the partner's personal facts.

Quick run:

```powershell
python -m src.experiments.empathic_response_eval --realtalk "D:\Postgraduate\Code\test\REALTALK\data\Chat_1_Emi_Elise.json" --profile "user\Emi_Kate_profile.json" --persona "agent\elise_persona.json" --memory-dir "data\realtalk_memory_runs\Chat_1_Emi_Elise_Emi_20260608_220406" --user-name Emi --max-samples 5 --mid-term-limit 1
```

Fuller run:

```powershell
python -m src.experiments.empathic_response_eval --realtalk "D:\Postgraduate\Code\test\REALTALK\data\Chat_1_Emi_Elise.json" --profile "user\Emi_Kate_profile.json" --persona "agent\elise_persona.json" --memory-dir "data\realtalk_memory_runs\Chat_1_Emi_Elise_Emi_20260608_220406" --user-name Emi --max-samples 20 --mid-term-limit 1
```

Outputs are saved under:

```text
data/experiments/empathic_response_eval/<run_id>/
```

Main files:

```text
summary.json
<chat_id>/samples.jsonl
<chat_id>/results.jsonl
<chat_id>/results.json
<chat_id>/summary.json
<chat_id>/process.jsonl
```

The summary reports structured metrics for both conditions and deltas:

```text
baseline_avg_reflective
with_profile_memory_avg_reflective
delta_reflective

baseline_avg_grounding
with_profile_memory_avg_grounding
delta_grounding

baseline_avg_emotion
with_profile_memory_avg_emotion
delta_emotion

baseline_avg_empathy
with_profile_memory_avg_empathy
delta_empathy

baseline_avg_intimacy
with_profile_memory_avg_intimacy
delta_intimacy

baseline_avg_persona_consistency
with_profile_memory_avg_persona_consistency
delta_persona_consistency

baseline_avg_human_likeness
with_profile_memory_avg_human_likeness
delta_human_likeness

baseline_avg_overall
with_profile_memory_avg_overall
delta_overall

empathy_wins

pairwise_empathy_preference_wins
pairwise_grounding_preference_wins
pairwise_persona_consistency_preference_wins
pairwise_overall_preference_wins

pairwise_overall_preference_with_profile_memory_win_rate
primary_result
```

All score metrics are judged from 1 to 5:

```text
1 = very poor
2 = weak
3 = acceptable
4 = good
5 = excellent
```

## 4. Agent Persona Generation

Generate an agent persona from the second speaker's REALTALK utterances. The output is saved as `agent/{speaker_name}_persona.json`.

```powershell
python generate_agent_persona.py --realtalk "D:\Postgraduate\Code\test\REALTALK\data\Chat_1_Emi_Elise.json"
```

Specify the second speaker manually if needed:

```powershell
python generate_agent_persona.py --realtalk "D:\Postgraduate\Code\test\REALTALK\data\Chat_1_Emi_Elise.json" --speaker-name Kate
```
