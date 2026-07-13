# REALTALK Experiment Suite — Complete Summary

## Overview

All 9 experiments across 3 groups have been completed, validating the three core innovations of the personalized companion agent using the REALTALK 21-day real-world conversation dataset (10 conversations, 728 QA pairs).

## Experiment Groups

### Group 2: Core Comparison Experiments

| Experiment | Code | Cases | Key Result |
|-----------|------|-------|------------|
| **2a** Persona Simulation | persona_simulation.py | 30 eval points | full_agent: ref=0.96(+60%), emp=1.54(+45%) vs baseline |
| **2b** Empathy Alignment | empathy_alignment_analysis.py | 29 cases | Aligned: 4.07/6 (+14.6%) vs Direct: 3.55/6, 0 excessive |
| **2c** Cross-Conversation Consistency | cross_conversation_consistency.py | 8+6 pairs | Same-person: 3.88/5 vs Different: 1.17/5, gap=2.71 |

### Group 1: Core Ablation Experiments

| Experiment | Code | Scope | Key Result |
|-----------|------|-------|------------|
| **1a** Profile Time Evolution | profile_time_evolution.py | 10 chats × 3 windows | Early-vs-late: 3.30/5, 88 novel traits accumulated |
| **1b** State Axis Validation | state_context_axis.py | 10 chats × 5 checkpoints | 3.7 unique emotions/chat, 37.5% valence transition rate |
| **1c** Context Axis Validation | state_context_axis.py | 10 chats × 3 segments | 2.1 unique contexts/chat, 114 unique context traits |

### Group 3: Additional Validation Experiments

| Experiment | Code | Scope | Key Result |
|-----------|------|-------|------------|
| **3a** Memory Probing | memory_association.py | 80 QA questions | Profile-guided: 0.55 (+12.9%) vs Random: 0.487 |
| **3b** Profile-Evidence Chain | memory_association.py | 37 chains | Avg 3.9 connecting tags bridging 4.5 evidence items |

## Innovation Validation Summary

### Innovation 1: Multi-Dimensional User Profile
- **State axis** (1b): validated — 3.7 unique emotions/chat, dynamic state tracking with 37.5% transition rate
- **Context axis** (1c): validated — 2.1 unique contexts/chat, 6 context types, 114 unique traits
- **Time axis** (1a): validated — profiles evolve (3.30/5 early-vs-late) while preserving stable traits, 88 new traits accumulated
- **Stability** (2c): validated — same-person consistency 3.88/5 vs different-person 1.17/5 (gap=2.71)

### Innovation 2: Empathy Alignment Mechanism
- **Alignment improves empathy** (2b): +14.6% EPITOME score, zero excessive-empathy failures
- **Full pipeline advantage** (2a): empathy score +45% over baseline, reflectiveness +60%
- **Strongest in high-distress cases** (2b): clear wins when user expresses genuine distress

### Innovation 3: Implicit Memory Association
- **Profile improves retrieval** (3a): +12.9% over random, prevents "insufficient evidence" failures
- **Tags connect disparate evidence** (3b): average 3.9 tags bridging 4.5 evidence items from different sessions

## File Index

### Experiment Code
- `src/experiments/persona_simulation.py` — Experiment 2a
- `src/experiments/empathy_alignment_analysis.py` — Experiment 2b
- `src/experiments/cross_conversation_consistency.py` — Experiment 2c
- `src/experiments/profile_time_evolution.py` — Experiment 1a
- `src/experiments/state_context_axis.py` — Experiments 1b & 1c
- `src/experiments/memory_association.py` — Experiments 3a & 3b

### Results Documents
- `src/experiments/PERSONA_SIMULATION_RESULTS.md` — 2a analysis
- `src/experiments/EMPATHY_ALIGNMENT_RESULTS.md` — 2b analysis
- `src/experiments/CROSS_CONVERSATION_RESULTS.md` — 2c analysis
- `src/experiments/TIME_EVOLUTION_RESULTS.md` — 1a analysis
- `src/experiments/STATE_CONTEXT_AXIS_RESULTS.md` — 1b & 1c analysis
- `src/experiments/MEMORY_ASSOCIATION_RESULTS.md` — 3a & 3b analysis

### Data Outputs
- `data/persona_simulation_eval/` — 2a results
- `data/empathy_alignment_eval/` — 2b results
- `data/consistency_eval/` — 2c results
- `data/time_evolution_eval/` — 1a results
- `data/state_axis_eval/` — 1b results
- `data/context_axis_eval/` — 1c results
- `data/memory_eval/` — 3a & 3b results

### Prompts
- `src/prompts/templates_en.py` — All English prompts
- `src/prompts/prompt_loader.py` — Centralized prompt loader (language toggle)
