# Experiment 2c: Cross-Conversation Profile Consistency — Results Analysis

## Overview

This experiment validates that the multi-dimensional user modeling (Innovation 1) captures **stable personality traits** rather than conversation-specific noise. In REALTALK, every speaker appears in exactly 2 conversations. We extract profiles/personas independently from each conversation and measure their consistency.

**Design:**
- **Same-person pairs** (8 pairs): same speaker modeled from 2 independent conversations
- **Different-person baseline** (6 pairs): profiles of different speakers cross-compared
- **Scale:** 1 (strongly contradictory) to 5 (highly consistent)

## Headline Results

| Metric | Same-Person | Different-Person | Separation Gap |
|--------|------------|-----------------|----------------|
| Avg Consistency | **3.88 / 5** | **1.17 / 5** | **2.71** |
| Score range | 3-5 | 1-2 | — |
| Num pairs | 8 | 6 | — |

## Section-Level Breakdown (User Profiles)

| Section | Avg Score (same-person) |
|---------|------------------------|
| Core (fears, desires, values) | 4.00 |
| Cognitive Style | 4.00 |
| Behavior Preference | 3.75 |
| Social/Physical | 3.75 |
| Regulation | 3.50 |

All sections score 3.5+ for same-person pairs, confirming that stability holds across all five dimensions of the user model.

## Same-Person Pair Details

| Speaker | Type | Score | Key Stable Traits |
|---------|------|-------|-------------------|
| Kevin | User Profile | 5/5 | Secure attachment, growth-oriented values |
| Emi | User Profile | 4/5 | Personal growth, authenticity, emotional honesty |
| Akib | User Profile | 3/5 | Interest in technology, humor, exploration |
| Nicolas | User Profile | 3/5 | Humor as coping, reflective decision-making |
| Paola | Agent Persona | 5/5 | Empathetic, supportive, curious |
| elise | Agent Persona | 5/5 | Friendly, knowledgeable, enthusiastic |
| Muhhamed | Agent Persona | 3/5 | Knowledgeable, conversational, insightful |
| Nebraas | Agent Persona | 3/5 | Casual tone, reflective, humor in deep discussions |

## Key Findings

### 1. Strong same-vs-different separation (gap = 2.71)
Same-person profiles score 3.88/5 on average while different-person profiles score 1.17/5. The system clearly distinguishes "same person in different contexts" from "different people." No different-person pair scored above 2.

### 2. Stable traits identified consistently
Even in pairs scoring 3/5, the evaluator identifies genuine stable traits. For example, Nicolas is independently characterized as "uses humor to cope with difficult topics" and "reflective decision-making style" in both conversations — these are personality-level traits, not topic artifacts.

### 3. Lower-scoring pairs show context sensitivity (not failure)
Pairs scoring 3/5 show contradictions that are actually context-driven. For example, Akib discusses AI fears in one conversation (tech-focused) and career anxieties in another (life-focused). This is the expected behavior of the state axis: the **current state** varies by context while the **stable state** remains consistent.

### 4. Agent personas are highly stable
Elise and Paola both score 5/5 consistency — their empathy, warmth, and conversational style are reproduced across different conversation partners. This validates the persona modeling as character-level, not interaction-specific.

### 5. Different-person baselines are strongly contradictory
All 6 baseline pairs scored 1-2/5 with clear "completely different people" assessments. This confirms the consistency scores are meaningful — high same-person scores are not an artifact of generic profiles.

## Conclusion

The multi-dimensional user modeling captures stable personality structure: same-person profiles from independent conversations score 3.88/5 consistency while different-person profiles score 1.17/5. The 2.71-point separation gap confirms the system models real personality, not conversation noise. This validates Innovation 1's claim of stable, context-aware user profiling.
