# Experiments 1b & 1c: State Axis + Context Axis Validation — Results (Ablation)

## Overview

These experiments validate the STATE AXIS and CONTEXT AXIS of the multi-dimensional user profile (Innovation 1):

- **1b (State Axis):** Tests whether the "current state" captures transient emotional states that change over time, distinct from the stable profile.
- **1c (Context Axis):** Tests whether different conversation contexts surface context-specific traits.

## Experiment 1b: State Axis Results

### Method
Extract 5 "current state" snapshots at evenly-spaced checkpoints through each conversation. Measure how much the emotional state varies.

### Headline Results

| Metric | Value |
|--------|-------|
| Chats analyzed | 10 |
| Checkpoints per chat | 5 |
| Avg unique emotions per chat | **3.7** |
| Valence transition rate | **37.5%** (15/40 transitions) |
| Positive states | 30 (50%) |
| Neutral states | 13 (21.7%) |
| Negative states | 7 (11.7%) |

### Per-Conversation State Trajectories

| Conversation | Valence Sequence | Unique Emotions |
|--------------|-----------------|-----------------|
| Chat_10_Fahim_Muhhamed | positive → positive → neutral → neutral → positive | 3 |
| Chat_1_Emi_Elise | positive → positive → positive → positive → positive | 1 |
| Chat_2_Kevin_Elise | positive → positive → positive → positive → positive | 3 |
| Chat_3_Kevin_Paola | positive → positive → positive → positive → positive | 4 |
| Chat_4_Emi_Paola | positive → positive → positive → positive → **negative** | 4 |
| Chat_5_Nicolas_Nebraas | neutral → neutral → neutral → neutral → **negative** | 3 |
| Chat_6_Vanessa_Nicolas | neutral → **negative** → **negative** → neutral → neutral | 4 |
| Chat_7_Nebraas_Vanessa | positive → neutral → positive → positive → positive | 5 |
| Chat_8_Akib_Muhhamed | positive → neutral → **negative** → neutral → **negative** | 5 |
| Chat_9_Fahim_Akib | neutral → positive → **negative** → positive → positive | 5 |

### Key Findings (State Axis)

1. **States are genuinely dynamic** — 3.7 unique emotions per chat on average, with some conversations showing all 5 checkpoints as distinct emotions.
2. **37.5% transition rate** confirms the current state is not static — it tracks the user's actual emotional journey.
3. **Emotionally complex conversations** (Chat_8, Chat_9) show rich state trajectories: positive→neutral→negative→neutral→negative, capturing the full emotional arc.
4. **Negative states surface naturally** — Chat_4 transitions to "grieving", Chat_5 to "anxious", validating that the state axis captures distress signals.

---

## Experiment 1c: Context Axis Results

### Method
Split each conversation into 3 chronological segments. Extract a context-specific profile from each segment, identifying the dominant context and context-specific traits.

### Headline Results

| Metric | Value |
|--------|-------|
| Chats analyzed | 10 |
| Avg unique contexts per chat | **2.1** |
| Total context traits extracted | 121 |
| Unique trait types | 114 |
| Context types observed | 6 (entertainment, relationships, education, health, daily_life, other) |

### Per-Conversation Context Trajectories

| Conversation | Context Sequence |
|--------------|-----------------|
| Chat_10_Fahim_Muhhamed | entertainment → entertainment/leisure → entertainment/leisure |
| Chat_1_Emi_Elise | entertainment/leisure → (mixed) → relationships |
| Chat_2_Kevin_Elise | relationships → education → health |
| Chat_3_Kevin_Paola | entertainment → health → entertainment/leisure |
| Chat_4_Emi_Paola | entertainment/leisure → (mixed) → health |
| Chat_5_Nicolas_Nebraas | (mixed) → daily_life → (mixed) |
| Chat_6_Vanessa_Nicolas | daily_life → (mixed) → daily_life |
| Chat_7_Nebraas_Vanessa | (mixed) → daily_life → daily_life |
| Chat_8_Akib_Muhhamed | entertainment/leisure → health → daily_life |
| Chat_9_Fahim_Akib | relationships → health → daily_life |

### Key Findings (Context Axis)

1. **Contexts genuinely vary across segments** — 2.1 unique contexts per chat, with 4 conversations showing 3 distinct contexts.
2. **121 traits extracted, 114 unique** — each context surfaces genuinely different facets of the user.
3. **Context transitions are natural** — conversations evolve from surface topics (entertainment) to deeper ones (health, relationships), mirroring real relationship development.
4. **Recurring traits validate stability** — a few traits recur across contexts (cultural curiosity, reading habits, cooking interest), confirming the stable profile underneath the context-specific layer.

---

## Combined Conclusion

The state axis and context axis are validated as distinct, meaningful dimensions:

- **State axis** captures transient emotional dynamics (3.7 unique emotions/chat, 37.5% transition rate) — this is the "current state" that the empathy alignment mechanism responds to.
- **Context axis** captures domain-specific traits (2.1 unique contexts/chat, 114 unique traits) — this is what enables context-aware personalization.

Together with the time axis (Experiment 1a), these confirm that the multi-dimensional user model has three independently validatable dimensions, each capturing distinct aspects of the user.
