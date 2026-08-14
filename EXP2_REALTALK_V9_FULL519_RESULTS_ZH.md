# REALTALK V9 Full-519 Results

## Scope

- Protocol: REALTALK Table 8, first three consecutive Ca/Cb sessions, 10 target speakers.
- Predictions: 519/519, SHA256 `ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81`.
- Aggregation: speaker macro mean and population standard deviation.
- GPT Judge: Appendix C full-prompt within-session v3, `gpt-4o-mini`.
- Judge completeness: 3,114/3,114 cells, zero unresolved errors.
- Comparison status: protocol-aligned reconstruction. The paper does not disclose an identical base-model runtime.

## Table 2 Comparison

Higher is better for ROUGE, BERTScore, Reflectiveness, Grounding, Sentiment, and Emotion. Lower is better for Intimacy AD and Empathy AD.

| Method | ROUGE | BERTScore | Reflect. | Grounding | Sentiment | Emotion | Intimacy AD | Empathy AD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper w/o FT | 0.14 +/- 0.04 | 0.76 +/- 0.08 | 0.62 +/- 0.13 | 0.40 +/- 0.13 | 0.53 +/- 0.22 | 0.43 +/- 0.22 | **0.06 +/- 0.01** | 1.80 +/- 0.55 |
| Paper w/ FT | 0.14 +/- 0.05 | 0.78 +/- 0.04 | **0.77 +/- 0.09** | **0.62 +/- 0.08** | 0.59 +/- 0.18 | 0.46 +/- 0.21 | 0.07 +/- 0.01 | 1.24 +/- 0.12 |
| Ours V9 | **0.154 +/- 0.033** | **0.858 +/- 0.012** | 0.698 +/- 0.083 | 0.596 +/- 0.088 | **0.628 +/- 0.186** | **0.524 +/- 0.205** | 0.072 +/- 0.007 | **1.215 +/- 0.167** |

## Delta Against Paper Column Best

| Metric | V9 delta | Result |
|---|---:|---|
| ROUGE | +0.014 | better |
| BERTScore | +0.078 | better |
| Reflectiveness | -0.072 | worse |
| Grounding | -0.024 | worse |
| Sentiment | +0.038 | better |
| Emotion | +0.064 | better |
| Intimacy AD | +0.012 | worse because lower is better |
| Empathy AD | -0.025 | better because lower is better |

V9 exceeds the paper's column-best value on five of eight metrics. The complete result does not support the earlier small-subset conclusion that all eight metrics exceed the paper.

## Judge Cache Reuse

- Exact prior V9 records reused: 387.
- Missing V9 records newly judged: 132.
- Existing Judge cells reused: 2,322.
- New Judge cells requested: 792.
- Duplicate labels were never averaged or overwritten; the fixed source-priority result was retained.
- Reuse identity checks required exact prediction text, exact ground truth, Judge protocol, and model name.

Server artifacts:

- Predictions: `/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-evidencefix-flash-5927bbf`
- Local metrics: `/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-local-metrics-v1`
- Complete Judge: `/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-judge-resume-v1`
