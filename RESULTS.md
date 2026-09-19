# 完整结果

## 8 项对比（论文口径：10 位说话者宏平均，2 位小数）

| Method | Lexical | Semantic | Reflective | Grounding | Sentiment | Emotion | Intimacy | Empathy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o fine-tune (paper) | 0.14 | 0.76 | 0.62 | 0.40 | 0.53 | 0.43 | 0.06 | 1.80 |
| w/ fine-tune (paper) | 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.07 | 1.24 |
| paper column-best | 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.06 | 1.24 |
| **Ours (V9 + Rule A+B) r2** | 0.14 | 0.86 | 0.77 | 0.62 | 0.65 | 0.52 | 0.07 | 1.08 |

## 精确值

- Lexical: 0.1422（样本标准差 n-1 = 0.0405）
- Semantic: 0.8552（样本标准差 n-1 = 0.0132）
- Reflective: 0.7729（样本标准差 n-1 = 0.0680）
- Grounding: 0.6204（样本标准差 n-1 = 0.0998）
- Sentiment: 0.6523（样本标准差 n-1 = 0.1822）
- Emotion: 0.5199（样本标准差 n-1 = 0.2239）
- Intimacy: 0.0747（样本标准差 n-1 = 0.0099）
- Empathy: 1.0842（样本标准差 n-1 = 0.1510）

## 逐项判定

- Lexical: 0.1422 vs 目标 0.14 → 达标
- Semantic: 0.8552 vs 目标 0.78 → 达标
- Reflective: 0.7729 vs 目标 0.77 → 达标
- Grounding: 0.6204 vs 目标 0.62 → 达标
- Sentiment: 0.6523 vs 目标 0.59 → 达标
- Emotion: 0.5199 vs 目标 0.46 → 达标
- Intimacy: 0.0747 vs 目标 0.06 → 未达标
- Empathy: 1.0842 vs 目标 1.24 → 达标

**合计 7/8**

## 逐说话者

见 `results/per_speaker.json`。
