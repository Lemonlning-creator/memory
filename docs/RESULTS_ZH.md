# V9 冻结结果

来源是服务器已完成的三个独立阶段，不是重新抽测；519 条、10 人、3,114 个 Judge 判断，使用 speaker macro mean。原始 JSON 汇总在 `reports/frozen/`，完整逐样本数据见结果包。

| Method | ROUGE-L ↑ | BERTScore ↑ | Reflect. ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy AD ↓ | Empathy AD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper w/o FT | 0.14 | 0.76 | 0.62 | 0.40 | 0.53 | 0.43 | 0.06 | 1.80 |
| Paper w/ FT | 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.07 | 1.24 |
| Ours V9 | 0.153925 | 0.857915 | 0.697900 | 0.595898 | 0.627794 | 0.524401 | 0.072131 | 1.215273 |

相对论文**逐列最优**，五项更好，Reflectiveness、Grounding、Intimacy AD 三项仍低于最优。当前用户选择 V9 为最佳保留版本，不等于统计证明它普遍优于所有后续版本，更不等于八项都已超过论文。置信区间和显著性不能仅由这一张均值表推出。

| 指标 | V9 population std |
|---|---:|
| ROUGE-L | 0.032585 |
| BERTScore | 0.011965 |
| Reflectiveness | 0.083240 |
| Grounding | 0.088370 |
| Sentiment | 0.186125 |
| Emotion | 0.205479 |
| Intimacy AD | 0.006526 |
| Empathy AD | 0.167253 |

## 指标含义与实现

- ROUGE-L：本版 `src/metrics.py` 的词级 LCS F1，不把它宣称为作者逐字一致的计算脚本。
- BERTScore：`roberta-large`、17 层、English、F1、idf=false、rescale_with_baseline=false。论文使用该类指标，但未披露完整包版本/权重 revision；这是本版固定实现。
- Emotion / Sentiment：对真值与预测分别用固定 Cardiff 模型取 top-1，标签相同计 1。
- Intimacy AD：固定 Cardiff 模型对两者预测得分后取绝对差；越低越接近原人物，不是越亲密越好。
- Reflectiveness / Grounding：Appendix C Prompt 由 gpt-4o-mini 分别判断真值与预测，比较布尔标签是否一致；不是一味增加反思或提问。
- Empathy AD：Emotional Reaction、Interpretation、Exploration 各 0–2，总分 0–6；比较预测和真值总分绝对差，越低越好。

先按人物平均，再对 10 人等权平均并计算 population std。不要用 message micro 平均替换主表，人物消息数不同。

## 限制必须一起交接

1. 519 是公开数据重建计数；生成累计三个 session 的历史、Judge 仅当前 session 历史，这是冻结实现，不声称作者未公开细节完全一致。
2. 模型是 DeepSeek Flash API，论文 Persona Simulation 基础模型未完全披露；Ours 未微调，论文包含微调行。
3. 这些 Cb 已参与探索诊断；报告应写 protocol-aligned exploratory comparison，不能包装成完全未见独立验证。
4. 原 V9 单主动作与 self-led 倾向、λ 的实际有限作用都保留，归档不顺手优化。
5. API 无固定 seed；同代码重跑不保证得到同样文本/分数。固定已有预测与 Judge 标签才是精确结果复核。
6. 完整结果包只收录 V9，不混入 v4-pro 的 27 条、不收录失败的 v4.1 尝试。

## 原始位置

```text
/amax/xidian_ty/Ly/personaemp-exp2/runs/
  realtalk-ours-v9-full519-evidencefix-flash-5927bbf/
  realtalk-ours-v9-full519-local-metrics-v1/
  realtalk-ours-v9-full519-judge-resume-v1/
```

这些原目录及 raw audit 不修改。干净包统一收录所需文件，见 [包内 README](RESULT_PACKAGE_README.md)。
