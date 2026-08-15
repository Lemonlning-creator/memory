# Exp2 REALTALK Ours 最终阶段报告

## 1. 实验目标

本实验只运行 Ours，不训练、不微调，也不重新运行论文基线。目标是在 REALTALK
Persona Simulation 协议下，让模型根据目标人物的 Ca 历史建立 Self Domain，并在 Cb
真实滚动历史中预测该人物下一条消息，再把 Ours 作为一行与论文 Table 2 并列。

本报告的当前主结果是 V9。V13 只属于未通过扩容门禁的后续诊断。

## 2. 数据与协议

- 数据来源：REALTALK 公开数据。
- 人物分配：严格使用论文 Table 8 的 10 组 Ca/Cb 映射。
- 历史范围：Ca 和 Cb 均取按时间排序的前三个连续 Session。
- 消息处理：连续同说话者气泡按论文规则无损合并。
- 测试规模：10 位目标人物，共 519 条重建目标消息。
- 历史：每个预测点读取当前答案之前的完整真实历史；不读取答案，不回灌生成文本。
- 聚合：先按人物计算，再报告 10 人 macro mean 与 population standard deviation。

`519` 是公开数据按协议重建所得，论文没有报告其官方准确样本数，因此这里称
“协议对齐重建”，不称“官方测试集逐条复现”。

## 3. Ours V9 实现

V9 协议名为 `realtalk_task1_ours_agentic_v8_low_specificity_continuity`，全部 Ours 阶段
使用 `deepseek-v4-flash`，关闭 thinking，不训练或微调。

流程为：

1. 用每位目标人物 Ca 前三 Session 建立固定 Self Domain。
2. 在 Cb 中按 Session 更新五层 User Domain。
3. 每个测试点基于完整历史、Self Domain 和可用 User Domain 生成 Situation、动态
   alignment/lambda trace 与唯一 Next Action。
4. Actor 根据完整历史和私有决策输出目标人物下一条纯文本消息。

V9 禁用 Omega、Future User State、历史压缩、历史裁剪、多候选策略、Verification、
重写和长度限制。所有 519 条均成功，生成 manifest 中记录 1,072 次模型调用、
5,550,793 prompt tokens 和 307,489 completion tokens。

## 4. 评价实现

八项指标与 REALTALK Table 2 对齐：

- ROUGE-L、BERTScore F1。
- Reflectiveness Accuracy、Grounding Accuracy。
- Sentiment Accuracy、Emotion Accuracy。
- Intimacy Absolute Difference、Empathy Absolute Difference。

Reflectiveness、Grounding 和 Empathy 使用论文 Appendix C 完整 Prompt，由
`gpt-4o-mini` 评价。519 条对应 3,114 个 Judge 单元，全部完成，零 unresolved。

BERTScore 使用 `bert-score 0.3.13`、`roberta-large` 第 17 层、English、`idf=false`、
`rescale_with_baseline=false`。论文明确使用 BERTScore，但没有公开完整评测代码、包版本
或 checkpoint revision，因此这是参数透明的标准重建。

## 5. Table 2 对比

上箭头越高越好，下箭头越低越好。

| 方法 | ROUGE ↑ | BERTScore ↑ | Reflect. ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy AD ↓ | Empathy AD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 w/o FT | 0.14 ± 0.04 | 0.76 ± 0.08 | 0.62 ± 0.13 | 0.40 ± 0.13 | 0.53 ± 0.22 | 0.43 ± 0.22 | **0.06 ± 0.01** | 1.80 ± 0.55 |
| 论文 w/ FT | 0.14 ± 0.05 | 0.78 ± 0.04 | **0.77 ± 0.09** | **0.62 ± 0.08** | 0.59 ± 0.18 | 0.46 ± 0.21 | 0.07 ± 0.01 | 1.24 ± 0.12 |
| **Ours V9** | **0.154 ± 0.033** | **0.858 ± 0.012** | 0.698 ± 0.083 | 0.596 ± 0.088 | **0.628 ± 0.186** | **0.524 ± 0.205** | 0.072 ± 0.007 | **1.215 ± 0.167** |

V9 超过论文逐列最优值 5/8 项：ROUGE、BERTScore、Sentiment、Emotion 和 Empathy AD。
仍落后的三项为：

- Reflectiveness：`-0.072`。
- Grounding：`-0.024`。
- Intimacy AD：高 `0.012`，该指标越低越好。

因此，当前结果支持“Ours 已有明显竞争力”，但不支持“八项全部超过论文”。

## 6. V13 优化结论

V13 引入渐进门禁 `6 → 18 → 30 → 60 → 120 → 519`，尝试强化条件化 Self Domain、
动态 lambda、交流义务和 Actor 风格控制。稳定版本 V13.4 与 V13.6 均完成 6/6、零
unresolved，但都未通过 Gate 1，未运行后续规模。

在固定困难 6 条上：

| 方法 | ROUGE ↑ | BERTScore ↑ | Reflect. ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy AD ↓ | Empathy AD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 匹配 V9 | 0.095 | 0.848 | 0.10 | 0.70 | 0.70 | 0.30 | 0.149 | 1.60 |
| V13.4 | 0.146 | 0.852 | 0.70 | 0.40 | 0.70 | 0.30 | 0.090 | 0.90 |
| V13.6 | 0.155 | 0.849 | 1.00 | 0.40 | 0.60 | 0.50 | 0.093 | 1.80 |

V13.4 改善反思、亲密度和共情误差，却显著损失 Grounding；V13.6 进一步过度反思并
恶化 Empathy。这个结果说明三项指标不是靠统一增加反思或追问即可同时提高。V13 当前
最平衡的诊断版本是 V13.4，但它不是可扩容主结果。

## 7. 当前状态与下一步

当前可提交的 Exp2 主结果已经完整：V9 519 条、八项指标、逐人物统计、模型和 Prompt
哈希、原始响应与检查点均存在。下一轮研究应从冻结 V9 出发，使用新的版本号和新目录，
先解决“是否回应伙伴的未履行交流义务”与“是否保持目标人物自身标签”之间的冲突；不得在
V13 固定 6 条上继续逐样本调参，也不得覆盖现有产物。

论文未披露 Persona Simulation 的相同基础模型运行时，因此主表必须保留
`protocol-aligned_not_runtime_identical` 说明。
