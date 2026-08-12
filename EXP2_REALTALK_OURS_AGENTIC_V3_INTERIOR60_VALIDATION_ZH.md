# REALTALK Ours Agentic V3 独立验证报告

## 1. 验证目的

本次验证用于检查此前在首尾 60 条开发样本上形成的通用 Prompt 改动能否泛化。

- 没有针对 sample ID、人物姓名、Ground Truth 文本或 Judge 标签写专门规则。
- 但首尾 60 条已经被用于错误分析和通用机制调优，因此只能称为开发集。
- 本次固定选择每位人物三个 Cb Session 中各两个中间位置，共 60 条。
- 新旧样本以 `(speaker, sample_id)` 为键严格零交集。
- 10 位 Table 8 目标人物全部覆盖，每人 6 条；不按难度或标签过滤。

## 2. 冻结身份

- Ours 模型：`qwen3-max-2026-01-23`
- 生成提交：`cd9d7fe4b8b2c6e0958913c089762af2a8208a7c`
- 预测记录：60，零 unresolved
- predictions SHA256：`d3815623c3b9083095e7bc064a69b8f02a2ace567c47cac355a62fb266930a2f`
- 选择器不读取消息文本、Ground Truth 或 Judge 标签。

## 3. 论文指标对齐

主值采用与 Table 2 相同的逐人物均值后再对 10 人求 macro mean 和 population std。

- Reflectiveness、Grounding、Empathy：`gpt-4o-mini`，论文 Appendix C 完整 Prompt，共 360/360 判断，零错误。
- Emotion、Sentiment、Intimacy：论文指定的 CardiffNLP Twitter RoBERTa 系列，固定 checkpoint revision。
- Lexical：ROUGE-L F1。
- Semantic：BERTScore F1，固定标准 English `roberta-large` 配置。

论文没有公开 Persona Simulation 完整评测代码，也没有披露 ROUGE/BERTScore 的软件版本与完整参数。因此前六项的定义和模型/Prompt 可直接依据论文固定；ROUGE 与 BERTScore 属于明确记录配置的协议重建值。

## 4. 八项结果

| 方法 | Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 w/o fine-tune | 0.14 ± 0.04 | 0.76 ± 0.08 | 0.62 ± 0.13 | 0.40 ± 0.13 | 0.53 ± 0.22 | 0.43 ± 0.22 | 0.06 ± 0.01 | 1.80 ± 0.55 |
| 论文 w/ fine-tune | 0.14 ± 0.05 | 0.78 ± 0.04 | 0.77 ± 0.09 | 0.62 ± 0.08 | 0.59 ± 0.18 | 0.46 ± 0.21 | 0.07 ± 0.01 | 1.24 ± 0.12 |
| Ours V3，interior-60 | **0.10 ± 0.04** | **0.84 ± 0.01** | **0.53 ± 0.16** | **0.55 ± 0.20** | **0.53 ± 0.23** | **0.47 ± 0.29** | **0.07 ± 0.04** | **1.67 ± 0.45** |

## 5. 判断

本次结果不支持“当前 Ours 已稳定超过论文微调行”。

- 明显较好：Semantic 高于论文两行；Emotion 与论文微调行相当。
- 居中：Grounding 高于无微调行但低于微调行；Sentiment 与无微调行相当；Intimacy 接近论文微调行。
- 明显不足：Reflectiveness 低于论文两行；Empathy 好于无微调行但明显落后微调行；Lexical 低于论文两行。

此前首尾开发集的 Reflectiveness `0.83`、Grounding `0.72`、Empathy AD `1.17` 没有在本次独立位置验证中复现。不存在样本硬编码，但通用 Prompt 已根据开发集错误反复调整，出现了开发集适配和泛化不足。

本验证仍来自同一批 10 人、同一 Cb 前三 Session，只是目标位置独立，因此它比开发集可靠，但不是外部数据集验证。下一轮方法修改应继续只使用开发集；interior-60 必须冻结，不再参与调参，后续最终判断应使用完整 519 条或另建第三个未见位置集合。

## 6. 服务器产物

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-agentic-v3-interior60-cd9d7fe`
- 本地五项：生成目录下 `offline-local-metrics-bdbb484/`
- GPT 三项：生成目录下 `judge-appendix-c-v3/`
