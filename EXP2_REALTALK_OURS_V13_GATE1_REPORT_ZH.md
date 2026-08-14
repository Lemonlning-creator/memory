# REALTALK Ours V13 Gate 1 报告

## 结论

V13 渐进式管线、Schema、检查点、Self Domain 条件统计、动态 lambda、问题权限、
Actor 和配对 Judge 已实现并在服务器跑通。最终稳定版本均达到 `6/6`、零 unresolved，
但没有版本同时改善 Reflectiveness、Grounding 和 Intimacy，因此按计划停止在 Gate 1，
未运行 18/30/60/120/519。

当前最平衡的诊断版本是 V13.4；V13.6 的 Actor 风格约束把 Reflectiveness 推得过高，
同时恶化 Empathy，不应继续使用。

## 固定困难集比较

该 6 条集合有意包含 V9 的 Reflectiveness、Grounding 和高 Intimacy 错误，覆盖 5 位
人物与全部三个 Session。它不是论文的随机全量测试集，不能用其绝对均值声称超过论文。

为减少 Judge 波动，V13.4 与 V13.6 复用完全相同的 18 个 Ground Truth 标签和 18 个
V9 标签，只新判断各版本的 18 个候选单元。

| 方法 | ROUGE ↑ | BERTScore ↑ | Reflect. ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy AD ↓ | Empathy AD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 匹配 V9 | 0.095 | 0.848 | 0.10 | 0.70 | 0.70 | 0.30 | 0.149 | 1.60 |
| V13.4 | 0.146 | 0.852 | 0.70 | 0.40 | 0.70 | 0.30 | 0.090 | 0.90 |
| V13.6 | 0.155 | 0.849 | 1.00 | 0.40 | 0.60 | 0.50 | 0.093 | 1.80 |

V13.4 明显改善 Reflectiveness、Intimacy 和 Empathy，但 Grounding 比匹配 V9 低
`0.30`。V13.6 保留 Grounding 下降，并使 Empathy AD 从 `0.90` 恶化到 `1.80`。

## 版本诊断

- V13.0：完整跑通；Reflectiveness 和 Empathy 改善，Grounding 下降。
- V13.1：新增未履行问题证据，但冗余动作枚举造成结构失败，`3/6`。
- V13.2：简化动作 Schema，`5/6`；剩余失败为无义务时多填来源 ID。
- V13.3：确定性清理来源并验证问题证据，`6/6`；Grounding 仍下降。
- V13.4：使用 Self 条件回问率门控辅助问题，修复 Akib 错误并保留 Fahim 正确回问；
  是当前最平衡版本，但未达到扩容条件。
- V13.5：过强问题索引指令制造错误义务，且无关 Self 重生成增加随机性，`3/6`。
- V13.6：撤回过强义务指令、固定复用 V13.4 Self、只保留近期 Cb 风格约束；
  Actor 更反思但过度参与，Empathy 退化，停止。

## BERTScore 说明

BERTScore 是 REALTALK Table 2 明确要求的论文原指标，不是 Ours 新增指标。论文未公开
Persona Simulation 的完整评测代码，也未锁定 BERTScore 包版本、checkpoint revision
和全部参数，因此本实验使用并披露标准英文重建配置：`bert-score 0.3.13`、
`roberta-large`、17 层、`idf=false`、`rescale_with_baseline=false`。这属于指标对齐、
实现参数透明的协议重建，不能称为论文官方逐代码复现。

## 服务器产物

- V13.4：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v13-4-progressive-v1-3dbcc4c`
- V13.6：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v13-6-progressive-v1-a358b28`
- 两个目录均保存 predictions、Decision/lambda、Self/User Domain、Prompt/Schema/源码哈希、
  原始响应、配对 Judge、本地指标、重试和零 unresolved 证明。
