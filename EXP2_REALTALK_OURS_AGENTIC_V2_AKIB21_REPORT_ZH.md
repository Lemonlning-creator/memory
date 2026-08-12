# REALTALK Ours Agentic V2：Akib 21 条预检报告

## 状态

- 实现提交：`c750af1`
- 协议：`realtalk_task1_ours_agentic_v2`
- 模型：`qwen3-max-2026-01-23`
- 范围：Akib，Cb 前三 Session 中前 21 个合并目标消息
- 生成：21/21
- unresolved：0
- 519 条正式运行：未启动

生成目录：

`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-agentic-v2-akib21-c750af1`

GPT Judge 目录：

`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-gpt4omini-agentic-v2-akib21-c750af1`

## 八项结果（GPT 三项为已废弃诊断版）

| 指标 | Akib 21 | 论文 w/o fine-tune | 论文 w/ fine-tune |
|---|---:|---:|---:|
| ROUGE-L | 0.130 | 0.14 | 0.14 |
| BERTScore | 0.841 | 0.76 | 0.78 |
| Reflectiveness Accuracy | 0.524 | 0.62 | 0.77 |
| Grounding Accuracy | 0.571 | 0.40 | 0.62 |
| Sentiment Accuracy | 0.714 | 0.53 | 0.59 |
| Emotion Accuracy | 0.381 | 0.43 | 0.46 |
| Intimacy AD | 0.062 | 0.06 | 0.07 |
| Empathy AD | 2.429 | 1.80 | 1.24 |

以上仅是单人物、小样本诊断，论文行是十人物聚合，不能作为正式 Table 2 横向结论。进一步核对论文 Appendix C 后发现首轮 GPT Judge 错误地使用了跨 Session 累积历史，且 Reflectiveness/Grounding Prompt 省略了论文示例。因此表中三项 GPT 指标已标记为废弃诊断值，必须使用 `realtalk_appendix_c_within_session_v2` 重算；五项本地指标不受影响。

## 门槛判断

| 门槛 | 要求 | 结果 | 状态 |
|---|---:|---:|---|
| Reflectiveness | ≥ 0.60 | 0.524 | 未通过 |
| Grounding | ≥ 0.55 | 0.571 | 通过 |
| Empathy AD | ≤ 1.80 | 2.429 | 未通过 |

因此不允许启动 519 条。

## 诊断

最终 Decision 分布：

- Orientation：15 balanced、5 self-led、1 partner-adaptive；平均 `lambda_trace=0.479`。
- Primary Move：10 self-disclose、6 follow-up、3 answer、1 topic-shift、1 acknowledge。
- Question Mode：15 none、6 follow-up。

相较初版的 21/21 `mixed`、19/21 follow-up，策略单一化和追问偏置已经明显修复。Grounding 达到预检门槛，BERTScore、Sentiment 和 Intimacy 也较好。

剩余核心问题是对伙伴内容仍然适应过强。Judge 将真实 Akib 的 Reflective/Grounded 各判为 10/21，而候选分别为 18/21、17/21；真实消息平均 EPITOME 为 1.286，候选为 3.524。即使 Primary Move 已转向 self-disclose，生成仍常附带对伙伴的解释、确认或情绪回应，造成 Reflectiveness 与 Empathy 过高。

下一轮应继续只调整 Self/Decision Prompt 或其严格策略契约，重点降低日常闲聊中的 balanced 倾向和附加情绪回应。数据、Table 8 映射、Session 规则、Actor、模型和评价器保持不变。
