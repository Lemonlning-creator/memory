# REALTALK Ours V11 Soft Actor 实验锚点

## 目标

V11 只重放 V9 的 Response Actor，用于验证 V9 的硬动作合同是否限制了自然人物表达。
Self Domain、User Domain、Situation、lambda、Behavior Policy、数据与 Judge Prompt 全部冻结。

## 唯一方法改动

- 先完成 V9 `primary_move`。
- 最多允许一个同话题的简短理由、感受或自然反应。
- 只有 V9 明确设置 `continuation_move=reciprocal-question` 时才允许追加互问。
- 禁止第二话题、通用安慰、治疗式分析、额外建议和采访式追问。
- `reflective_marker_rate` 是描述性风格证据，不再作为禁止自然理由或感受的硬上限。

## 固定比较

- `first5_30`：Table 8 前五人，每个 Session 按 full-span 取首尾两条，共 30 条。
- `second5_30`：Table 8 后五人使用相同规则，共 30 条。
- `all10_60`：前后两组原样合并，不重新生成前五人。
- V9 基线从完整 V9 产物按同一 `result_id` 清单抽取，不重新生成。
- GPT Judge 复用 Appendix C Prompt；Ground Truth 每条只判断一次，供 V9/V11 共享。

## 继续标准

`first5_30` 必须 30/30 成功、零 unresolved、无结构或身份泄漏，并满足：

- Reflectiveness 相对 V9 不下降超过 0.05；
- Grounding 相对 V9 不下降超过 0.05；
- Empathy AD 相对 V9 不恶化超过 0.20。

通过后才生成 `second5_30` 并组成 `all10_60`。若 Actor-only 方向失败，停止本轮，不针对
这些样本继续调 Prompt。

## 报告边界

30/60 条结果是匹配子集诊断，不是完整 Table 2 主结果。报告固定并列论文 w/o FT、论文
w/ FT、V9 和 V11，并完整披露八项指标与 speaker-macro 聚合。
