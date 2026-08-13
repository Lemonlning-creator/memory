# REALTALK Ours Agentic V10 优化锚点

## 冻结前提

- V9 的 519 条生成、既有本地指标和 GPT Judge 结果保持不变。
- 不修改 REALTALK Table 8 Ca/Cb 映射、前三个连续 Session、无损历史、模型或八项评价。
- 不按 Ground Truth 难度筛选人物、Session 或测试点。

## V9 诊断

十位人物、381 条渐进式诊断显示：Reflectiveness、Grounding 的总体正例率已经接近参考答案，主要问题是行为出现时机不匹配，而不是绝对数量不足。Actor 同时收到“answer 不可 return question”和“随后追加 reciprocal question”的冲突合同，也会使已规划问题丢失。

## V10 唯一机制变化

Decision Agent 在原有 Situation、动态 lambda 和唯一 Behavior Policy 内增加：

- `partner_continuation_need`：区分只回答、邀请伙伴展开、对称返回同一话题槽和无需继续。
- `self_revelation_need`：区分无需自我表达、只陈述状态、自然披露一个简短理由或感受。
- `self_revelation_mode`：Actor 对上述决定的严格执行字段。

同时消除 answer + reciprocal-question 的 Actor 合同冲突。V10 不要求整体增加追问、反思、温暖或亲密表达，也不把评价指标名称或目标分数输入生成模型。

## 首轮门槛

- 十位人物各 6 条，共 60 条；每个 Session 均匀取 2 条。
- 与相同 result_id 的 V9 输出比较，使用完全相同的本地指标和论文 Appendix C Judge。
- 首轮只判断机制方向；达标后才扩到更大样本。
