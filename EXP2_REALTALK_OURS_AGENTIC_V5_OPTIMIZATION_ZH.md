# REALTALK Ours Agentic V5 小规模优化记录

## 结论状态

- 当前代码提交：`47701ee`。
- 协议：`realtalk_task1_ours_agentic_v5_statistically_calibrated_thinking_decision`。
- 模型：`qwen3-max-2026-01-23`。
- Self Domain、User Domain、Generation 关闭 thinking；仅 Decision 开启 thinking。
- 本页结果均为提示词开发诊断，不是完整 10 人、519 条 Table 2 主结果。

## 修改内容

1. Self Domain 只描述 Ca 中反复出现的表层行为，不把理想化的“温暖、反思、支持、好奇”自动写成人格。
2. 将确定性统计作为 Response Actor 的优先校准信号：问句率、反思标记率、评价式开场率和典型消息长度。
3. 收紧互惠追问条件；低问句率人物只有在显著未完成话题下才追问。
4. 普通事实、偏好和热情表达不再自动解释为情绪或支持需求。
5. 删除无证据的通用赞美、验证和反思式说明，保留目标人物自身表达习惯。
6. 修复 Decision 阶段错误关闭 thinking 的实现偏移，并新增单元测试强制验证四阶段 thinking 配置。

## 验证

- 聚焦测试：15 项通过。
- 全测试套件：通过。
- Generation 输出无 JSON、reasoning 或内部策略泄漏。
- 合规开发集：Kevin、Vanessa 各前 5 条，共 10 条，10/10 成功，零 unresolved。
- Ground Truth 的 GPT Judge 标签固定复用旧运行同一 result ID 的参考判断；只重评新 Candidate，避免 Judge 随机变化影响配对结论。
- GPT 三项使用 REALTALK Appendix C 完整 Prompt 和 `gpt-4o-mini`。

## 同样本配对结果

| 版本 | ROUGE-L ↑ | BERTScore ↑ | Sentiment ↑ | Emotion ↑ | Intimacy AD ↓ | Reflectiveness ↑ | Grounding ↑ | Empathy AD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 8B | 0.125 | 0.850 | 0.70 | 0.60 | 0.080 | 0.90 | 0.60 | 1.20 |
| Max 第一轮校准 | 0.115 | 0.846 | 0.90 | 0.80 | 0.105 | 0.90 | 0.70 | 1.20 |
| V5 统计校准 + Decision thinking | **0.130** | 0.846 | 0.70 | **0.70** | **0.079** | **0.90** | 0.60 | **0.90** |

V5 在同样本上降低了过度共情，改善 Emotion 和 ROUGE-L；Grounding 没有异常抬高。10 条规模太小，不能据此声称超过论文完整 Table 2。

## 行为观察

- 10 条中 4 条含问句，4 条是显式 reciprocal question，平均 103.5 字符。
- Kevin 的问句与展开明显多于 Vanessa，符合两人的 Ca 表层差异。
- 仍存在无法仅凭历史命中的新事实，例如 Kevin 的真实下一条提到 Albania，而模型生成其他地点。这属于 next-message prediction 的内容不可预测性，不应通过读取答案或筛除样本解决。

## 运行风险

Decision thinking 的延迟远高于无思考模式，正常单条约 1–3 分钟，并观察到一次请求超过 20 分钟不返回。Paola/Nicolas 各 10 条 holdout 因该请求挂起而中止；随后恢复脚本误带 `--fresh`，覆盖了未完成 checkpoint。该 holdout 没有完成、没有评分、不得用于结论。

完整 519 条前需要先修复两项执行可靠性：

1. 给单次 SDK 调用增加可验证的硬超时或 watchdog，避免连接无限等待。
2. 将 fresh-run 和 resume-run 分成两个脚本；resume 脚本禁止携带 `--fresh`，并在启动前校验 checkpoint signature 与协议、模型、数据哈希一致。

完成可靠性修复后，应重新运行未参与调参的 Paola/Nicolas holdout；只用于泛化确认，不再继续按 holdout 调提示词。
