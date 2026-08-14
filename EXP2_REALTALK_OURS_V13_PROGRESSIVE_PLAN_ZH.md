# REALTALK Ours V13 渐进式优化锚点

## 1. 固定目标

V13 只优化 REALTALK Persona Simulation 的 Ours 一行。模型固定为
`deepseek-v4-flash`，不训练、不运行论文基线，不改变 REALTALK Table 8 的
Ca/Cb 分配、每段对话的前三个连续 Session、519 条测试样本、完整真实历史和
论文 Appendix C Judge。

V13 从冻结的 V9 数据与中间产物重新运行增强后的 Self Domain、Decision 和
Actor。五层 User Domain 的结构与更新边界保持不变。V9-V12 的代码与产物不覆盖。

## 2. 方法改动

### Self Domain

- 从 Ca 前三个连续 Session 的完整原文生成，每位人物一次。
- 区分可跨伙伴迁移的稳定行为与可能只适用于 Ca 伙伴的行为。
- 建模开场、直接提问、伙伴自我披露、普通陈述和结束场景下的条件行为。
- 同时保存程序确定性计算的条件化回问率、第一人称率、反思标记率、消息长度和
  合并气泡规模。
- Cb 当前真实历史与 Self Domain 冲突时，以 Cb 历史为准。

### User Domain

- 保持 `Core / Regulation / Cognition / Identity / Behavior` 五层。
- Session 1 开始为空；完成 Session 1 后更新一次供 Session 2 使用；完成 Session 2
  后更新一次供 Session 3 使用。
- evidence 只能引用已经完成的伙伴真实消息；Decision 最多激活两条相关事实。

### Decision

- 识别当前交流义务，生成一个主要动作和最多一个同槽位辅助动作。
- 同次输出动态 `lambda_trace` 与 `self-led / balanced / partner-adaptive`。
- `lambda_trace` 范围分别为 `0.00-0.35 / 0.36-0.70 / 0.71-1.00`，并必须与策略一致。
- 明确问题权限、反思深度、关系语气和消息结构。
- 对方直接提问时不得退化为 `self-led`；无历史开场才固定为 `self-led`。

### Actor

- 输入完整真实滚动历史、精简 Self View、Situation 和唯一 Behavior Policy。
- 完成主要动作，只执行被 Decision 明确许可的辅助动作与问题。
- 不读取完整 User Domain、lambda 理由、Judge 指标或 Ground Truth。
- 不启用候选搜索、Verification、重写或模型输出回灌。

## 3. 固定渐进门

样本清单从 V9 的 519 条预测、完整 Judge 和本地指标确定性构造；所有大门包含小门。

| Gate | 数量 | 用途与停止条件 |
|---|---:|---|
| 1 | 6 | 2 条 Reflectiveness 错误、2 条 Grounding 错误、2 条高 Intimacy AD；只验证链路和方向 |
| 2 | 18 | 增加三类错误与正确对照；仍允许升级 Prompt 版本 |
| 3 | 30 | 10 人 × 3 Session；通过后冻结 Prompt、Schema 和模型配置 |
| 4 | 60 | 每人每 Session 两条；失败则停止该版本 |
| 5 | 120 | 每人每 Session 四条；要求目标指标相对 V9 稳定改善 |
| 6 | 519 | 仅在 Gate 5 通过后运行其余样本 |

同一 Prompt 版本扩容时只调用新增样本。Prompt 或 Schema 一旦改变，协议升级为新的
V13.x，并从 6 条重新开始。Gate 1 的均值不用于宣称最终性能。

## 4. 完整结果门槛

完整 519 条最终需要严格超过论文逐列最优：ROUGE `>0.14`、BERTScore `>0.78`、
Reflectiveness `>0.77`、Grounding `>0.62`、Sentiment `>0.59`、Emotion `>0.46`、
Intimacy AD `<0.06`、Empathy AD `<1.24`。

## 5. 审计要求

- 所有阶段关闭 thinking，模型精确 ID 为 `deepseek-v4-flash`。
- 结构化阶段使用严格 Schema，最多三次格式重试；正式 Gate 要求零 unresolved。
- 保存嵌套样本 ID、Prompt/Schema/输入文件 SHA256、原始响应、token、重试和检查点。
- V9/V13 配对 Judge 共享同一 Ground Truth 标签并使用论文 Appendix C Prompt。
- Gate 1 完成后先人工检查六条 Decision、lambda、Actor 与 V9/GT，再决定是否进入 18 条。
