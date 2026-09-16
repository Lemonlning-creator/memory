# V9 方法与输入输出

本页描述冻结实现，不把后续优化方案写成本版能力。源码为 `src/experiments/realtalk_ours.py`，严格结构见 `realtalk_ours_schemas.py`；完整英文 Prompt 已导出至 `contracts/`。

## 任务身份

模型要扮演 Table 8 指定的目标人物。Self Domain 是这个人物；User Domain 是当前 Cb 伙伴，不能对调。它不是理想客服或共情治疗助手。目标是自然预测真实人物下一条话语，标签答案只用于评价。

## 调用链

```text
Ca 前 3 个 session 的完整双方文本
  -> 程序计算目标人物行为统计
  -> Self Domain（每人一次，之后固定）

Cb session 1：User Domain 为空
Cb session 1 完成：完整 session 1 + 空画像 -> 更新伙伴 User Domain
Cb session 2 完成：完整 session 2 + 上次画像 -> 再更新一次
Cb session 3 完成后不再更新

每个测试点：
完整真实 Cb 前缀 + Self Domain + 当前 User Domain
  -> Decision（Current Situation + λ + 唯一 Next Action）
  -> Actor（真实历史 + Self 行为视图 + Situation + Next Action）
  -> 目标人物文本，与隐藏真值配对保存
```

正常完整运行的逻辑调用基数：10 次 Self、20 次 User 更新、519 次 Decision、519 次 Actor，共 1,068 次；另有端点预检、网络重试及最多三次逻辑尝试。不能把基数当作实际计费次数。Judge 另算。

## Self Domain

输入是 Ca 前三段完整会话，两方消息提供语境，但人物事实应来自目标人物自己的发言。不是使用师姐预写默认 Persona，也不使用 Cb 答案。

| 字段 | 用途 |
|---|---|
| identity_context | 自我描述、生活背景、关系、长期兴趣 |
| communication_signature | 语气、词汇、信息密度、消息规模、表达模式 |
| interaction_policy_prior | 主动性、自我披露、追问、话题延续/切换、建议和情绪回应倾向 |
| affective_social_signature | 情绪、情感、内省、跟进、温暖和亲密表达习惯 |
| boundaries_and_uncertainty | 稳定边界与不确定属性 |
| observable_statistics | 程序计算且要求原样复制的统计 |

统计包括目标消息数、平均/中位字符数、问句率、第一人称率、反思标记率、评价式开头率、合并气泡数量中位数。它们是脚本代理统计，不是 Judge 标签或心理学测量。

最终 Actor **不接收完整身份事实**：`_behavioral_self_domain` 只给表达/互动/情感行为部分与统计校准，减少把 Ca 旧地点、旧活动挪成 Cb 当前事实的问题。Decision 仍读完整 Self Domain。

## User Domain

五层固定为 `core / regulation / cognition / identity / behavior`。每条事实为：

```json
{"value": "...", "confidence": "low|medium|high", "evidence_ids": ["session_1:turn_3"]}
```

还包括严格 `update_summary`。只接受已完成 session 中的伙伴证据；唯一可解析的简写 turn ID 会规范化，歧义或错误证据会报错。Decision 最多激活两条相关事实，且必须逐字来自当前画像白名单。完整 User Domain 不直接进入 Actor。

## State、λ 与 Behavior Policy

同一次 Decision 调用输出：

- `situation`：话题、伙伴动作、明确情感、支持请求、开放问题、未完成话题及缺失信息等。
- `relevant_user_domain`：0–2 条当前相关伙伴画像。
- `alignment`：`orientation`、`lambda_trace`、`decision_basis`。
- `next_action`：唯一主要动作、内容方向、自我表达、伙伴适配、语气、规模、问题模式、可选互惠问题等。

V9 将 λ 定义为“为了伙伴需要而偏离本人常态的程度”；普通日常交流提示为 self-led、λ 在 0–0.25。它是 LLM 同次决策的可审计数值，不是公式混合器，也没有确定性保证数值变化必然改变输出。V9 确实偏 self-led；本次不暗中修正。

主要动作有 open / self-disclose / answer / acknowledge / follow-up / topic-shift。仅符合既定条件时允许附加 reciprocal-question。V9 保留这一相对严格的单主动作策略，不加入 V11/V15 的放宽或多气泡控制。

## Actor

System Prompt 原文：

```text
You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private next-action decision naturally.
Output only the message, not the speaker name.
```

User Prompt 的完整内容见 `contracts/generation_user_template.txt`；包含动作合同、低具体度新自我表达、人物统计校准等规则。没有统一 2–4 句限制，但确有统计驱动的长度引导。不能简写为“完全无约束”。

其中 `{action_contract}` 由原代码 `_action_contract(primary_move, continuation_move)` 生成，并非另一次模型调用；完整规则以该函数为准。部分分支还有一句式/简短主回复约束。本归档保留这些限制，不把模板占位符误当成实际完整输入。

## 固定配置

| 阶段 | 模型 | temperature | top_p | max_tokens | thinking |
|---|---|---:|---:|---:|---|
| Self | deepseek-v4-flash | 0.2 | 0.9 | 3000 | false |
| User | deepseek-v4-flash | 0.2 | 0.9 | 3000 | false |
| Decision | deepseek-v4-flash | 0.2 | 0.9 | 1600 | false |
| Actor | deepseek-v4-flash | 0.6 | 0.9 | 300 | false |

不设置 seed。不启用 Omega、Future State、候选搜索或语义 Verification。没有摘要、检索替代或历史裁剪。格式校验失败可重试，不等于语义优化重写。

结构化阶段发送 JSON Schema 并本地校验；端点声称支持 Schema 不代表永不截断或永不出错。原实现逻辑尝试上限为 3，底层网络尝试另计。保留检查点、原始输出与 unresolved；零 unresolved 才可视为生成完成。
