# Exp2: REALTALK Table 2 + Ours 最终实验协议

> **执行状态说明（2026-08-15）**：本文保留了实验设计形成过程，其中第 13 节的
> `qwen3-8b` 是早期预案，不是最终全量运行配置。当前唯一完成 10 人、519 条生成及
> 八项评价的权威结果是 Ours V9，模型为 `deepseek-v4-flash`。实际配置与结果请以
> `EXP2_REALTALK_FINAL_REPORT_ZH.md`、`EXP2_REALTALK_V9_FULL519_RESULTS_ZH.md`
> 和对应运行 manifest 为准。

状态：生成协议冻结稿 v2；GPT 三项评价端点待补  
方法锚点：`OURS_METHOD_ANCHOR_ZH.md`  
事实审计：`EXP2_REALTALK_INPUT_OUTPUT_AUDIT_ZH.md`  
实施锚点：`EXP2_REALTALK_OURS_QWEN3_8B_PLAN_ZH.md`

## 0. 一句话说明

对 REALTALK 的每一位真实参与者，使用 Table 8 指定的 Ca 对话建立该参与者的 Self Domain，再让模型在 Table 8 指定的 Cb 对话中扮演该参与者；模型每次只看到目标消息之前的真实历史，滚动理解测试伙伴并生成目标参与者下一条消息，最后按 REALTALK Table 2 的八项指标与真实消息比较。

## 1. 最通俗的完整例子

以 Emi 为目标人物：

```text
Ca：Emi 与 Paola 的对话
Cb：Emi 与 elise 的对话
```

实验先从 Ca 前 3 个 Session 中建立 Emi Self Domain。进入 Cb 后，模型固定扮演 Emi；elise 是当前被回应和被理解的人。

假设 Cb 的真实对话为：

```text
elise: I had another awful day at work.
Emi: Oh no, what happened this time?
elise: My manager rejected the proposal again.
Emi: That's awful. You worked so hard on it.
```

预测第一条 Emi 消息时：

```text
可见：elise 的第一条消息
不可见：真实 Emi 目标消息及其之后的内容
输出：模型以 Emi 身份生成的一条消息
真值：Oh no, what happened this time?
```

预测第二条 Emi 消息时：

```text
可见：截至第二条 Emi 消息前的全部真实历史
不可见：第二条真实 Emi 消息及其之后的内容
输出：模型再次以 Emi 身份生成
真值：That's awful. You worked so hard on it.
```

上一条模型生成结果不进入下一条样本。下一条样本始终使用真实历史，避免误差累积并与论文 `H_t={M_1,...,M_(t-1)}` 保持一致。

## 2. 数据和十组分配

数据固定为官方 REALTALK 仓库 commit：

```text
b903e06a9770bf4e5fe9018c3e132889666d3b4a
```

本地 10 个预处理 JSON 已确认与官方对应文件字节一致。

每位参与者有两条不同伙伴的长对话。论文 Table 8 对每位参与者分别指定一条 Ca 和一条 Cb：

| 目标说话者 / Ours Self | Ca：建立目标 Self | Cb：正式测试 | Cb 伙伴 / Ours User | Ca 前 3 Session Self turns | Cb 前 3 Session targets |
|---|---|---|---|---:|---:|
| Emi | Chat 4 Emi-Paola | Chat 1 Emi-elise | elise | 20 | 37 |
| Nicolas | Chat 5 Nicolas-Nebraas | Chat 6 Vanessa-Nicolas | Vanessa | 73 | 117 |
| Kevin | Chat 3 Kevin-Paola | Chat 2 Kevin-elise | elise | 24 | 25 |
| Akib | Chat 9 Fahim-Akib | Chat 8 Akib-Muhhamed | Muhhamed | 40 | 37 |
| Muhhamed | Chat 10 Fahim-Muhhamed | Chat 8 Akib-Muhhamed | Akib | 24 | 37 |
| Nebraas | Chat 5 Nicolas-Nebraas | Chat 7 Nebraas-Vanessa | Vanessa | 72 | 51 |
| Paola | Chat 4 Emi-Paola | Chat 3 Kevin-Paola | Kevin | 21 | 23 |
| Vanessa | Chat 7 Nebraas-Vanessa | Chat 6 Vanessa-Nicolas | Nicolas | 51 | 116 |
| elise | Chat 2 Kevin-elise | Chat 1 Emi-elise | Emi | 26 | 36 |
| Fahim Khan | Chat 10 Fahim-Muhhamed | Chat 9 Fahim-Akib | Akib | 26 | 40 |

总测试目标：

```text
37 + 117 + 25 + 37 + 37 + 51 + 23 + 116 + 36 + 40 = 519
```

这 519 是公开数据按 Table 8、前三 Session 和论文的连续同说话者消息合并规则重建得到的数量。论文没有公布 Table 2 的准确样本数，不能把 519 称为论文官方报告数字。

## 3. Ca 和 Cb 分别做什么

### 3.1 论文原流程

对目标说话者 `S`：

- `w/o fine-tune`：不使用 Ca，通用模型直接在 Cb 测试；
- `w/ fine-tune`：用 S 在 Ca 中的消息单独微调，然后模型权重固定，在 Cb 测试；
- Table 2 只报告 Cb 测试结果；
- Ca 不产生 Table 2 分数。

Ca/Cb 是 speaker-specific，而不是文件的全局标签。同一文件可以是一个人的 Ca、另一个人的 Cb。

### 3.2 Ours 对应流程

Ours 不训练模型：

```text
S 的 Ca 前 3 Session
-> 显式生成 S Self Domain
-> 进入 Cb 后固定
```

它与论文 `w/ fine-tune` 使用相同来源、近似相同 3-Session 目标人物信息预算，但使用方式不同：

```text
论文 fine-tune：Ca -> 模型权重
Ours：Ca -> 显式 Self Domain
```

正式报告必须写为 `explicit modeling without fine-tuning`，不能把 Ours 叫作论文微调复现。

## 4. 消息预处理和测试样本

### 4.1 合并规则

同一人在同一 Session 中连续发送的消息气泡合并为一条语义消息；不跨 Session 合并。这对应论文第 4.1 节“在分析前合并连续同说话者消息”的明确规则。

### 4.2 Ca 范围

使用 Ca 按时间排序的前 3 个 Session 建立目标 Self Domain。

### 4.3 Cb 范围

使用 Cb 按时间排序的前 3 个 Session。对其中每一条属于目标说话者 S 的合并消息 `M_t` 构造一个测试样本。

### 4.4 每条样本的历史

```text
H_t = Cb 选定片段中严格位于 M_t 之前的全部真实合并消息
```

- Session 之间继承真实历史；
- 当前 Session 内也包含目标位置以前的真实消息；
- 不读取 `M_t`；
- 不读取未来消息；
- 不把模型上一次生成结果回灌；
- 不在 Cb 更新模型权重；
- 不用摘要替换论文原本可见的完整历史，除非模型上下文上限迫使截断；发生截断时只能从最旧完整 turn 开始删除并记录。

## 5. Ours 的角色定义

对每个 speaker-specific 实验：

```text
目标说话者 S = 模型当前扮演的角色 = Self Domain 主体
Cb 对话伙伴 P = 被回应的人 = User Domain 主体
S 的真实 M_t = Ground Truth
```

目标说话者身份是硬约束：高 `lambda_t` 只允许 S 更适配 P，不能让 S 变成或模仿 P。

## 6. Self Domain 的建立

### 6.1 输入

```json
{
  "target_speaker": "S",
  "source": "S 的 Ca 前 3 个 Session",
  "conversation": "包含双方对话作为语境",
  "evidence_binding": "只有 S 自己的发言可作为 S 属性证据"
}
```

### 6.2 输出

```json
{
  "identity": {},
  "persona": {
    "personality": "",
    "tone": "",
    "expression_patterns": []
  },
  "behavior_policy_prior": {
    "interaction_principles": [],
    "emotional_response_style": "",
    "guidance_style": "",
    "initiative": "low|medium|high"
  },
  "hard_constraints": []
}
```

Self Domain 在 Cb 测试前生成一次并缓存，测试中不更新。Ca 伙伴发言只能帮助理解 S 的响应语境，不能被复制成 S 的身份或经历。

## 7. User Domain 的建立和变化

Cb 开始时，不额外读取伙伴 P 的另一条对话。User Domain 从空或无证据状态开始，只使用当前目标前已经出现在 `H_t` 中的 P 发言滚动更新。

### 7.1 每次更新输入

```json
{
  "previous_user_domain": {},
  "new_observed_partner_turns": [],
  "context": "这些 P 发言出现时的真实 Cb 历史"
}
```

### 7.2 输出

```json
{
  "core": {},
  "regulation": {},
  "cognition": {},
  "identity": {},
  "behavior": {},
  "update_summary": {}
}
```

只允许 P 自己的发言支持 P 的属性。早期 User Domain 稀疏是合法冷启动；不能读取未来 Cb 消息补齐。

技术上可以读取 P 自己的 Ca 预初始化 User Domain，但那会比论文当前目标模型多使用一份跨对话数据，只能作为 `enhanced-data` 附加分析，不进入直接拼接 Table 2 的主 Ours 行。

## 8. 每个测试目标的三次逻辑调用

### 调用 1：User Domain Update

输入：旧 User Domain + 自上次目标后新观察到的 P 发言。  
输出：更新后的 P 五层 User Domain。  
若没有新 P 发言，则复用缓存，不发生 API 调用。

### 调用 2：State + Adaptive Alignment + Policy

输入：

```json
{
  "history": "H_t",
  "current_query": "H_t 中最新的 P 发言；没有则为空",
  "self_domain": "S Self Domain",
  "user_domain": "P User Domain",
  "previous_user_state": {}
}
```

输出：

```json
{
  "user_state": {
    "emotion": "",
    "emotional_intensity": "low|medium|high",
    "intent": "",
    "main_need": "",
    "interaction_expectation": "",
    "evidence": [],
    "uncertainty": "low|medium|high"
  },
  "alignment": {
    "lambda": 0.0,
    "orientation": "self-dominant|self-leaning|user-leaning|strongly-user-oriented",
    "basis": {},
    "policy_effect": ""
  },
  "behavior_policy": {
    "response_objective": "",
    "perspective_taking": "",
    "emotion_alignment": "",
    "personalization": "",
    "self_domain_expression": "",
    "directness": "low|medium|high",
    "guidance": "none|light|direct",
    "question_policy": "none|optional|necessary",
    "tone": "",
    "avoid": []
  }
}
```

同一次调用内部按顺序先识别 User State，再判断显式、自适应 `lambda_t`，最后生成一个且仅一个 Behavior Policy。不生成多个候选策略，不固定 `lambda_t`，不启用 `omega(t)`。

### 调用 3：Message Generation

输入：`H_t`、S Self Domain 的生成安全视图、P 相关 User Domain、P User State、`lambda_t` 和唯一 Behavior Policy。  
输出：一条以 S 身份说出的消息正文。

完整 S Self Domain 已在 Alignment 中参与决策。生成安全视图保留 Persona、表达风格、Behavior Policy Prior 与 Hard Constraints，但不直接披露身份事实和兴趣列表，防止小模型把长期背景误写成当前活动；完整对象继续保存和审计，不被删除或重新生成。

最终生成必须保留论文 Appendix D.1 的任务约束：

```text
You are {S}. Continue the conversation.
Output only the message, not the speaker name.
```

Ours 的内部状态只用于决定如何继续对话，不得出现在最终文本中。没有统一 1--2 句或 2--4 句限制，长度应匹配 S 的 Self Domain 和当前真实历史。

## 9. 每轮哪些变化、哪些不变化

### 每轮变化

- 目标位置 `t`；
- 真实历史 `H_t`；
- 已观察到的伙伴发言；
- P User Domain；
- P Current User State；
- adaptive `lambda_t`；
- Behavior Policy；
- 最终生成消息。

### 每轮不变化

- 当前目标身份 S；
- S Self Domain；
- Ca 数据；
- 生成模型权重；
- Prompt 和 Schema 版本；
- 模型生成结果不进入下一轮真实历史。

## 10. Table 2 评价

对生成消息和真实 `M_t` 使用同一评价流程：

| Table 2 指标 | 实现 | 方向 |
|---|---|---|
| Lexical | ROUGE 重建配置 | 越高越好 |
| Semantic | BERTScore 重建配置 | 越高越好 |
| Reflective | Appendix C.1 / 官方 Prompt，标签一致率 | 越高越好 |
| Grounding | Appendix C.2 / 官方 Prompt，标签一致率 | 越高越好 |
| Sentiment | CardiffNLP 官方模型，标签一致率 | 越高越好 |
| Emotion | CardiffNLP 官方模型，标签一致率 | 越高越好 |
| Intimacy | CardiffNLP 官方模型，绝对差 | 越低越好 |
| Empathy | Appendix C.3 EPITOME，总分绝对差 | 越低越好 |

官方评价模型：

```text
Reflectiveness / Grounding / Empathy: gpt-4o-mini
Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
Emotion: cardiffnlp/twitter-roberta-large-emotion-latest
Intimacy: cardiffnlp/twitter-roberta-large-intimacy-latest
```

先对每位目标说话者汇总，再对 10 位说话者计算 speaker-macro mean 和 standard deviation，形成 Ours 的 Table 2 行。message-micro 结果只作诊断。

Topic 不属于 Table 2。Persona Consistency 是论文 Table 8 的跨对话说话者 EI 差异诊断值，不是逐消息生成指标。

## 11. 论文参考结果

| 方法 | Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o fine-tune | 0.14 ± 0.04 | 0.76 ± 0.08 | 0.62 ± 0.13 | 0.40 ± 0.13 | 0.53 ± 0.22 | 0.43 ± 0.22 | 0.06 ± 0.01 | 1.80 ± 0.55 |
| w/ fine-tune | 0.14 ± 0.05 | 0.78 ± 0.04 | 0.77 ± 0.09 | 0.62 ± 0.08 | 0.59 ± 0.18 | 0.46 ± 0.21 | 0.07 ± 0.01 | 1.24 ± 0.12 |
| Ours | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

Ours 与论文行同任务、同公开 Ca/Cb 分配、同目标消息和同评价定义，但论文没有公开 Persona Simulation 的基础生成模型和运行配置，因此只能称为 protocol-aligned comparison，不能称为完全相同运行时的严格复现。

### 11.1 后续优化的固定论文参照

所有提示词、流程或模型变更都必须输出完整八项，并与论文 `w/o fine-tune`、`w/ fine-tune` 两行并列。优化主参照固定为 `w/ fine-tune`；不得只展示相对上一版的提升，也不得只选择有利指标。

- 高值指标：报告 `Ours - paper`；
- Intimacy/Empathy AD：报告 `paper - Ours`；
- 所有差值统一为正值代表 Ours 更好；
- Reflectiveness/Grounding 是对真实消息标签的一致率，不以更多反思或更多追问为目标；
- 小规模开发集只用于配对诊断，最终比较必须采用完整 10 人、519 条的 speaker-macro mean 与 population std。

优化时优先缩小当前冻结验证中落后的 Reflectiveness、Empathy、Grounding、Sentiment 和 Lexical，同时保护已领先或持平的 Semantic、Emotion 和 Intimacy。任何使用论文结果指导 Prompt 的修改都必须保持通用行为规则，禁止根据具体测试答案、人物或 result ID 定制。

## 12. 原文逐项验证

| 设计项 | 论文/官方状态 | 本协议处理 |
|---|---|---|
| 10 人、每人两条对话 | 论文明确 | 完全沿用 |
| Cb 选择较低 EI 对话 | 论文明确 | 完全沿用 Table 8 |
| 每位说话者单独训练/测试 | 论文明确 | 10 个 speaker-specific 实验 |
| `H_t={M_1,...,M_(t-1)}` | 论文明确 | 真实 causal prefix |
| 目标人物原消息作为真值 | 论文明确 | 完全沿用 |
| D.1 角色 Prompt | 论文明确 | 保留为最终生成任务约束 |
| 连续同说话者消息合并 | 论文第 4.1 节明确 | 同 Session 内合并 |
| 3 Session 条件及其后趋于饱和 | 论文明确 | 采用 3-Session 主条件 |
| 前 3 Session 的精确逐消息构造 | 官方代码未公开 | 按时间顺序取前 3 Session 重建并披露 |
| 精确 Persona Simulation 代码 | 未公开 | 本地重建并记录 hash |
| 具体基础生成模型 | 未公开 | Ours 自行固定并披露，不能伪称相同 |
| 微调参数/checkpoint | 未公开 | Ours 不训练，不复现该运行时 |
| decoding 参数 | 未公开 | Ours 固定并披露 |
| Reflect/Ground/Empathy 评价 | 官方代码和 Prompt 公开 | 应直接沿用官方定义 |
| CardiffNLP 三模型 | 官方代码公开 | 固定模型与 revision |
| ROUGE/BERTScore 具体配置 | 论文未完整公开 | 采用标准重建配置并披露 |
| Self/User Domain、State、lambda、Policy | Ours 新增 | 按方法锚点实现，不冒充论文模块 |
| Cb User Domain 滚动更新 | Ours 新增 | 只使用 `H_t` 已可见信息 |

## 13. 初始 Qwen 运行预案（历史记录）

生成阶段已经冻结并在 `EXP2_REALTALK_OURS_QWEN3_8B_PLAN_ZH.md` 落地：

- Self/User Domain、Alignment、Generation 统一使用 `qwen3-8b`；
- thinking 关闭；
- Self/User/Alignment 使用 `temperature=0.2, top_p=0.9`；
- Generation 使用 `temperature=0.6, top_p=0.9, max_tokens=300`；
- 上下文最多 60,000 字符，只从最旧完整 turn 开始截断；
- 结构化调用最多 3 次逻辑尝试，后两次只允许格式修复；
- ROUGE-L、标准 English BERTScore 和固定 revision 的 CardiffNLP 三模型；
- Prompt、Schema、数据、源码、模型与解码参数全部进入运行签名和 manifest；
- API 密钥不进入仓库或实验产物。

该阶段唯一待补项曾是可验证的 `gpt-4o-mini` API 端点。此预案后来被 Agentic V9
全量运行取代，不能再用于描述当前完成状态。

## 14. 实际完成配置

实际全量结果为 V9：全部阶段使用 `deepseek-v4-flash`，关闭 thinking，完整披露前三个
Session 历史且不压缩、不裁剪。生成完成 `519/519`，独立 Judge 目录随后完成
`3114/3114` 个评价单元，零 unresolved。详细配置、哈希和结果以
`EXP2_REALTALK_FINAL_REPORT_ZH.md` 与 `EXP2_REALTALK_ARTIFACT_MANIFEST.json` 为准。
