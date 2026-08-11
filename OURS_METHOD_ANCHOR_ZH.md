# Towards Deep Empathy: Ours 方法锚点

状态：方法定义冻结稿 v1  
用途：固定 Ours 的概念边界、运行顺序和接口；后续实验只能进行任务适配，不得悄然改变方法本体。  
依据优先级：师姐《理论.pptx》核心框架 > 师姐明确确认的实验要求 > 仓库现有代码。仓库代码是待适配实现，不反向定义方法。

## 1. 方法目标

Ours 是一个面向个性化陪伴智能体的动态双域决策框架。它同时建模：

1. 用户长期特征与当前状态；
2. 智能体自身 Persona、行为倾向与边界；
3. 当前交互中 Self Domain 与 User Domain 的动态权衡；
4. 权衡后得到的单一 Behavior Policy；
5. Behavior Policy 到最终输出的生成过程；
6. 新交互证据对后续用户模型的更新。

方法主链路为：

```text
Observed History
  -> User Domain update
  -> Current User State inference
  -> adaptive Self/User alignment with lambda_t
  -> one Behavior Policy
  -> final response
  -> later observed interaction becomes new evidence
```

## 2. 核心对象及严格边界

### 2.1 User Domain

User Domain 表示智能体对用户的显式模型，包含三个时间层次：

- Stable User Model：长期、相对稳定的五层画像；
- Current User State：当前交互时刻的情绪、意图、需要与状态；
- Future User State：在当前交互和所选行为策略影响下，用户可能产生的后续状态或反应。

Stable User Model 使用五层结构：

```text
U = {S_core, S_reg, S_cog, S_id, S_beh}
```

- `core`：核心恐惧、核心欲望、价值观、依恋模式、意义来源；
- `regulation`：回避、控制、讨好、攻击、幽默化、沉迷、理性化等调节模式；
- `cognition`：表达风格、信息密度、情绪显性、社交距离、决策风格；
- `identity`：职业、年龄、社会关系、家庭、经济、设备与空间环境；
- `behavior`：内容偏好、消费偏好、娱乐偏好、习惯和长期行为模式。

长期画像不能保存短暂情绪、单次事件或无证据推断。每个画像属性必须保留 `value`、`confidence` 和内部证据来源。

### 2.2 User State

Current User State 只描述用户当前怎样，包括：

- 当前情绪及强度；
- 当前意图；
- 当前主要需要；
- 当前互动期待；
- 状态依据和不确定性。

Future User State 描述用户之后可能怎样，或可能如何响应当前行为策略。它不是建议文本，也不是 Agent 应采取的策略。

`recommended_intervention`、`response_guidance`、`activated_persona` 等行动字段不属于 User State，必须放入 Behavior Policy 或 Alignment 输出。

### 2.3 Self Domain

Self Domain 表示 Agent 自身，包括：

- Persona：稳定人格、语气和表达方式；
- Behavior Policy Prior：默认互动原则、情绪回应方式、指导方式和主动性；
- Generative Constraints：最终生成时必须遵守的角色和事实约束；
- Hard Constraints：不可因用户偏好或 `lambda_t` 改变的安全、诚实和边界要求。

Self Domain 通常在实验初始化时生成或配置一次并缓存，不随每条用户消息重建。只有专门研究 Agent 自我演化时才允许更新 Self Domain。

### 2.4 Adaptive Empathy Weight `lambda_t`

`lambda_t in [0, 1]` 是当前第 `t` 轮 Self Domain 与 User Domain 的自适应权衡系数：

```text
G_social = (1 - lambda_t) * G_self + lambda_t * G_other
```

其操作性含义为：

- 较低 `lambda_t`：Behavior Policy 更多保持 Agent 的自然 Persona 和软性行为倾向；
- 较高 `lambda_t`：Behavior Policy 更强地适配用户当前需要、互动期待和相关长期偏好；
- 任意 `lambda_t`：Self Domain 的 Hard Constraints 始终有效。

`lambda_t` 不是：

- 固定实验超参数；
- 共情强度本身；
- User Profile 的置信度；
- Explore/Exploit 权重；
- 由最终回复事后补写的解释。

`lambda_t` 必须每轮显式输出并记录。它由 Alignment 模型根据当前历史、query、User Domain、User State 和 Self Domain 自适应判断，不需要单独一次模型调用，也不需要构造多个候选 Behavior Policy。

为了使数值具有稳定语义，Prompt 必须提供区间锚点：

| `lambda_t` 区间 | 权衡语义 |
|---|---|
| `[0.00, 0.25)` | Self-dominant |
| `[0.25, 0.50)` | Self-leaning |
| `[0.50, 0.75)` | User-leaning |
| `[0.75, 1.00]` | Strongly user-oriented |

判断依据至少包含：当前需要强度、情绪脆弱性、互动风险、个性化相关性，以及 Self/User 之间是否存在张力。画像置信度只决定某条画像证据能否使用，不直接决定 `lambda_t`。

### 2.5 Behavior Policy

Behavior Policy 是 Alignment 后得到的唯一行动策略，不生成多个候选，也不进行候选搜索或数值排序。它至少包含：

- response objective；
- perspective-taking；
- emotion alignment；
- personalization approach；
- Self Domain expression；
- directness；
- guidance level；
- question policy；
- tone；
- avoid list。

Behavior Policy 必须明确体现 `lambda_t` 对 Self/User 取舍造成的实际影响，但不能突破 Hard Constraints。

### 2.6 Generative Model

Generative Model 只负责把选定的 Behavior Policy 实现为自然文本。它不得重新计算 `lambda_t`、另选策略或在回复中披露 Profile、User State、Behavior Policy、`lambda_t` 等内部结构。

## 3. 标准运行顺序

### 3.1 初始化阶段，不计入每轮三次调用

#### 初始化 User Domain

输入：目标用户在当前允许时间范围内的历史对话。  
输出：五层 Stable User Model。

#### 初始化 Self Domain

输入：Agent 的预设 Persona，或当前实验允许使用的 Agent 历史发言。  
输出：Persona、Behavior Policy Prior、Generative Constraints 和 Hard Constraints。

两个初始化过程互不依赖，可以并行。完成后分别缓存。

### 3.2 调用 1：更新 User Domain

输入：

```json
{
  "previous_user_domain": {},
  "new_observed_history": [],
  "evidence_cutoff": "strictly before the current target"
}
```

输出：

```json
{
  "user_domain": {
    "core": {},
    "regulation": {},
    "cognition": {},
    "identity": {},
    "behavior": {}
  },
  "update_summary": {
    "added": [],
    "strengthened": [],
    "weakened": [],
    "unchanged": []
  }
}
```

运行规则：

- 只使用当前目标之前已观察到的历史；
- 可以按 Session 更新并缓存，不要求每条消息重复调用；
- 当前 Session 内的瞬时变化由 User State 表示；
- 更新完整画像，但后续规划只披露与当前 query 相关的画像值和置信度；完整证据留在内部日志中。

### 3.3 调用 2：State + Adaptive Alignment + Policy

这是方法核心的一次结构化调用。

输入：

```json
{
  "recent_history": [],
  "current_query": "",
  "user_domain": {},
  "self_domain": {},
  "previous_user_state": {}
}
```

模型必须按以下语义顺序决策：

```text
1. 以 current_query 和 recent_history 为主要证据识别 Current User State
2. 从 User Domain 中识别当前相关而且有证据支持的信息
3. 判断当前 Self/User 权衡并输出 adaptive lambda_t
4. 基于 lambda_t 生成一个且仅一个 Behavior Policy
5. 可选地预测该策略下的 Future User State
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
    "state_evidence": [],
    "uncertainty": "low|medium|high"
  },
  "relevant_user_domain": {
    "core": [],
    "regulation": [],
    "cognition": [],
    "identity": [],
    "behavior": []
  },
  "alignment": {
    "lambda": 0.0,
    "orientation": "self-dominant|self-leaning|user-leaning|strongly-user-oriented",
    "basis": {
      "current_need_intensity": "low|medium|high",
      "emotional_vulnerability": "low|medium|high",
      "interaction_risk": "low|medium|high",
      "personalization_relevance": "low|medium|high",
      "self_user_tension": "none|low|medium|high"
    },
    "rationale": "",
    "policy_effect": "how lambda_t changes the selected behavior"
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
  },
  "future_user_state": {
    "enabled": true,
    "expected_reaction": "",
    "risk_of_misalignment": "",
    "uncertainty": "low|medium|high"
  }
}
```

Future User State 属于完整方法。某项实验可以不对它评分，但必须在实验适配文档中明确是“保留为内部规划信息”还是“该适配中禁用”，不能静默删除。

### 3.4 调用 3：最终回复生成

输入：

```json
{
  "recent_history": [],
  "current_query": "",
  "self_domain": {},
  "relevant_user_domain": {},
  "user_state": {},
  "alignment": {
    "lambda": 0.0,
    "orientation": "",
    "policy_effect": ""
  },
  "behavior_policy": {}
}
```

输出：仅最终回复文本。

生成约束：

- 执行唯一 Behavior Policy，不重新规划；
- 当前 query 优先于长期画像；
- 不机械披露画像或内部推断；
- 不捏造事实、经历、身份或当前事件；
- 回复长度和形式匹配当前对话，不设置脱离任务的统一句数限制；
- 不输出任何内部 JSON、分数、`lambda_t` 或分析过程。

## 4. `omega(t)` 与 Exploration

`omega(t)` 控制信息增益与 Explore/Exploit，不等同于 `lambda_t`。它不是当前 Ours 方法锚点中的必需实现，可以按实验目的显式禁用。

禁用时必须记录：

```json
{
  "omega_enabled": false,
  "active_exploration_control": false
}
```

禁用 `omega(t)` 不得删除 User Domain、User State、Self Domain、adaptive `lambda_t` 或 Behavior Policy。

## 5. 允许的工程简化

以下简化不改变方法本体：

- 将 User State、adaptive `lambda_t` 和 Behavior Policy 合并为一次结构化 Alignment 调用；
- User Domain 按 Session 更新并缓存；
- Self Domain 初始化一次并缓存；
- 不生成多个候选 Behavior Policy；
- 不显式估算 `G_self`、`G_other` 的伪精确数值；
- 不增加独立 Verification 或重写调用；
- 某实验不评价 Future User State，但需声明其保留或禁用方式；
- 根据实验禁用 `omega(t)`。

## 6. 不允许的偏移

以下变化会改变或削弱 Ours，不能作为无声明适配：

- 把 `lambda_t` 固定为全数据统一常数；
- 让 `lambda_t` 完全隐含且不记录；
- 先生成 Behavior Policy，再事后补写 `lambda_t`；
- 让最终生成器忽略或重新决定 Behavior Policy；
- 将建议、干预方式或 Persona 激活写入 User State；
- 用未来消息构造当前 User Domain 或 User State；
- 强迫每个画像层都在当前轮被激活；
- 因画像置信度低而机械降低对用户当前需要的关注；
- 将 `lambda_t` 与共情强度、画像置信度或 `omega(t)` 混同；
- 加入统一的 1--2 句、2--4 句等与具体实验协议无关的回复限制；
- 引入候选策略搜索、额外 Judge 或 Verification 后仍声称是同一实现而不披露。

## 7. 实现可审计性要求

每个实验样本至少保存：

- 因果历史截止位置；
- User Domain 版本和输入哈希；
- Self Domain 版本和输入哈希；
- 调用 2 的完整结构化输出；
- `lambda_t`、区间、依据和 `policy_effect`；
- 最终 Behavior Policy；
- 最终回复；
- Prompt、Schema、模型、解码参数及其哈希；
- 是否启用 Future User State 和 `omega(t)`；
- 重试、检查点和失败记录。

这些记录用于证明方法链路真实运行，而不是只在最终 Prompt 中出现方法名词。

## 8. 与实验适配的边界

本文件只冻结 Ours 方法，不决定具体 Benchmark 的角色映射。以下事项必须在独立实验协议中确认：

- Benchmark 要模型扮演谁；
- User Domain 对应谁；
- Self Domain 对应谁；
- 当前 query 和 Ground Truth 分别是哪条消息；
- 历史如何滚动以及哪些 Session 可见；
- 训练、建模、验证和测试边界；
- Future User State 是否属于主指标；
- Ours 输出是否与论文表格中的输出对象完全一致；
- 评价器、模型、解码和统计单位。

只有完成角色和数据映射后，Ours 结果才可与论文结果横向比较。实验适配可以改变数据接口、缓存粒度和启用模块，但不得改变第 2--6 节定义的方法内核。

## 9. 当前冻结结论

Ours 的当前确定实现为：

```text
Five-layer evolving User Domain
+ stable Self Domain
+ current User State
+ explicit adaptive lambda_t
-> one Behavior Policy
-> final generation
```

在线主流程采用三次逻辑调用：

```text
1. User Domain update
2. User State + adaptive lambda_t + one Behavior Policy
3. Final response generation
```

其中调用 1 可按 Session 缓存，Self Domain 在初始化阶段生成并缓存。下一阶段只确认实验协议以及它与本方法的角色映射，不再重新定义 Ours。
