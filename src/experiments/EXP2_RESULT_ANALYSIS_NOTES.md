# Experiment 2：结果分析与决策记录

更新时间：2026-08-14

这份文档只记录已经由全量结果、逐样本诊断或代码审计支持的结论。它不是新的 Prompt，也不用于解释模型输入。后续调整前应先检查本文件，避免重复走已经失败的方向。

## 1. 当前目标

主目标仍然是：在保持 REALTALK Table 2 官方评估协议不变的前提下，让 Ours 超过论文 `w/ fine-tune` 的大部分指标；允许一到两个指标暂时没有超过，但不能通过修改评估 Prompt、筛除不利对话或混用缓存来实现。

当前最需要解决的组合问题是：

- 保留 V18 已经取得的 Reflective 优势；
- 保留 V7/V23 较低的 Empathy 误差；
- 将 Grounding 从 V18 的 `0.5765` 提高到论文的 `0.62` 以上；
- 不牺牲已经较好的 Semantic、Emotion 和 Intimacy。

论文 `w/ fine-tune` 目标值：

| Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.07 | 1.24 |

## 2. 已确认的关键全量结果

### 2.1 V18 是目前最接近主目标的综合版本

来源目录：

```text
data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate
```

| Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1070 | 0.8330 | 0.7915 | 0.5765 | 0.5892 | 0.4955 | 0.0670 | 1.2591 |

相对论文 `w/ fine-tune`：

- 已超过：Semantic、Reflective、Emotion、Intimacy。
- 基本持平：Sentiment，仅低 `0.0008`。
- 仍未达到：Lexical、Grounding、Empathy。
- V18 的价值不是“全部指标最好”，而是首次可靠地让 Reflective 超过论文，同时没有破坏大部分其他指标。

### 2.2 V7 是重要的行为基线，不是最终答案

来源目录：

```text
data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation
```

已确认的关键全量值：

| Reflective ↑ | Grounding ↑ | Sentiment ↑ | Empathy ↓ |
|---:|---:|---:|---:|
| 0.7171 | 0.5661 | 0.6511 | 1.1079 |

- 优势：Empathy 明显优于论文，Sentiment 较好，整体真实人物表面风格稳定。
- 不足：Reflective 和 Grounding 都没有超过论文。
- 因此后续版本应该保留 V7 的自然行为基线，但不能简单把新的指标规则不断追加在 V7 后面。

### 2.3 V23 证明低 Empathy 可以保留，但没有解决 Reflective/Grounding

已确认的关键全量值：

| Reflective ↑ | Grounding ↑ | Empathy ↓ |
|---:|---:|---:|
| 0.7189 | 0.5544 | 0.9346 |

- V23 的 Empathy 是目前记录中最亮眼的结果之一。
- 但 Reflective 和 Grounding 都明显低于论文，也低于 V18 的 Reflective。
- 它说明“低 Empathy 误差”与“高 Reflective”不是天然一起出现的，需要查清状态输入和回复行为规则各自的影响。

### 2.4 V25 的集成方式失败

已确认的关键全量值：

| Reflective ↑ | Grounding ↑ | Empathy ↓ |
|---:|---:|---:|
| 0.6865 | 0.4592 | 1.3199 |

V25 不是偶然波动，而是出现了明确的行为偏差：

- Reflective 混淆：`TP=10, TN=78, FP=18, FN=11`，生成回复 Reflective 阳性率为 `23.9%`。
- Grounding 混淆：`TP=42, TN=19, FP=53, FN=3`，生成回复 Grounding 阳性率为 `81.2%`。
- 问句率达到 `90.6%`，平均长度约 `42.3`。
- Empathy 总分相对 reference 的偏差约为 `+0.667`；其中 Exploration 偏差约为 `+0.436`。

直接原因是 V25 允许模型“先正常回应，再问一个具体问题”。模型把这个许可近似执行成了默认追问，导致大量 Grounding 假阳性、过度 Exploration 和更高 Empathy 误差。

结论：Grounding 不是“多问相关问题”即可提高。该指标衡量 generated 与 reference 是否同为 Grounding，而不是问句越多越好。

## 3. 从 Prompt 版本结果得到的主要规律

### 3.1 不能继续使用纯叠加式 Prompt

已经失败的模式是：

```text
V7 基础规则
+ Reflective 规则
+ Grounding 规则
+ Empathy/关系规则
+ 更多例外和补充说明
```

问题不只是 Prompt 太长，而是多个局部目标同时争夺同一条回复：模型为了满足所有规则，容易生成“回应 + 自我反思 + 追问 + 情绪理解”的复合回复。这与 REALTALK 中大量普通、单一行为的真实回复不一致。

后续 Prompt 必须是完整重写或明确择优，而不是继续把上一版失败规则全部保留后再加新条款。

### 3.2 显式加入关系建模没有自动改善 Empathy

V17 `v17_role_relationship_calibrated` 的已确认结果包括：

```text
Reflective 0.7106
Grounding 0.4473
Emotion 0.5351
Intimacy 0.0663
Empathy 1.8586
```

它说明：在最终回复 Prompt 中显式强调双方关系距离，并不会自然让回复更接近 reference 的 Empathy。关系距离容易被模型理解为要主动调整、解释或表达关系，从而产生更多非真实人物行为。

因此目前不能得出“缺少显式关系建模是 Empathy 差的主因”。关系可以作为边界，但不应成为必须在回复中执行的动作。

### 3.3 Reflective 和 Grounding 都是标签匹配问题

- Reflective 高，不代表每条回复都要自我反思；reference 为 False 时，生成 False 同样得分。
- Grounding 高，不代表尽可能追问；reference 不 Grounding 时，生成普通回复才正确。
- 优化重点应该是判断每条消息需要哪一种 response act，而不是全局提高某个行为的出现率。
- V18 的 A/B/C 单一主行为选择曾有效提高 Reflective；V25 允许附加第二行为后反而退化，这是重要证据。

## 4. 已发现的实验混杂因素

此前把 V7、V18、V23、V25 直接当作“只改变最终回复 Prompt”的比较并不完全严格，因为每个版本都会重新运行 alignment：

```text
alignment temperature = 0.3
response temperature = 0.4
```

逐条输入审计结果：

- 四个版本的 `relevant_memory`：`117/117` 完全一致。
- 四个版本的完整 `previous_empathy_state`：只有 `10/117` 完全一致，这十条主要是每个对话的第一个评测点。
- 四个版本的三个 Empathy 数值同时一致：`56/117`。
- 四个版本的前置 `current_state` 完全一致：只有 `10/117`。
- V18 到 V25 两两比较：完整 previous state 一致 `10/117`，三个数值一致 `75/117`。
- V18 到 V25 两两比较：前置 current state 完全一致 `11/117`，核心类别字段一致 `63/117`。

这意味着后续轮次不仅最终回复 Prompt 不同，传入回复 Prompt 的状态轨迹也不同。由此不能直接断言：

- V18 的 Reflective 提升全部来自 V18 response Prompt；
- V23 的低 Empathy 全部来自 V23 response Prompt；
- V25 的退化全部来自 V25 response Prompt。

V25 的高问句率有明确 Prompt 因果证据，但多个版本之间的总体差异仍包含随机状态轨迹影响。

## 5. 完整上一轮状态可能存在的具体干扰

主实验默认把完整 `empathy_state` 传给下一轮最终回复，其中不只有三个数值，还包括：

```text
emotional_reaction
interpretation
exploration
activated_tone
response_guidance
```

需要重点检查的是 `activated_tone` 和 `response_guidance`：

- 它们是上一轮 alignment 针对上一条消息生成的自然语言指令；
- 到下一轮使用时已经滞后一轮；
- `response_guidance` 可能包含“表达理解”“询问具体感受”“保持温暖”等行为建议；
- 即使最终回复 Prompt 没有显式写关系规则，这些字段仍可能间接带入关系深度和回复动作；
- 它们可能覆盖当前消息和 V7/V18 的 response-act 判断。

初轮受控消融已经确认：保留三个数值、删除其余非数值字段的
`scores_only` 优于彻底删除状态的 `no_state`，且在 Reflective 和 Empathy
上优于 `full_state`。但初轮同时删除了 `activated_tone` 和
`response_guidance`，尚不能把改善只归因于某一个字段。

## 6. 受控状态消融结果

入口：

```text
src/experiments/exp2_controlled_state_ablation.py
```

说明：

```text
src/experiments/README_exp2_controlled_state_ablation.md
```

测试冻结同一份完成版 V18 的：

- 用户消息；
- teacher-forcing 历史和 `relevant_memory`；
- 用户画像和智能体人设；
- 前置 `current_state` 轨迹；
- V18 response system/user Prompt；
- 模型、温度和最大输出长度。

初轮比较：

| 条件 | 传入最终回复的上一轮状态 |
|---|---|
| `full_state` | 完整来源状态 |
| `scores_only` | 三个 Empathy 数值 |
| `no_state` | `{}` |

重要原则：真正的对照是本次重新生成的 `full_state`，不是历史 V18 回复。历史 V18 只作为已有结果背景，因为历史回复是在另一时间、另一随机采样中生成的。

汇总前脚本会逐 `example_id` 比较三组 `frozen_input_sha256`。只要非消融输入不一致，就拒绝生成报告。

### 6.1 初轮输入审计

结果目录：

```text
data/exp2_controlled_state_ablation_v18
```

审计结果：

```text
condition_count = 3
example_count = 117
frozen_input_audit.verified = true
```

三组的用户消息、teacher-forcing 历史、`relevant_memory`、画像、人设、前置
`current_state`、V18 response Prompt、模型和温度完全一致；只有传入最终回复的
上一轮 empathy payload 不同。因此三组之间可以进行受控因果比较。

### 6.2 初轮主结果

| 条件 | Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_state` | 0.1082 | 0.8324 | 0.7153 | **0.5872** | 0.6092 | 0.4905 | **0.0667** | 1.2389 |
| `scores_only` | 0.1050 | 0.8319 | **0.7906** | 0.5681 | 0.5904 | 0.4967 | 0.0738 | **1.1217** |
| `no_state` | 0.1075 | 0.8330 | 0.7549 | 0.5336 | 0.6124 | 0.4994 | 0.0755 | 1.2359 |
| `scores_plus_tone` | 0.1040 | 0.8316 | **0.8180** | 0.4941 | 0.5720 | 0.4468 | 0.0744 | 1.1597 |

`scores_only` 相对论文 `w/ fine-tune` 已超过 Semantic、Reflective、Sentiment、
Emotion 和 Empathy，共 5/8 项；Intimacy 只差 `0.0038`，仍未达到的是 Lexical、
Grounding 和 Intimacy。

初轮结论：

- `no_state` 在 Reflective、Grounding、Empathy 和 Intimacy 上都弱于
  `scores_only`，所以不能删除上一轮状态整体。
- 三个 Empathy 数值包含有效信息，必须保留。
- `scores_only` 相对 `full_state` 将 Reflective 从 `0.7153` 提高到 `0.7906`，
  将 Empathy 误差从 `1.2389` 降到 `1.1217`。
- Reflective 逐样本变化为 11 条改善、3 条退化、103 条不变；方向明确但单次
  运行尚不能声称统计显著。
- `scores_only` 的 Intimacy 和宏平均 Grounding 弱于 `full_state`，说明被一起
  删除的 `activated_tone` 可能仍有价值。

### 6.3 Grounding 诊断

`scores_only` 的 Grounding 混淆矩阵：

```text
TP = 32
TN = 39
FP = 33
FN = 13
```

reference Grounding 阳性率为 `38.46%`，generated 为 `55.56%`。当前主要错误是
误触发 Grounding，而不是漏掉 Grounding，因此后续优化方向是减少假阳性，不是
增加问题。`full_state` 和 `scores_only` 的问句率同为 `53.85%`，说明 Reflective/
Empathy 改善并非简单来自减少问号。

Grounding 具有明显 speaker 异质性：`scores_only` 在 Elise、Nebraas、Paola 上
改善，在 Fahim Khan、Vanessa 上退化，在 Muhhamed 上不变。逐样本微平均为
13 条改善、12 条退化，但论文采用 speaker 宏平均，所以最终宏平均下降。不能用
统一的“多问”或“少问”规则解决，必须按目标 speaker 的近期行为频率校准。

### 6.4 对过期 guidance 的直接观察

人工抽查发现多条自然语言 guidance 与当前主题错位：

- 当前讨论 Guardians，guidance 仍要求回应博物馆和恐龙；
- 当前讨论 Harry Potter，guidance 仍要求回应 Guardians；
- 当前讨论狼人场景，guidance 仍要求围绕 Sirius/Goblet 追问。

这与既定的一轮滞后时序吻合。三个数值可以表示较抽象的上一轮强度，内容级
`response_guidance` 却容易在下一轮过期。该观察强烈支持排除 guidance，但初轮
受控条件仍不能区分 `activated_tone` 和 `response_guidance`
各自的贡献。

## 7. 第四组结果与最终状态策略

第四组严格以 `scores_only` 为基线：

```text
scores_plus_tone
= scores_only
+ activated_tone
```

`response_guidance` 始终排除。它同时提供两组因果比较：

- 相对 `scores_only`：测试加回抽象 tone 能否恢复 Intimacy/Grounding。
- 相对 `full_state`：两组只相差 `response_guidance`，可观察排除 guidance 后的
  净变化。

第四组结果：

```text
Reflective  0.8180
Grounding    0.4941
Empathy      1.1597
Intimacy     0.0744
```

相对 `scores_only`，tone 只使 Reflective 提高 `0.0274`，但同时造成：

```text
Grounding  -0.0741
Sentiment  -0.0184
Emotion    -0.0499
Empathy    +0.0380 error
Intimacy   +0.0007 error
Lexical/Semantic 轻微下降
```

行为诊断：

- Reflective：`TP=14, TN=86, FP=10, FN=7`，比 `scores_only` 少 2 个 FP。
- Grounding：`TP=31, TN=32, FP=40, FN=14`；相对 `scores_only` 增加 7 个 FP，
  并增加 1 个 FN。
- Grounding generated 阳性率从 `55.56%` 升到 `60.68%`。
- 问句率反而从 `53.85%` 降到 `52.99%`，说明 tone 诱发的是更宽泛的
  共同理解、情绪承接或隐式探索行为，不是简单增加问号。
- tone 的影响具有明显 speaker 异质性：它严重损害 Paola 和 Muhhamed 的
  Grounding，并未形成稳定的整体收益。

最终决定：固定采用 `scores_only`，不再把 `activated_tone` 或
`response_guidance` 传给最终回复。原因是它是唯一同时达到 Reflective 和
Empathy 论文目标、且综合损失最小的状态条件。`scores_plus_tone` 虽然取得最高
Reflective，但其 Grounding、Sentiment、Emotion、Intimacy 和 Empathy 均弱于
`scores_only`，不具备采用价值。

固定 `scores_only` 不等于删除核心算法：alignment、三个 Empathy 数值、
`current_state` 更新、用户画像和人设全部保留；只收窄最终回复读取的上一轮状态
接口。

## 8. `scores_only` Grounding 逐条误差复盘

### 8.1 总体错误并不等于“问号太多”

`scores_only` 的 33 条 Grounding FP 中：

- 21 条包含问号；
- 12 条完全没有问号，但包含主动概括、解释、个人类比、主题扩展或较丰富的
  自我披露，仍被 judge 判为 Grounding。

对 21 条带问号 FP 的人工归类：

- 15 条是当前话题相关、但 reference 没有采用的可选追问或互问；
- 2 条转向过期或无关话题；
- 4 条是修辞问句、附加问句或软确认，而不是真正必要的澄清。

对 12 条无问号 FP 的人工归类：

- 5 条把普通回应扩写成解释性概括或抽象推论；
- 6 条增加了个人经历、类比或超出 reference 焦点的自我披露；
- 1 条把直接回答扩成了多选项推荐。

因此后续不能只写“少问问题”。当前 judge 会把某些内容扩展和主动建立共同话题也
识别为 Grounding。ORDINARY 分支本身必须保持单一、直接，不能在不需要时通过
解释、类比或个人经历继续扩展用户的话题。

### 8.2 FN 证明不能全局禁止 Grounding

13 条 FN 可以分成：

- 6 条漏掉 reference 中直接且相关的澄清或追问；
- 5 条漏掉 reference 中不带问号、但确实推进当前共同话题的信息扩展；
- 2 条虽然 generated 带问号，但问题过于宽泛或转向另一个焦点，仍被判为
  非 Grounding。

问号与 Grounding 标签的关系本身很弱：

```text
reference: Grounding=True  且有问号 18
reference: Grounding=True  且无问号 27
reference: Grounding=False 且有问号 19

generated: Grounding=True  且有问号 44
generated: Grounding=True  且无问号 21
generated: Grounding=False 且有问号 19
```

用户消息包含问题时，reference Grounding 阳性率反而只有 `31.82%`；用户消息没有
问题时为 `42.47%`。这说明“用户问了问题，所以回复后也问一个问题”是错误规则。
直接回答用户问题通常已经构成完整回复，不需要再附加追问。

### 8.3 最明显、且可由输入观察的错误条件是缺少同 Session 行为证据

当当前 Session 在用户消息前最多只有 2 个 history bubble 时：

```text
评测点                         14
reference Grounding 阳性        0
Grounding FP                    6
Grounding FN                    0
```

这 6 条 FP 分布为：Muhhamed 2、Elise 2、Nebraas 1、Fahim Khan 1。它们通常发生在
缺少可比较的当前 Session 回复时，模型改用抽象 Persona 推断行为，并产生多主题
回应、可选追问、个人类比或虚构的具体经历。

如果只把这 6 条 FP 改正确、其余样本完全不变，speaker 宏平均 Grounding 的理论值
可从 `0.5681` 上升到约 `0.6383`，已经超过论文 `0.62`。这只是基于现有标签的
反事实上限，不代表任意 Prompt 都能无副作用地实现，但它指出了当前最高收益、
最小改动的测试方向。

合理规则不是硬编码“Session 开头一定不 Grounding”，而是：当看不到同一目标
speaker 在可比较情境中的近期真实回复时，不允许仅凭 Persona 中的 `frequent`、
`occasional`、`thoughtful` 或 `curious` 推导可选行为；只回应最后一个直接问题或
最后一个活跃点，并默认一个普通动作。

### 8.4 speaker 宏平均暴露了两个不同问题

| Speaker | n | Ref Grounding | Gen Grounding | FP | FN | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Fahim Khan | 7 | 1 | 3 | 3 | 1 | 0.4286 |
| Muhhamed | 17 | 4 | 14 | 10 | 0 | 0.4118 |
| Elise | 16 | 4 | 10 | 7 | 1 | 0.5000 |
| Paola | 16 | 7 | 10 | 4 | 1 | 0.6875 |
| Nebraas | 28 | 13 | 13 | 4 | 4 | 0.7143 |
| Vanessa | 33 | 16 | 15 | 5 | 6 | 0.6667 |

- Muhhamed、Elise、Fahim Khan 的主要问题是 Grounding 总频率明显过高，优先提高
  precision。
- Nebraas、Vanessa 的生成总频率已经接近 reference，问题是具体回合放错位置；
  对它们全局降频会把 FP 转换成 FN。
- 论文采用 speaker 宏平均，因此 Fahim Khan 的一个样本变化比 Vanessa 的一个
  样本变化权重更大。后续必须同时报告逐样本混淆矩阵和 speaker 宏平均，不能只看
  117 条的微平均。

Persona 审计没有发现 schema 错误，但发现了使用方式上的问题：Muhhamed Persona
明确写着“rare direct personal questions”以及“frequent topic expansion”，模型的
主要过触发确实表现为无问号的主题扩展；Elise Persona 写着“frequent follow-up
questions”，但 held-out 测试段 reference 仅 `4/16` 为 Grounding，generated 却为
`10/16`。训练段抽取的粗粒度频率可以约束总体风格，但不能直接充当当前回合的
Grounding 决策。

### 8.5 Grounding FP 同时污染其他指标

33 条 Grounding FP 的平均 Empathy 绝对误差为 `1.4545`，其余 84 条为 `0.8690`；
FP 子集的 Reflective accuracy 为 `0.7879`，其余样本为 `0.8571`。因此减少可选
追问、情绪探索和过度解释，有机会同时保护 Empathy 与 Reflective，而不只是提高
Grounding。FP 与非 FP 的 Semantic 基本相同，Lexical 也没有显示可直接获益，
所以本轮不能声称该修改会解决 Lexical。

### 8.6 下一版只应验证一个完整假设

固定 V18、`scores_only`、画像、人设、模型、温度和所有冻结输入，只重写最终回复
的 response-act 选择逻辑：

1. 先锁定最后一个直接问题；没有直接问题时才选择最后一个活跃点。
2. 用户已经提出直接问题时，直接回答默认消耗本回合唯一动作，不自动反问。
3. 只有当前消息存在真正未解决的信息点，并且近期真实目标 speaker 在可比较情境
   中确实采用过追问或共同话题扩展，才允许 Grounding。
4. 缺少可比较的近期目标 speaker 回复时，Persona 只能约束措辞和事实边界，不能
   单独触发追问、类比、解释性扩展或个人经历。
5. ORDINARY 必须是一个直接回答、简短反应、意见或必要事实；不要通过附加问句、
   解释性复述、个人平行经历形成第二个动作。
6. 保留 V18 的 Reflective 判定边界，不修改 Empathy 数值状态，也不新增关系规则。

这是一个“局部证据可用性 + 单一活跃点”的精确率实验，不是继续在 V18 后追加一串
指标条款。先用同一份冻结轨迹跑该单版本；若 Grounding 提高但 Reflective 或
Empathy 明显退化，再逐条查看发生变化的样本，不同时开启第二个 Prompt 变量。

实现版本：`v26_local_evidence_single_act`。该 Prompt 为独立重写，response-state
policy 固定为 `scores_only`；受控运行使用 V18 作为 `--source-prompt-version`，
使用 V26 作为 `--response-prompt-version`，因此不会重新运行 prepare 或 alignment。
一键入口为 `scripts/run_exp2_v26_controlled.sh`。

### 8.7 V7 → V18 混淆矩阵与 V27 假设

V7 和原始 V18 在同一 117 条样本上的 Grounding 混淆矩阵为：

| 版本 | TP | TN | FP | FN | generated positive |
|---|---:|---:|---:|---:|---:|
| V7 | 34 | 34 | 38 | 11 | 72 |
| V18 | 33 | 39 | 33 | 12 | 66 |

V18 修正了 13 条 V7 FP 和 6 条 V7 FN，但同时新产生 8 条 FP 和 7 条 FN；逐样本为
19 条改善、15 条退化、83 条不变。它确实降低了过触发并将 precision 从 `47.2%`
提高到 `50.0%`，但 Grounding 位置仍在大量互换。角色层面 Vanessa 明显改善、
Nebraas 小幅改善、Paola 退化，其余角色宏平均正确率不变。因此不能继续使用统一的
全局降频规则。

V27 `v27_grounding_three_mode_gate` 固定 V18 的 Reflective/Ordinary 边界和
`scores_only` 状态接口，只将 Grounding 分成 `DIRECT_RESPONSE`、
`REPAIR_GROUNDING`、`ELABORATION_GROUNDING`。它验证的唯一假设是：区分完整直接
回应、必要理解修复和有局部角色证据的具体延展，能减少 FP，同时避免 V26 式的
全局 Grounding 抑制造成新的 FN。运行入口为
`scripts/run_exp2_v27_controlled.sh`，结果必须写入新的 V27 目录。

## 9. 后续不得重复的做法

- 不再通过增加默认追问来优化 Grounding。
- 不再通过增加通用情绪理解、安慰或探索来优化 Empathy。
- 不再把用户画像全部字段视为每条回复都必须利用的内容。
- 不修改当前固定用户画像和人设 schema 来追逐某一版 Prompt 的分数。
- 不对用户画像做字段选择；当前主实验继续完整传入固定画像。
- 不在线更新长期用户画像；测试阶段只更新既定的短期用户状态。
- 不因三条或二十四条 diagnostic 子集表现好，就直接认定全量一定有效。
- 不覆盖旧版本目录；Prompt、协议或来源输入变化时必须使用新输出目录。
- 不修改 REALTALK Table 2 评估 Prompt 来适配某个生成版本。
- 不把同一输出目录中的旧 annotation 当作新回复结果；缓存必须绑定 candidate 和 context 哈希。

## 10. 下一步顺序

1. 将主实验 response-state policy 固定为 `scores_only`，不再进行状态字段消融。
2. 后续 Prompt 对比必须使用同一份冻结状态轨迹，避免重新运行 alignment 造成
   输入混杂。
3. 下一次只测试第 8.7 节的 V27 Grounding 三模式落点，不同时修改 Reflective、
   Empathy、画像、人设或评估 Prompt。
4. 首先检查当前 Session 行为证据不足的 14 条，其中现有结果包含 6 个 FP、0 个
   FN；随后再检查 Muhhamed 和 Elise 的剩余 FP。
5. Grounding 修改必须是单一的 precision 优化，不允许增加默认追问，也不允许
   继续叠加过去所有 Prompt 条款；Nebraas 和 Vanessa 只调整落点，不全局降频。
6. Lexical 最后处理；允许它作为最终一到两个未超过论文的指标之一。

## 11. 尚未证明的事项

以下内容目前都只能称为假设：

- `activated_tone` 在其他模型、数据集或任务中一定有害；当前结论只适用于本次
  REALTALK/V18 最终回复接口；
- `response_guidance` 对所有指标都只有负面作用；它在本次 Grounding/Intimacy
  上可能提供局部收益，但综合代价更高；
- V18 response Prompt 单独就能稳定复现 `0.7915` Reflective；
- 删除关系信息一定能改善 Empathy；
- 增加关系信息一定能改善个性化；
- 问句率降低一定能提高 Grounding；
- 单次温度 0 的三组差异具有统计显著性。

最终论文运行前应至少复验一次 `scores_only` 主策略，但不再重复已经被综合结果
淘汰的 `no_state` 和 `scores_plus_tone`。
