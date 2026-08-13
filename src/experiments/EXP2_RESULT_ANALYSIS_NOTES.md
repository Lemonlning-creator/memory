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

## 7. 下一项受控优化：以 `scores_only` 为基线

主优化基线确定为 `scores_only`，不是退回 `full_state`。第四组定义为：

```text
scores_plus_tone
= scores_only
+ activated_tone
```

`response_guidance` 始终排除。这个条件同时提供两组因果比较：

- 相对 `scores_only`：测试加回抽象 tone 能否恢复 Intimacy/Grounding，
  同时保护 Reflective/Empathy。
- 相对 `full_state`：两组只相差 `response_guidance`，可更直接判断过期 guidance
  的净影响。

选择规则：

- 如果第四组保持 Reflective `>= 0.77`、Empathy `<= 1.24`，并让 Intimacy
  回到 `<= 0.07` 或 Grounding 高于 `scores_only`，则采用第四组。
- 如果第四组退回 `full_state` 的 Reflective/Empathy 水平，则固定采用
  `scores_only`，不再加回 tone。
- 无论哪组胜出，都保留 alignment、三个 Empathy 数值、`current_state` 更新、
  用户画像和人设；这不是删除核心算法。

## 8. 后续不得重复的做法

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

## 9. 下一步顺序

1. 只补跑 `scores_plus_tone`，不重跑已经完成的三组。
2. 用同一份报告比较第四组与 `scores_only`、`full_state`。
3. 按第 7 节阈值固定 response-state policy。
4. 固定策略后，优先处理 Grounding 的 33 条 FP；不再调整已经达标的
   Reflective 和 Empathy。
5. Grounding 修改必须是单一、speaker-conditioned 的精确率优化，不允许增加
   默认追问，也不允许继续叠加过去所有 Prompt 条款。
6. Lexical 最后处理；允许它作为最终一到两个未超过论文的指标之一。

## 10. 尚未证明的事项

以下内容目前都只能称为假设：

- `response_guidance` 是初轮改善的唯一原因；
- `activated_tone` 一定有益或一定有害；
- V18 response Prompt 单独就能稳定复现 `0.7915` Reflective；
- 删除关系信息一定能改善 Empathy；
- 增加关系信息一定能改善个性化；
- 问句率降低一定能提高 Grounding；
- 单次温度 0 的三组差异具有统计显著性。

第四组完成后，如果胜出策略相对次优策略的差异仍很小或 speaker 方向明显冲突，
应只重复两个候选条件，而不是重新运行全部四组或继续凭单次结果改 Prompt。
