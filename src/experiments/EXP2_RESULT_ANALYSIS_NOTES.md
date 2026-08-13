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
empathy_level
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

这目前是有代码依据的假设，但还不是实验结论。不能在受控消融结果出来前直接删除完整状态。

## 6. 当前新增的受控状态消融

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

只比较：

| 条件 | 传入最终回复的上一轮状态 |
|---|---|
| `full_state` | 完整来源状态 |
| `scores_only` | 三个 Empathy 数值 |
| `no_state` | `{}` |

重要原则：真正的对照是本次重新生成的 `full_state`，不是历史 V18 回复。历史 V18 只作为已有结果背景，因为历史回复是在另一时间、另一随机采样中生成的。

汇总前脚本会逐 `example_id` 比较三组 `frozen_input_sha256`。只要非消融输入不一致，就拒绝生成报告。

## 7. 受控结果出来后的决策规则

### 情况 A：`scores_only` 稳定优于 `full_state`

如果 Reflective/Grounding 提高且 Empathy 误差降低，并且逐样本 wins 明显多于 losses，则说明自然语言 tone/guidance 是主要干扰。下一步只调整 response-state policy，保留三个数值，不改 V18 Prompt、画像或人设。

### 情况 B：`no_state` 稳定优于另外两组

说明上一轮 Empathy 状态整体对 REALTALK 下一回复模拟帮助较小，甚至产生滞后干扰。下一步让最终回复不读取上一轮 empathy state；alignment 和用户短期状态更新仍可继续用于算法内部与分析，不能据此删除核心算法。

### 情况 C：`full_state` 最好

说明完整状态总体有效，之前版本不稳定主要仍在 response Prompt 的行为选择。下一步保留完整状态，以 V18 为基础分析 Grounding FN/FP，不再围绕“删状态”调参。

### 情况 D：三组宏观结果接近、逐样本胜负混合

说明状态不是主因，不能因为某一项小数点波动立即决定。下一步应固定状态输入后，对 V18 的错误样本按以下四类分析：

```text
Reflective FP
Reflective FN
Grounding FP
Grounding FN
```

然后只处理占比最大的错误类型，不再同时优化所有指标。

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

1. 先跑 V18 受控状态消融，不再新增 V26 Prompt。
2. 查看 `controlled_state_ablation_summary.md` 的官方宏平均和逐样本 wins/losses/ties。
3. 根据第 7 节四种情况选择状态策略。
4. 固定胜出的状态策略后，再对 V18 的 Reflective/Grounding 混淆矩阵做错误归因。
5. 只有在错误归因明确后，才设计下一版 Prompt；下一版必须写清楚只修哪一种错误，不允许把过去所有规则继续叠加。

## 10. 尚未证明的事项

以下内容目前都只能称为假设：

- 完整 empathy state 一定有害；
- V18 response Prompt 单独就能稳定复现 `0.7915` Reflective；
- 删除关系信息一定能改善 Empathy；
- 增加关系信息一定能改善个性化；
- 问句率降低一定能提高 Grounding；
- 单次温度 0 的三组差异具有统计显著性。

若受控三组差异较小，应考虑增加重复生成，而不是继续凭单次结果改 Prompt。
