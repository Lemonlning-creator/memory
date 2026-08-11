# Exp2 / REALTALK 输入输出与评价协议审计

状态：实验协议确认稿 v1（尚未冻结最终 Ours 适配）  
前置方法定义：`OURS_METHOD_ANCHOR_ZH.md`  
目的：先固定 REALTALK 论文真实任务、公开数据、输入、输出和评价，再决定 Ours 如何适配。

## 1. 依据与可复现边界

### 1.1 主要依据

- REALTALK 论文：`REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation`；
- 论文重点位置：Section 4、Section 6.1、Appendix C、Appendix D.1、Appendix E.2/E.3；
- 官方仓库：`https://github.com/danny911kr/REALTALK`；
- 已审计官方仓库 commit：`b903e06a9770bf4e5fe9018c3e132889666d3b4a`；
- 师姐实验要求：`Experiment Settings(exp2).docx`；
- Ours 理论依据：师姐《理论.pptx》和 `OURS_METHOD_ANCHOR_ZH.md`。

### 1.2 官方公开情况

官方仓库公开了：

- 10 个预处理 REALTALK JSON 对话；
- 原始 XLSX 和图片；
- 情绪、情感、亲密度分类器调用代码；
- Reflectiveness、Grounding、Empathy 的评价 Prompt 和代码；
- Memory Probing 代码。

官方仓库没有公开：

- Persona Simulation 的样本构造代码；
- Persona Simulation 使用的基础生成模型；
- 微调方式、超参数和 checkpoint；
- 生成 temperature、top-p、max tokens、seed；
- ROUGE 与 BERTScore 的具体软件版本和完整配置。

官方 README 在 Persona Simulation 处仍标记 `Coming soon`。因此 Table 2 的数据与评价逻辑可以按论文重建，但不能声称逐代码、逐 checkpoint 完全复现。

## 2. REALTALK 数据结构

数据集包含：

- 10 个真实人类参与者；
- 每人分别与两个不同伙伴对话；
- 10 条长对话；
- 219 个 Session；
- 8,944 条原始消息；
- 平均每条对话 21.9 个 Session、894.4 条原始消息。

本地 `dataset/Chat_*.json` 已与上述官方仓库 commit 的对应 JSON 做过字节级核对，数据源一致。

Persona Simulation 采用公开 JSON 的原始消息气泡作为 `M_t`，不合并连续同说话者消息。论文没有公开样本构造代码，但 Appendix D.1 将真值定义为 speaker's original message；更关键的是 Figure 7 中 Akib 的逐 Session 累计消息量与原始气泡计数吻合，而与合并 turn 计数明显不符。

## 3. REALTALK Task 1 的真实任务

Task 1 名为 Persona Simulation。

对目标说话者 `S` 的某条真实消息 `M_t`：

```text
输入：H_t = {M_1, ..., M_(t-1)}
输出：预测的目标说话者下一条消息 M_hat_t
真值：目标说话者的原始消息 M_t
```

论文 Appendix D.1 的系统 Prompt 为：

```text
You are {speaker}. Continue the conversation.
Output only the message, not the speaker name.
```

User Prompt 为此前对话历史，末尾给出目标说话者 cue。Assistant Ground Truth 是该说话者的原始消息。

因此 Table 2 评价的是：模型能否扮演某个真实参与者，并生成该参与者下一条真实消息。它不是“AI Agent 读取当前用户消息后回复用户”的任务。

## 4. Ca / Cb 划分

每位参与者有两条不同伙伴的对话：

- `Ca`：建模或微调对话；
- `Cb`：测试对话；
- 论文选择该参与者总体 EI 较低的对话作为 `Cb`；
- 每个参与者单独训练、单独测试，避免说话者跨集合泄漏。

论文 Table 8 给出的映射如下：

| 目标说话者 | Ca | Cb |
|---|---|---|
| Emi | Emi-Paola | Emi-Elise |
| Nicolas | Nicolas-Nebraas | Vanessa-Nicolas |
| Kevin | Kevin-Paola | Kevin-Elise |
| Akib | Fahim-Akib | Akib-Muhhamed |
| Muhhamed | Fahim-Muhhamed | Akib-Muhhamed |
| Nebraas | Nicolas-Nebraas | Nebraas-Vanessa |
| Paola | Emi-Paola | Kevin-Paola |
| Vanessa | Nebraas-Vanessa | Vanessa-Nicolas |
| elise | Kevin-Elise | Emi-Elise |
| Fahim Khan | Fahim-Muhhamed | Fahim-Akib |

论文报告模型表现约在 3 个 Session 后饱和，并以 3 个 Session 的微调模型和非微调模型进行 Table 2 比较。论文没有公开 Persona Simulation 构造代码，因此“Ca 前 3 个 Session 建模，Cb 前 3 个 Session 测试”是最接近论文叙述的重建规则，而不是已公开官方脚本事实。

## 5. 公开历史是否足够生成 Profile 和 Persona

按 Ca 前 3 个 Session 的原始目标说话者消息统计：

| 目标说话者 | 目标说话者原始消息数 | 目标发言字符数（约） |
|---|---:|---:|
| Emi | 37 | 7,534 |
| Nicolas | 163 | 6,099 |
| Kevin | 39 | 6,882 |
| Akib | 121 | 8,698 |
| Muhhamed | 51 | 6,321 |
| Nebraas | 148 | 6,341 |
| Paola | 29 | 7,136 |
| Vanessa | 85 | 5,326 |
| elise | 36 | 4,167 |
| Fahim Khan | 73 | 6,710 |

结论：

- 足以生成每位目标说话者的初始 Profile 和说话风格 Persona；
- 不足以保证五层画像的每一层都存在可靠信息；
- Profile Prompt 必须允许某层为空，禁止为了完整度强行推断；
- `core`、`identity` 等深层字段只接受目标说话者自己的证据；
- Profile 的 confidence 和 evidence 必须保存；
- 这些是初始模型，不应描述为完整、真实的人格全貌。

按 Cb 前 3 个 Session 构造测试集，可得到 1,076 条目标说话者原始消息。论文没有公布 Table 2 的准确测试样本总数，所以 1,076 是根据公开数据、Table 8、三 Session 描述和 Figure 7 数量交叉验证得到的重建统计。

用于 Ours 双域建模的数据覆盖如下。`Ca Self messages` 是目标说话者在 Ca 前 3 个 Session 的原始消息；`Cb Partner messages` 是测试伙伴在 Cb 前 3 个 Session 中最终可被因果观察到的原始消息总数：

| 目标说话者 | Ca Self messages | Cb 测试伙伴 | Cb Partner messages |
|---|---:|---|---:|
| Emi | 37 | elise | 55 |
| Nicolas | 163 | Vanessa | 182 |
| Kevin | 39 | elise | 36 |
| Akib | 121 | Muhhamed | 75 |
| Muhhamed | 51 | Akib | 157 |
| Nebraas | 148 | Vanessa | 85 |
| Paola | 29 | Kevin | 39 |
| Vanessa | 85 | Nicolas | 247 |
| elise | 36 | Emi | 52 |
| Fahim Khan | 73 | Akib | 121 |

所有 Ca/Cb 文件都至少包含 18 个 Session，因此每个目标都满足前 3 个 Session 的建模和测试要求。Ca 的 Self Domain 证据量足够生成初始 Persona；Cb 伙伴在 3 个 Session 结束时有 36--247 条原始消息可支持 User Domain。测试最早位置仍可能只有 0--1 条伙伴发言，这是协议本身的冷启动状态，不能读取未来消息补齐。

每位测试伙伴作为另一位参与者，也拥有论文 Table 8 指定的个人 Ca。技术上可以用该 Ca 预生成伙伴 User Domain，但这会让当前目标实验额外使用一份论文目标说话者微调基线没有使用的跨对话信息。直接与 Table 2 比较的主设置应采用：

```text
Self Domain：目标说话者自己的 Ca 前 3 Session
User Domain：当前 Cb 中目标位置以前已观察到的伙伴发言，冷启动并滚动更新
```

使用伙伴个人 Ca 预初始化 User Domain 只能作为单独的 enhanced-data analysis，必须披露额外信息预算，不能替代主设置。

## 6. REALTALK Table 2 的最终输入和输出

### 6.1 每个测试样本输入

```json
{
  "target_speaker": "S",
  "history": [
    "Cb 前 3 个 Session 中严格位于 M_t 之前的所有原始消息"
  ],
  "speaker_cue": "S"
}
```

要求：

- `M_t` 不进入输入；
- `M_t` 后面的消息不进入输入；
- 每条样本重新使用真实历史 `H_t`；
- 上一个模型生成结果不回灌到后一个样本；
- 历史是随目标位置增长的 causal prefix；
- 图片使用公开 JSON 中已有的文本化内容；
- 不对历史进行与论文无关的摘要替换。

### 6.2 每个测试样本输出

```text
目标说话者 S 的一条预测消息 M_hat_t
```

只能输出消息正文，不输出说话者名称、JSON 或分析。

### 6.3 Ground Truth

```text
公开数据中目标说话者 S 在位置 t 的真实原始消息 M_t
```

## 7. Table 2 评价标准

预测消息与真实消息分别经过同一评价流程。

| 指标 | 计算 | 方向 |
|---|---|---|
| Lexical | ROUGE | 越高越好 |
| Semantic | BERTScore | 越高越好 |
| Reflective | 预测/真值二分类标签一致率 | 越高越好 |
| Grounding | 预测/真值二分类标签一致率 | 越高越好 |
| Sentiment | 预测/真值三分类标签一致率 | 越高越好 |
| Emotion | 预测/真值情绪分类标签一致率 | 越高越好 |
| Intimacy | 预测与真值分数绝对差 | 越低越好 |
| Empathy | 预测与真值 EPITOME 总分绝对差 | 越低越好 |

评价器：

- Reflectiveness：`gpt-4o-mini`，使用 Appendix C.1 / 官方代码 Prompt；
- Grounding：`gpt-4o-mini`，使用 Appendix C.2 / 官方代码 Prompt；
- Empathy：`gpt-4o-mini`，使用 Appendix C.3 EPITOME Prompt；
- Sentiment：`cardiffnlp/twitter-roberta-base-sentiment-latest`；
- Emotion：`cardiffnlp/twitter-roberta-large-emotion-latest`；
- Intimacy：`cardiffnlp/twitter-roberta-large-intimacy-latest`。

正式结果应按参与者先汇总，再计算 10 位参与者的 speaker-macro 平均和标准差，以对应 Table 2 `mean +/- std`。同时保存 message-micro 结果作为诊断，但不替代主表。

Table 2 参考值：

| 方法 | Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o fine-tune | 0.14 | 0.76 | 0.62 | 0.40 | 0.53 | 0.43 | 0.06 | 1.80 |
| w/ fine-tune | 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.07 | 1.24 |

## 8. 师姐 Exp2 要求中的三个不同输出任务

当前材料同时出现了三个不同任务，必须分开：

### 8.1 当前用户理解

输入：历史 + 当前真实用户消息。  
输出：Emotion、Sentiment、Topic 等结构化状态。  
特点：当前消息可见，不生成 Table 2 的下一条消息。  
可比性：不是 REALTALK Table 2 原任务，必须单独成表。

### 8.2 未来用户理解 / Persona Simulation

输入：目标消息之前的历史。  
输出：目标用户的下一条预测消息。  
特点：当前目标消息不可见。  
可比性：这才是 REALTALK Task 1 / Table 2。

### 8.3 完整 Companion Agent 回复

输入：历史 + 当前真实用户消息 + User Domain + Self Domain。  
输出：Agent 对用户的回复。  
特点：可以完整运行 adaptive `lambda_t` 和 Behavior Policy。  
可比性：Ground Truth 应是公开对话中的下一条伙伴消息；这不是 Table 2 的目标消息，因此不能直接把结果追加到 Table 2。

## 9. Ours 在 Table 2 中的角色映射

REALTALK 没有固定的“AI Agent”角色。模型被要求扮演当前目标说话者 `S`，因此可以将该说话者视为本样本中的 simulated agent：

```text
目标说话者 S
  = 模型当前扮演的角色
  = Ours Self Domain 的主体

测试对话伙伴 P
  = Ours User Domain 的主体

S 的真实下一条消息 M_t
  = Ours 的生成 Ground Truth
```

例如 Emi 的 Table 8 实验：

```text
Emi 的 Ca 发言 -> Emi Self Domain
Cb 中截至当前目标前已观察到的 Elise 发言 -> Elise User Domain
当前因果历史 -> Elise User State
Emi Self Domain + Elise User Domain/State
-> adaptive lambda_t -> Emi Behavior Policy
-> 生成 Emi 的下一条消息
```

该映射没有把 Emi 同时作为 Self Domain 和 User Domain，也没有改变 Table 2 的输出对象。它将“扮演真实参与者”解释为创建一个具有该参与者 Self Domain 的模拟 Agent。

## 10. Table 8 是否双向测试

论文不是规定每条对话都机械地交换身份测两次。真实规则是：

- 10 位参与者分别作为目标说话者运行一次 speaker-specific 实验；
- 每位参与者依据其两条对话的 EI 选择一条 Ca 和一条 Cb；
- 因此每位参与者在主结果中恰好对应一行 Table 8 speaker result；
- 当同一条对话恰好同时被双方选作各自 Cb 时，该对话会以相反目标角色分别使用两次；
- 如果双方没有都选它作为 Cb，则该对话不会自动双向测试。

公开 Table 8 中明确双向作为 Cb 使用的例子包括：

- `Emi-Elise`：Emi 和 elise 分别作为目标；
- `Vanessa-Nicolas`：Vanessa 和 Nicolas 分别作为目标；
- `Akib-Muhhamed`：Akib 和 Muhhamed 分别作为目标。

其他目标说话者仍按各自 Table 8 的 Ca/Cb 独立运行，不额外添加论文未报告的交换组。

## 11. Ours 的 Table 2 适配输入

### 11.1 Self Domain

使用目标说话者 `S` 的 Ca 前 3 个 Session 中由 `S` 自己说出的内容，结合对话上下文，生成 `S` 的 Persona、Behavior Policy Prior、表达风格和边界。它与论文 `w/ fine-tune` 使用相同来源和近似相同 3-Session 信息预算，但 Ours 将信息显式建模而不是更新模型权重。

### 11.2 User Domain

主体是 Cb 中的对话伙伴 `P`。为避免获得论文基线没有使用的额外跨对话信息，正式主实验只使用当前 Cb 目标位置之前已经出现的 `P` 发言因果更新 User Domain，不读取 `P` 的另一条对话。

### 11.3 User State 与 query

使用 `M_t` 之前的 Cb 真实历史推断 `P` 的 Current User State。通常最新一条 `P` 消息是当前 query；若 `S` 主动开启新 Session、目标前没有新的 `P` 消息，则 query 为空，继承此前状态但提高不确定性，不能虚构一个用户请求。

### 11.4 Alignment 和输出

```text
S Self Domain
+ P User Domain
+ P Current User State
+ H_t
-> adaptive lambda_t
-> one Behavior Policy for S
-> M_hat_t spoken as S
```

最终输出仍严格遵守 Appendix D.1：只输出目标说话者消息正文，并与 `M_t` 使用 Table 2 的八项指标比较。

## 12. 当前结论

通过“目标说话者 = simulated agent / Self Domain，测试伙伴 = User Domain”的明确映射，完整 Ours 可以在不改变 Table 2 输出对象的前提下运行。

但应披露两点：

- REALTALK 官方没有发布该 Ours 映射，它是依据 Ours 框架设计的 Benchmark Adapter；
- 论文 Persona Simulation 代码、基础生成模型和微调配置未公开，所以数据逻辑与评价可以紧密对齐，但不是官方运行时的逐代码复现。

如果当前完全不运行下一消息生成，就只能进行第 8.1 节当前用户理解，无法得到 Table 2 的 Ours 行；要追加 Table 2，必须运行上述下一消息生成协议。
