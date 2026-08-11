# REALTALK Task 1：Ours 三 Session 正式执行锚点

版本日期：2026-08-11  
实验状态：正式生成已启动  
实现提交：`a642f8247b127ebeb877f995f9f759224310c1f1`

## 1. 唯一正式目标

在 REALTALK Persona Simulation（Task 1）协议下，仅运行一行 `Ours / qwen3-8b`，并与论文 Table 2 的两行结果并列展示。

本次不训练、不微调、不运行论文基线，不运行 Memory Probing 或未来用户理解任务。

## 2. 最终固定条件

```text
目标人物数                 10
Ca 信息范围                按时间排列的前 3 个 Session
Cb 测试范围                按时间排列的前 3 个 Session
消息单位                   同 Session 连续同说话者气泡合并后的消息
重建测试目标               519
生成模型                   qwen3-8b
Thinking                   关闭
Omega                      禁用
Verification/重写          禁用
回复句数限制               无
历史推进                   只使用真实消息，不回灌模型输出
```

## 3. 论文协议证据

### 3.1 论文明确规定

- 每位参与者拥有两段不同伙伴的对话 `Ca` 和 `Cb`。
- 整段对话的 speaker-level EI 较低者被指定为 `Cb`。
- `w/o fine-tuning` 在 `Cb` 测试；`w/ fine-tuning` 在 `Ca` 训练、在 `Cb` 测试。
- 每位目标人物独立训练和测试。
- 预测目标人物下一条消息 `M_t`，输入是此前真实历史 `H_t`。
- 目标人物原始消息是 Ground Truth。
- 同一说话者连续发送的气泡在分析前合并。
- 不同历史长度实验中，训练与测试使用相同数量的 Session。
- 三个 Session 后指标趋于饱和；Table 2 报告三 Session 主条件。
- Table 2 由 Appendix E.2 / Table 8 的逐人物结果平均得到。

### 3.2 高置信协议重建

论文未发布 Persona Simulation 构造脚本，也没有逐字写出“first three sessions”。本实验按时间取前三个 Session，依据是：

1. `H_t={M_1,...,M_{t-1}}` 定义的是时间前缀；
2. Figure 7 将 Session 数逐步增加，语义上对应累计历史；
3. Akib 的 Figure 7 横轴与公开 JSON 按时间取前 N 个 Session、合并连续消息后的累计数量吻合；
4. 论文没有报告随机 Session 抽样、seed 或非连续选择规则。

因此“时间前 3 个 Session”是当前最接近论文且可复查的重建方式，但不宣称为官方脚本级复现。

## 4. 数据与角色

论文 Table 8 的 speaker-specific Ca/Cb 分配保持不变：

| 目标人物 | Ca：建立 Self Domain | Cb：正式测试 | 测试伙伴 | Cb 目标数 |
|---|---|---|---|---:|
| Emi | Chat 4 Emi-Paola | Chat 1 Emi-Elise | Elise | 37 |
| Nicolas | Chat 5 Nicolas-Nebraas | Chat 6 Vanessa-Nicolas | Vanessa | 117 |
| Kevin | Chat 3 Kevin-Paola | Chat 2 Kevin-Elise | Elise | 25 |
| Akib | Chat 9 Fahim-Akib | Chat 8 Akib-Muhhamed | Muhhamed | 37 |
| Muhhamed | Chat 10 Fahim-Muhhamed | Chat 8 Akib-Muhhamed | Akib | 37 |
| Nebraas | Chat 5 Nicolas-Nebraas | Chat 7 Nebraas-Vanessa | Vanessa | 51 |
| Paola | Chat 4 Emi-Paola | Chat 3 Kevin-Paola | Kevin | 23 |
| Vanessa | Chat 7 Nebraas-Vanessa | Chat 6 Vanessa-Nicolas | Nicolas | 116 |
| Elise | Chat 2 Kevin-Elise | Chat 1 Emi-Elise | Emi | 36 |
| Fahim Khan | Chat 10 Fahim-Muhhamed | Chat 9 Fahim-Akib | Akib | 40 |

合计 `519`。论文没有公布 Table 2 的官方精确样本数，因此 519 是公开数据协议重建值。

> 注：代码中的 Kevin Ca 文件为论文 Table 8 已核定映射；运行 manifest 是最终机器可审计来源。文档表格不得替代 manifest。

## 5. 一个测试样本如何产生

对目标人物 `S` 的一条真实合并消息 `M_t`：

```text
可见输入：Cb 选定三 Session 中，严格位于 M_t 之前的全部真实合并消息
隐藏答案：M_t
模型输出：预测的目标人物消息 M_hat_t
评分阶段：比较 M_hat_t 与 M_t
```

- 只有目标人物的消息产生测试样本；伙伴消息只进入历史。
- 当前答案和未来消息不可见。
- 后续样本继续使用真实历史，不使用先前生成文本。
- 不跨 Session 合并消息。

## 6. Ours 实现

### 6.1 Self Domain：每人一次

输入：目标人物 Ca 前 3 个 Session，保留双方上下文。  
证据边界：只有目标人物自己的发言可支持其稳定事实。  
输出：Identity、Persona、Behavior Policy Prior、Hard Constraints、Uncertainties。  
测试期间固定，不读取 Cb 答案更新。

### 6.2 User Domain：因果滚动

输入：Cb 当前时刻前已经出现的伙伴真实消息。  
输出：Core、Regulation、Cognition、Identity、Behavior 五层伙伴模型。  
没有新的可靠伙伴证据时，直接复用，不调用模型。

### 6.3 Alignment：每条样本一次

输入：Self Domain、User Domain、当前真实历史。  
一次输出：Current/Future User State、动态 `lambda_t`、权衡依据、唯一 Behavior Policy。

`lambda_t` 越高，当前回复越重视适应伙伴；Self Domain 始终保持目标人物身份硬约束。

### 6.4 Generation：每条样本一次

系统约束：

```text
You are {speaker}. Continue the conversation.
Output only the message, not the speaker name.
```

输出纯文本目标人物消息。不得输出 JSON、画像、策略说明或说话者名字。

## 7. 明确禁用

- `Omega` 与主动探索控制；
- Verification 或额外重写调用；
- 多候选 Behavior Policy；
- 强制共情、建议、追问或积极表达；
- 1–2 句、2–4 句等长度限制；
- 用完整 Ca 替代三 Session 主条件；
- 任何 Cb 未来信息或模型输出回灌。

## 8. 模型与可靠性

- 所有 Ours LLM 阶段固定为精确 ID `qwen3-8b`；
- thinking 关闭；
- Self/User/Alignment 使用严格 JSON Schema；
- 结构化调用最多 3 次，后两次只修复格式或 Schema；
- 每个阶段按确定性 operation key 缓存，支持幂等续跑；
- 正式完成要求 `519/519` 且零 unresolved；
- CPU-only；密钥只从服务器独立环境文件读取。

## 9. 评价

本地五项：

1. ROUGE-L；
2. BERTScore F1；
3. Sentiment Accuracy；
4. Emotion Accuracy；
5. Intimacy Absolute Difference。

待 `gpt-4o-mini` 端点补齐：

6. Reflectiveness Accuracy；
7. Grounding Accuracy；
8. Empathy Absolute Difference。

聚合顺序：逐消息计算，先求每位人物均值，再对 10 位人物做 macro mean 与 population standard deviation。

没有三项 GPT 指标时，只允许生成 `GENERATION_COMPLETE` 和 `LOCAL_METRICS_COMPLETE`，不得生成最终 `PIPELINE_COMPLETE`。

## 10. 当前正式运行

```text
screen:
realtalk-ours-qwen3-8b-merged-full-v1

output:
/amax/xidian_ty/Ly/personaemp-exp2/runs/
realtalk-ours-qwen3-8b-merged-full-v1-a642f82
```

错误的 1,076 原始气泡版本已停止并隔离，仅保留为诊断，不得并入正式结果。

## 11. 最终报告措辞

Ours 可以放在论文 Table 2 两行之后，但必须注明：

- 使用相同公开数据、Table 8 Ca/Cb、三 Session 条件、消息合并规则和评价维度；
- Persona Simulation 官方构造代码及基础模型未公开；
- Ours 使用 qwen3-8b 和显式 Self/User Domain，不等同于论文的 Ca 微调；
- 结果属于 protocol-aligned comparison，不宣称官方运行时精确复现。
