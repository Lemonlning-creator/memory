# PersonaEmp Exp1 设计核对

核对依据：

- 师姐最新版 `Experiment Settings(1).docx` 开头的 Experiment 1；
- PersonaEmp 论文 arXiv:2606.00728v1；
- PersonaEmp 官方仓库固定提交 `b555447f267b8057039aab39a4be44725718ea7f`。

## 结论

当前实现的任务方向、生成输入边界、指标结构和官方评测对接是正确的；当前 Kimi 公开案例结果只是流程小测，还不是师姐设计要求的正式 Exp1 主结果。

## 逐项核对

| 师姐要求 | 当前状态 | 说明 |
|---|---|---|
| RQ1：Deep Empathy 是否提升个性化共情 | 已实现 | Base 与 Ours 在同一 query 上生成个性化共情回复。 |
| PersonaEmp 官方 Random/OOD split | 等待资源 | 当前只有论文公开案例；官方 `English.json` 和固定 split ID 尚未公开。 |
| 生成输入遵循 PersonaEmp | 已实现 | PersonaEmp 公式是 `y = policy(memory, query)`；Base/Ours 都只读取同一份 memory 与 query。 |
| Persona、Situation 不泄露给生成器 | 已实现 | 二者只进入固定 criteria 与 Judge，和论文评测公式一致。 |
| Table 1 全部 baseline 加 Ours | 未完成 | 当前适配器只有 Base 与 Ours；正式实验还需 Qwen3-8B、Memory、RAG、SFT、RLPA、PERM 等原表方法。 |
| Resonation、Expression、Reception、Average | 已实现接口 | 固定为官方 1--5 分；Average 为三项算术平均。 |
| Qwen 与 DeepSeek 双 Judge | 等待接口 | 当前只完成 Kimi Pilot Judge 链路，不能写入正式主表。 |
| 固定 criteria | 已实现 | 包装器调用固定官方提交；并增加生成后完整性校验，避免官方异步脚本把空 criteria 当成功。 |
| 复用 Table 1，仅增加 Ours | 当前不可执行 | 只有正式数据指纹、正式模型与双 Judge 全部一致时才能直接追加。当前清单会主动禁止错误声明。 |
| Qualitative Analysis | 已准备原始数据 | Ours 结果保存五层画像、理解、未来预测、主动探索和 empathy state，便于后续制作案例图。 |

## 当前 Pilot 结果

数据：PersonaEmp 论文公开案例中的 1 个 query。

Judge：Kimi K2.6 Pilot Judge，复用官方固定 criteria/Judge Prompt 和 1--5 分解析规则。

| 方法 | Res | Exp | Rec | Avg |
|---|---:|---:|---:|---:|
| Ours | 4.00 | 5.00 | 3.00 | 4.00 |
| Base Model | 3.00 | 3.00 | 3.00 | 3.00 |

这条案例中，Ours 的优势主要来自更准确地使用“小而深的社交圈”和“有限社交能量”；Reception 没有提升，是因为回复中的直接口头措辞和追问仍可能让内向用户感到压力。

样本量只有 1，且生成器与 Judge 都是 Kimi，因此只能说明流程正确并提供案例观察，不能说明统计显著性，也不能与 Table 1 横向比较。

## 正式实验还差什么

1. PersonaEmp 官方数据与 Random/OOD 固定划分。
2. 师姐指定的正式生成模型。
3. Table 1 全 baseline 的原始预测或在同一环境下重跑。
4. DeepSeek 生成的固定 criteria。
5. Qwen3-30B-A3B-Instruct 与 DeepSeek-v4-flash 两套 Judge 分数。

上述资源到位后，现有预测导出、固定 evaluator、汇总和可视化代码可以直接复用。
