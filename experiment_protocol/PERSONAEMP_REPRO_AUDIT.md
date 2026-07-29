# PersonaEmp Exp1 复现审计

## 正式方向

- 研究问题：Deep Empathy 是否能提升个性化共情回复。
- 数据：PersonaEmp Random/OOD split。
- 正式生成模型：按师姐最新设计指定；当前 Kimi 仅用于流程和行为小测。
- 方法：论文 Table 1 的全部 baseline 加 Ours。
- 指标：Resonation、Expression、Reception、Average。
- Judge：固定 criteria，分别由 Qwen 与 DeepSeek 评分。

## 当前代码已经保证

1. Base 与 Ours 的原始证据完全相同，只有 `extracted_memory + query`。
2. `persona`、`scenario/situation`、`category`、`conversation` 不进入任何生成 Prompt。
3. Ours 的五层画像只从 `extracted_memory` 派生，缓存键也不含隐藏对话。
4. Ours 的 Deep Empathy alignment 只读取共享 memory、当前 query 和派生画像，不使用外部 agent persona。
5. 两组最终回答使用完全相同的回复 system contract、temperature 和最大输出长度。
6. 画像预处理成本与在线推理成本分开记录。
7. 生产核心 Prompt 和 Agent 行为文件没有被修改，并由基线校验脚本保护。
8. 每次运行记录数据指纹、代码提交、Prompt 指纹、Token、延迟、重试和失败原因。

## 为什么不再直接使用生产 Direct Response Prompt

生产 Prompt 服务于持续陪伴系统，包含“不要完全解决问题”“优先维持对话”等产品行为。PersonaEmp 是单轮回复评测，部分问题明确要求建议、决策或措辞；直接使用生产 Prompt 会让 Ours 因任务合同不同而吃亏。

因此实验适配层增加共享的 PersonaEmp 回复合同，但不改生产 Prompt。Base 和 Ours 都遵守它，所以人设表达仍可自然温暖，同时不再把“刻意留问题到下一轮”误当成共情不足。

## 与官方论文仍有的边界

官方仓库未公开论文使用的 `English.json`、Random/OOD 固定样本 ID、Table 1 的逐样本预测和固定 criteria。未取得这些资源前：

- 不能把 Pilot 或重新生成数据上的 Ours 分数直接追加到原 Table 1；
- 可以验证输入输出、运行稳定性、官方 evaluator 对接与定性效果；
- 若只能重新生成数据，所有需要比较的方法必须在同一新数据和同一 Judge 条件下重跑。

当前适配器不是完整 Table 1 复现器：它先实现并验证 `base_model` 与 `ours`。其余 baseline 和正式双 Judge 流程要在数据与模型资源确定后补齐。
