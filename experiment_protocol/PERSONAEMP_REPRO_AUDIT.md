# PersonaEmp 复现审计

## 审计对象

- 论文：[From Empathy to Personalized Empathy](https://arxiv.org/abs/2606.00728)
- 官方代码：[ZhengWwwq/PersonalizedEmpathy](https://github.com/ZhengWwwq/PersonalizedEmpathy)
- 官方代码固定提交：`b555447f267b8057039aab39a4be44725718ea7f`
- 本项目上游基线：`Lemonlning-creator/memory@4911459`

## 论文确定的正式设置

- PersonaEmp 源自 AlpsBench/WildChat 的长期用户交互。
- Random Split 按用户 9:1 划分。
- OOD Split 根据 Big Five 人格特征做 KModes 聚类，并留出一个人格簇测试。
- 基础生成模型是 Qwen3-8B。
- 主指标是 Resonation、Expression、Reception 和三者 Average，范围 1–5。
- 固定 criteria 先由 DeepSeek 为每个查询生成。
- 同一份固定 criteria 分别交给 Qwen3-30B-A3B-Instruct 和 DeepSeek-v4-flash 评分。

## 官方仓库实际公开范围

官方仓库公开了：

- PersonaEmp 数据过滤、画像、情境和查询生成脚本；
- 三项查询质量检查；
- 固定 criteria 生成脚本；
- Resonation、Expression、Reception 的评分脚本；
- PereGRM 训练代码。

官方仓库没有公开：

- 论文使用的 `English.json`；
- Random/OOD 的固定样本 ID 或划分文件；
- Table 1 各 Baseline 的逐样本预测；
- Table 1 的固定 criteria 文件；
- Qwen3-8B、Memory、RAG 等方法的完整推理脚本；
- 数据生成所需的原始 `prepare_dataset/dataset/by_label_json/`。

因此，在获得官方数据和划分前，不能把新生成数据上的 Ours 分数直接作为 Table 1 的新增一行。

## 当前实施策略

1. 输入和输出严格兼容官方 PersonaEmp JSON 结构。
2. 先使用论文案例构造的 Pilot 数据完成流程测试，该结果不进入论文主表。
3. 官方数据到位后无需改代码，只替换 `--dataset`。
4. 若只能重新生成数据，则在同一新数据上补跑必要 Baseline。
5. 只有数据 SHA-256 与明确登记的 Table 1 数据指纹相同，运行清单才允许标记为可直接比较。
6. 官方 evaluator 不复制或改写，通过固定提交的外部 checkout 调用。

## 核心 Prompt 保护

Ours 的回复生成继续使用上游已有的：

- 五层画像提取 Prompt；
- Deep Empathy alignment Prompt；
- Direct Response System Prompt；
- Direct Response User Prompt。

生成阶段只把 PersonaEmp 的 `extracted_memory` 和 `query` 映射到现有字段。
Ours 从同一份 memory 证据生成五层画像，再执行 Deep Empathy 对齐。
数据集内的 `persona` 与 `scenario` 属于构造和评审元数据，不向任何生成方法披露；
它们只由固定 criteria 和官方 Judge 使用。

现有英文模板把变量拼写为 `DDIRECT_RESPONSE_SYSTEM_PROMPT`。适配器只兼容读取该变量，没有修改模板内容。核心 Prompt 和 Agent 行为文件仍由 `tools/verify_core_prompt_baseline.py` 校验。

## 当前可执行范围

- 可以验证 PersonaEmp 数据 Schema。
- 可以用 Qwen3-8B 生成 `base_qwen3` 和 `ours` 回复。
- 可以断点恢复，并记录网络重试、逻辑重试、Token 和延迟。
- 可以输出官方 evaluator 需要的 dataset/prediction 对。
- 可以调用固定版本官方脚本生成 criteria。
- 可以分别运行 Qwen 与 DeepSeek Judge，避免结果文件互相覆盖。
- 可以汇总 raw 1–5 和 normalized 0–1 两种结果。

## 尚缺资源

- PersonaEmp 正式 `English.json` 和 Random/OOD 划分；
- DeepSeek-v4-flash 的可用接口；
- Qwen3-30B-A3B-Instruct Judge 接口；
- 运行 Qwen3-8B 的本地权重或兼容 API。

其中 Qwen3-8B 和 Qwen Judge 可以通过兼容服务配置；具体平台只影响运行资源，不影响实验代码。
