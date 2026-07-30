# PersonaEmp 公开复现实验实现核对与测试报告

## 1. 一眼结论

当前实现与已确认的正确方案**总体一致，可以作为 PersonaEmp 非训练方法的公开数据受控复现框架**。

最关键的实验边界已经正确实现：

- 四种方法的共同原始信息都是 `extracted_memory + query`。
- Base 使用完整 memory；Memory 使用由完整 memory 生成的扁平摘要；RAG 从完整 memory 检索 Top-3；Ours 从完整 memory 生成五层画像和深度共情状态。
- 数据集自带的 `persona`、`situation`、`category`、原始 `conversation` 不进入四种方法的回复生成。
- `persona + situation + memory + query + response` 只交给官方 Judge。
- `relevant_mem` 不参与 RAG 检索，只用于事后计算 Recall@3。
- 四种方法共用模型、回复 system prompt、温度和回复长度要求。
- Ours 的生产画像、理解、预测、探索和共情对齐 prompt 没有改写。

但当前不能表述为“已经复现官方 Table 1”，原因是：

1. 官方最终 `English.json`、精确原始子集和逐样本结果没有公开，只能基于 AlpsBench 重建新数据。
2. Base、Memory、RAG 的作者原始生成脚本没有公开，目前是根据论文定义补全的实现。
3. 现有 20 项是代码协议测试，不是 20 条正式模型样本，也不是官方 Res/Exp/Rec 实验结果。
4. 目前只完成 1 条 synthetic fixture 的真实 Kimi 四方法烟测；正式 12 条平衡样本、Random/OOD 各 30 条和双 Judge 尚未运行。

因此最准确的项目名称是：

> PersonaEmp 公开数据受控复现实验，而不是官方绝对分数复现。

## 2. 数据和实验流程

```text
AlpsBench 原始历史对话
        ↓ 官方 gold memory
extracted_memory（结构化记忆）
        ↓ 官方 PersonaEmp 数据流水线
persona + situation + ES/HEQ/SS query
        ↓ 只把 memory + query 交给生成方法
Base / Memory / RAG / Ours
        ↓
Judge 读取 memory + persona + situation + query + response
        ↓
Resonation / Expression / Reception / Average
```

这里的两种 Profile 必须区分：

- **数据集 persona**：官方数据流水线生成，供 Judge 制定 criteria 和评分，不给生成模型。
- **Ours 五层 Profile**：Ours 根据与 baseline 相同的 `extracted_memory` 自行生成，是被测方法的一部分。

## 3. 正确方案逐项对照

| 核对项 | 正确方案 | 当前实现 | 结论 |
|---|---|---|---|
| 原始公开数据 | AlpsBench Task 1 dev + validation | 已固定公开 revision，下载 466 + 466 条 | 一致 |
| 结构化记忆 | 使用公开 gold memory | 按 `benchmark_id` 连接 reference，共检查到 2,564 条 memory | 一致 |
| Intent | 使用官方 allowlist 补建 `intents_ranked` | 严格 Schema、模型/prompt/数据版本缓存 | 一致 |
| 官方数据流水线 | filter → persona → situation/query → inspection → final filter | 固定官方 commit `b555447`，在隔离 worktree 运行 | 一致 |
| HEI/HEQ | 统一为同一类别 | 统一映射为 High-EQ Interaction / HEQ 统计 | 一致 |
| Base | 完整 memory + query | `BASE_MODEL_USER_PROMPT` 同时包含两者 | 一致 |
| Memory | 完整 memory → 普通扁平摘要 → query | 两次调用；摘要不使用五层结构、预测或探索 | 一致 |
| RAG | query 检索 Top-3 memory | 固定 `intfloat/e5-base-v2`、归一化 cosine、Top-3 | 一致 |
| Ours | memory → 五层画像 → 理解/预测/探索/共情对齐 → 回复 | 保留当前完整链路，不启用 Bayesian Updating | 一致 |
| 数据集 persona | 不给生成方法 | runner 明确排除 | 一致 |
| situation/category | 不给生成方法 | runner 明确排除 | 一致 |
| conversation | 不给生成方法 | 只保存在重建数据中，生成时排除 | 一致 |
| relevant_mem | 只作 RAG 诊断 | 不参与排序，只计算 Recall@3 | 一致 |
| 共同回复约束 | 四方法相同 | 共用同一 system prompt、温度和 max tokens | 一致 |
| Random | 用户级 9:1 | seed 42、用户不交叉 | 一致 |
| OOD | Big Five 三档 + KModes + 留出人格簇 | K=2..8、Hamming silhouette、最远簇主测试、另存 LOCO | 一致 |
| Criteria | 每题生成一次，四方法共享 | 调用固定官方 `prepare_criteria.py` | 一致 |
| Judge | Qwen3 与 DeepSeek 双 Judge | 官方评价 wrapper 已实现 | 一致，但尚未真实运行 |
| 主指标 | Res、Exp、Rec、Avg，1--5 越高越好 | 主报告只展示这四项 | 一致 |
| 官方参考值 | 与重建实验分表展示 | 单独输出官方参考 CSV，不与实测表拼接 | 一致 |
| 成本 | 离线画像/摘要/embedding 与在线回复分开 | summary 中分项记录 tokens、延迟和 calls | 一致 |
| 可恢复运行 | checkpoint 后跳过成功样本 | 稳定 sample id + JSONL checkpoint | 一致 |

## 4. 无法完全等同官方实验的部分

| 项目 | 当前处理 | 对结论的影响 |
|---|---|---|
| 官方最终 `English.json` 未公开 | 使用 932 条公开 AlpsBench 输入和 gold memory 重新生成 | 四方法内部公平，但样本和官方 Table 1 不同 |
| 作者精确原始子集未公开 | 使用公开 dev + validation，不使用无 gold memory 的 test | 不能声称逐样本复现 |
| 作者 Random/OOD ID 未公开 | 按论文规则重新划分并冻结 manifest | 具体测试用户不同，协议相同 |
| 官方 criteria 文件未公开 | 用固定官方脚本重新生成，四方法共享 | 绝对分数可能变化，内部比较仍公平 |
| Memory/RAG 原始 prompt 未公开 | 按论文功能定义实现并记录 prompt hash | baseline 强弱可能受实现选择影响 |
| 官方逐样本回复和评分未公开 | 自行重跑并保存 raw results | 不能核对官方每一题，只能比较总体趋势 |

## 5. Ours 核心 Prompt 核对

基线：`5de4271`  
当前提交：`e81f7c4`

| Prompt | 当前 SHA256 | 与基线相比 |
|---|---|---|
| 共同回复 system prompt | `78aff91c0024678864b4381b2629d3c1bcdcd140d66bdfe9df46c33e03d8af93` | 未改变 |
| Ours 回复 user template | `23d4afc0cff5c16ba000e2dddadfb80c408097bb7630a97610fe14906d31340a` | 未改变 |
| 五层画像 system prompt | `580bcce2ebdc97e9861ecaf74cd34e3e3d14a24536eca8aeab951809d6479724` | 未改变 |
| 五层画像 user template | `4e9023f5de8af87cd1e4dc3a14ace141e2586361f893335f38a995ebbbca5dff` | 未改变 |
| 共情对齐 system prompt | `5f7b8de42ddbc39651dcbb7519bfd0bdf81eb464b4864e242d7fd0369ec80f18` | 未改变 |
| 共情对齐 user template | `5bcad64f641c4ce1a94b2ddaef4bad6eeb83283a1a14af64081b234c4f7f1a00` | 未改变 |

其中后四项定义在生产文件 `src/prompts/templates_en.py`；该文件相对基线完全没有修改。

## 6. 20 项自动化测试

运行命令：

```powershell
python -m pytest tests/test_personaemp_exp1.py `
  tests/test_personaemp_public_reproduction.py -vv
```

总结果：

```text
collected 20 items
20 passed in 2.59s
通过率：100%
```

> 以下测试的“指标”主要是协议断言、计数和合成数据结果，不是正式论文指标。

| # | 测试内容 | 核心检查与结果指标 | 结果 |
|---:|---|---|---|
| 1 | 鉴权错误重试策略 | HTTP 401 不重试；HTTP 429 可重试 | 通过 |
| 2 | Kimi K2.6 非思考参数 | `temperature=0.6`；`thinking.type=disabled` | 通过 |
| 3 | 官方数据形状读取 | 1 个 session、3 个 query、SHA256 长度 64 | 通过 |
| 4 | 重复 query id 拒绝 | 构造重复 id 后必须抛出数据错误 | 通过 |
| 5 | checkpoint 恢复 | 第二次运行成功数仍为 1，LLM 调用数不增加，prediction 仅 1 条 | 通过 |
| 6 | Schema 重试成本累计 | 2 次逻辑调用、3 次底层 attempts、15 prompt tokens、5 completion tokens、0.3 秒 | 通过 |
| 7 | Ours 信息隔离和共同回复约束 | Base/Ours 共用 system prompt、温度和长度；Ours 有画像/预测/探索；persona/situation/category/conversation 均未泄露 | 通过 |
| 8 | Criteria 对齐 | 3 条完整 criteria 可通过；错 query id 必须拒绝 | 通过 |
| 9 | Prediction 与官方评分汇总 | 1 session、3 query；合成 Res=4、Exp=3、Rec=5，Avg=4.0，归一化 0.8 | 通过 |
| 10 | AlpsBench 适配 | 1 条输入正确连接 1 条 gold memory；生成合法 intent；不提前产生 persona/situation | 通过 |
| 11 | 平衡抽样不足时拒绝 | fixture 中 SS 只有 3 条，要求每类 4 条必须报错，不伪装成 12 条平衡样本 | 通过 |
| 12 | HEI/HEQ 统一及 relevant memory | `HEI` 归一为 High-EQ Interaction；索引清洗为 `(2,3)` | 通过 |
| 13 | Intent 缓存失效 | 同模型重复调用只请求 1 次；模型从 A 变 B 后必须重新请求 1 次 | 通过 |
| 14 | Kimi 严格结构化输出 | 使用强制 function tool；存在 `tools/tool_choice`，不使用被 Kimi 忽略的 `response_format` | 通过 |
| 15 | Memory baseline 隔离 | 共 2 次调用：先摘要再回复；回复含摘要和 query，不含数据集 persona/situation | 通过 |
| 16 | OOD 用户级划分 | 12 个合成用户，K 搜索 2..5；train/test 均非空、无交集、并集覆盖全部用户 | 通过 |
| 17 | 用户级配对 Bootstrap | 2 个用户；Ours 合成增益 `+1.0`；95% CI `[1.0,1.0]` | 通过 |
| 18 | RAG Top-3 和防泄露 | 固定检索结果 `[2,3,4]`；对 gold `[1]` 的 Recall@3 为 0；回复 prompt 不含 persona | 通过 |
| 19 | Random 用户级 9:1 | 20 用户 → 18 train / 2 test；重复运行一致；无用户交叉 | 通过 |
| 20 | 四方法报告产出 | 合成 Ours Avg 提升 `+0.5`；成功生成中文 MD、PNG、总汇 CSV、用户级 CSV、官方参考 CSV | 通过 |

## 7. 真实 Kimi 单样本烟测

配置：

- 模型：`kimi-k2.6`
- 中国区 URL：`https://api.moonshot.cn/v1`
- 数据：本地 synthetic fixture
- 样本：1 个 query
- 方法：Base、Memory、RAG、Ours

结果：

```text
成功回复：4 / 4
失败回复：0
```

### 在线回复成本

| 方法 | 在线逻辑调用 | Prompt tokens | Completion tokens | 总 tokens | 延迟 |
|---|---:|---:|---:|---:|---:|
| Base | 1 | 301 | 135 | 436 | 4.6536 s |
| Memory | 1 | 291 | 104 | 395 | 3.3974 s |
| RAG | 1 | 298 | 116 | 414 | 8.5233 s |
| Ours | 2 | 3,244 | 1,371 | 4,615 | 39.8615 s |

### 离线预处理成本

| 项目 | 逻辑调用 | Prompt tokens | Completion tokens | 延迟 |
|---|---:|---:|---:|---:|
| Memory 扁平摘要 | 1 | 136 | 48 | 3.6917 s |
| Ours 五层画像 | 1 | 369 | 763 | 45.2860 s |
| RAG memory embedding | 0 次 LLM | 0 | 0 | 本地完成 |

人工检查结论：

- 四个方法都正常回答当前 query，没有空回复或格式错误。
- Base、Memory、RAG、Ours 均未显式提及隐藏画像、memory 或生成流程。
- Ours 更明确识别了用户“家庭义务与小圈层忠诚”的价值冲突，并给出适度探问。
- 单样本只能证明链路和基本输出合理，不能证明 Ours 在整体指标上优于 baseline。

注意：该 smoke 在提交这些改动之前运行，因此其 `run_manifest.json` 中记录的 Git commit 仍是 `5de4271`。正式实验必须在当前已提交版本上重新运行，不能直接把这份 smoke 当论文结果。

## 8. 公开数据结构验证

| 验证项 | 结果 |
|---|---:|
| AlpsBench Task 1 dev | 466 条 |
| AlpsBench Task 1 validation | 466 条 |
| 合计输入 | 932 条 |
| 成功连接 reference | 932 条 |
| 缺失 reference | 0 条 |
| 公开 gold memory | 2,564 条 |
| 官方 `filter.py` 能否读取适配结构 | 能 |

结构烟测中曾统一赋予合法 `Personal Advice` intent，官方 memory filter 输出 558 条。这个 558 只证明数据结构兼容，**不是正式数据集规模**；正式规模必须以真实 intent 重建和完整官方流水线的 manifest 为准。

另有两项真实结构化调用验证：

- Kimi intent Schema 成功返回：`Learning Support`。
- Kimi Big Five Schema 成功返回五个合法的 low/medium/high 标签。

## 9. 当前还没有的正式指标

下列结果尚未生成，报告中不得填入或推测：

- 正式 12 条平衡样本的四方法回复。
- Random 30 条和 OOD 30 条预检结果。
- Qwen3-30B-A3B-Instruct Judge 的 Res/Exp/Rec/Avg。
- DeepSeek-v4-flash Judge 的 Res/Exp/Rec/Avg。
- 全量 Random/OOD 主实验、置信区间和定性样例。

缺少这些结果不是当前代码逻辑错误，而是尚未启动完整付费数据重建，且尚未配置两个正式 Judge 的 API。

## 10. 最终判断

### 可以确认

- 实验任务没有退回早期 REALTALK 情绪预测。
- 当前实现确实是 PersonaEmp 的个性化共情回复任务。
- 四方法输入边界、隐藏信息隔离、官方评价维度和用户级划分符合已确定方案。
- Ours 保留原有核心 prompt 和行为，仅增加 benchmark 适配。
- 代码已经具备从公开数据重建到四方法生成、Random/OOD、双 Judge、统计和可视化的完整路径。

### 不能过度表述

- 不能说已经复现官方绝对分数。
- 不能把 20 项自动化测试说成 20 条正式模型实验。
- 不能把当前 1 条 synthetic smoke 当成论文效果证明。
- 不能把结构烟测的 558 条当成正式最终数据量。

### 下一步正式执行顺序

1. 完整运行 932 条公开数据重建并冻结 reconstruction manifest。
2. 用重建数据运行 ES/HEQ/SS 各 4 条的 12 条平衡小测。
3. 生成人格标签和固定 Random/OOD split manifest。
4. Random/OOD 各运行 30 条四方法预检。
5. 使用 DeepSeek 生成并冻结 criteria。
6. 使用 Qwen3 与 DeepSeek 双 Judge 评分。
7. 确认无错位和异常后再运行全量实验。
