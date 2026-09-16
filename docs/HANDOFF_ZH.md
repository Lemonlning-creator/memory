# 交接与维护

## 本次整理边界

冻结 V9 源码和传递导入依赖共 28 个 Python 文件，逐字节取自 `5927bbf`；包括 5 个原测试文件。整理不改 Prompt、Schema、解码、策略、数据样本或评价计算。发布入口与校验工具单独放 `release_tools/`。

分支采用独立根提交，干净树不携带历史的大量实验计划、Web UI、个人配置或旧运行产物。完整历史仍在原仓库分支；`provenance/canonical_source.json` 记录原提交、每个文件哈希与 AST 导入边。它是源代码依赖索引，不把静态导入当作运行链路。

该分支不应直接合并到 main 以删除其他工作；它是专门 checkout、交接、复核 V9 的干净分支。

## 代码职责

| 文件 | 职责 |
|---|---|
| src/experiments/realtalk_ours.py | 编排、4 阶段 Prompt、样本生成、画像/策略/输出校验 |
| src/experiments/realtalk_ours_schemas.py | Self/User/Decision Schema 与本地规范化 |
| src/experiments/exp1_protocol.py | 历史文件名；实际提供 REALTALK Table 8、合并与因果测试点 |
| src/experiments/persona_simulation.py | 历史 helper 文件；V9 使用 session_keys/flatten_messages，不调用其中旧实验 CLI |
| src/experiments/personaemp/client.py | 共享 OpenAI-compatible API 客户端，不表示本版在跑 PersonaEmp |
| src/experiments/operation_checkpoint.py | 操作/结果缓存、签名、失败记录 |
| src/experiments/realtalk_local_metrics.py | 已冻结预测的本地五项评价入口 |
| src/experiments/realtalk_evaluator.py | 固定分类器与权重 revision |
| src/experiments/exp2_generation.py | V9 使用其中 BERTScore helper，不使用其旧响应生成器 |
| src/metrics.py | V9 使用 ROUGE-L；其余函数不是新增本版步骤 |
| src/experiments/realtalk_gpt_judge.py | Appendix C Prompt、Judge 请求、reference/candidate 配对与汇总 |
| release_tools/cli.py | 锁定真实 V9 配置、防误用默认 Qwen、防覆盖、评价输入绑定 |
| release_tools/data.py | 固定源数据获取、519 条重建与预测因果一致性校验 |
| release_tools/report.py | 完整八项汇总与报告级完成检查 |
| release_tools/package_frozen.py | 已有 V9 独立归档，零模型调用 |

保留少量共享文件的完整内容是为了字节可核对，不代表所有历史类/函数都进入 V9。本分支没有额外依赖 Milvus、Chroma 或应用服务器。

## 新操作者顺序

1. 先读 README 确认 V9 = DeepSeek Flash、无微调。
2. 校验冻结包哈希，看 TABLE2 和每人物汇总；不要先开收费任务。
3. 准备官方十个 JSON，确认 519 条和 Table 8 映射。
4. 跑离线单测与 `verify-code`；阅读原 Prompt/Schema。
5. 真要重现才新建输出目录执行 preflight / smoke / full，再本地指标与 GPT Judge。
6. 未来改进另开新版本；不覆盖 V9、不给 V9 旧结果打新模型标签。

## 明确未完成/未承诺的事

- 不实现训练，不补实验一/三，不继续 Pro/4.1 模型对照。
- 不声称比论文八项全胜、不声称独立无污染测试。
- 不保证端点模型永远可用或无 seed API 重跑逐字相同。
- 没有复制完整服务器虚拟环境；分类器 revision、已知包版本和源代码都可核对。
- 不把旧记录里 `v8` / `V2` / `qwen3-max` 的默认命名误当作正式模型；入口已经显式固定。

## 验证证据

本次校验结果另写 `reports/RELEASE_VALIDATION.md`；包含单测数量、完整数据计数、代码哈希、结果包完整性以及是否进行了真实 API 调用。只有实际执行通过的项才记通过。
