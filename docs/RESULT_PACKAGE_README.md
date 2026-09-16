# REALTALK Ours V9 · 519 条冻结结果包

本包来自已完成的 V9，不是重跑，不含 V10 以后实验。模型 `deepseek-v4-flash`；全部阶段 thinking=false；无微调。原实现提交 `5927bbff03fda74eebaeb99e0c57203a644cfd74`。

## 文件导航

| 路径 | 内容 |
|---|---|
| TABLE2.md | 论文两行与 V9 一行的八项对比 |
| summary.json | 从逐样本指标重算的汇总、逐人物和逐样本八项分数 |
| generation/predictions.jsonl | 519 条预测和 Ground Truth，逐样本 User Domain、State、λ、Next Action、历史 ID/哈希 |
| generation/self_domains.json | 10 人的固定 Self Domain |
| generation/run_manifest.json | 原模型、thinking、解码、Prompt/Schema 哈希、调用统计 |
| generation/dataset_manifest.json | Ca/Cb 映射、输入文件哈希、人物计数 |
| generation/unresolved_errors.json | 原生成未解决错误，必须为空 |
| local/results_with_local_metrics.jsonl | 519 条五项本地指标和分类器标签，含原预测 |
| local/local_metrics_summary.json | 原本地汇总与评价模型 revision |
| judge/scored.jsonl | 519 条 reference/candidate 标签与三项分数 |
| judge/checkpoint.json | 3,114 个固定 Judge 判断及模型审计，不重新判断真值 |
| judge/summary.json | 原 Judge 完成状态与人物汇总 |
| judge/reuse_manifest.json、reuse_conflicts.json | 原运行的标签复用审计，保留不改 |
| dataset/ | 原 V9 使用的十个官方 JSON 完整副本；实际仍只使用前三个 session |
| contracts/ | 从冻结源码导出的完整英文 Prompt 和 JSON Schema |
| provenance/ | 原始源码、运行、数据来源与 SHA256 |
| ORIGINS.json | 每个复制结果文件来自哪里、原始哈希 |
| PACKAGE_MANIFEST.json | 本包全部其他文件的大小和 SHA256 |
| PIPELINE_COMPLETE | 封装层完成证明，不代表分数全部优于论文 |

Self Domain 不在每条记录中重复，靠人物名和 `self_domain_hash` 对照；每条记录中的 User Domain 是当时实际可见版本。没有单独的训练权重，因为 V9 不训练。

原 generation/run_manifest.json 的 `gpt_evaluation_status` / `pipeline_complete` 可能仍表示生成结束时“待 Judge”，这是历史快照。后来完成的本地与 Judge 结果在各自目录；本包顶层完成证明是在 519 条和 3,114 判断交叉核对后新增，不回写旧 manifest。

## 校验

在配套代码仓库根目录执行：

```bash
python -m release_tools.package_frozen --output /path/to/this-package --verify-only
```

正式 predictions SHA256：

```text
ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81
```

结果包复制位置：

- 本地：`D:\codex_workspace\deliveries\realtalk-v9-frozen-519-v1\`
- 服务器：`/amax/xidian_ty/Ly/personaemp-exp2/releases/realtalk-v9-frozen-519-v1/`
- 配套代码分支：`https://github.com/Lemonlning-creator/memory/tree/release/exp2-realtalk-v9-clean`

本包含对话和推断画像。应遵守官方数据来源的使用条件，画像只作为模型推断，不视为现实人物已证实的心理属性。不将全部数据/画像再上传公开 Git。没有复制密钥、env、训练缓存、后续版本数据或服务器启动日志。原 raw_responses 和生成 checkpoint 仍在原运行目录，详见 ORIGINS 的父目录。
