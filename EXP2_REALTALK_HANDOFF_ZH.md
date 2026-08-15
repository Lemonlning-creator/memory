# Exp2 REALTALK Ours 交接文档

## 1. 接手时先看什么

1. `EXP2_REALTALK_PACKAGE_INDEX_ZH.md`
2. `EXP2_REALTALK_FINAL_REPORT_ZH.md`
3. `EXP2_REALTALK_ARTIFACT_MANIFEST.json`
4. `EXP2_REALTALK_OURS_FINAL_PROTOCOL_ZH.md`

当前权威结果是 V9 全量 519 条。V13 只完成 Gate 1，不要把 V13 小样本数字写入主表。

## 2. Git 状态

- Fork：`https://github.com/Nobody-ly/memory.git`
- 师姐仓库：`https://github.com/Lemonlning-creator/memory.git`
- Exp2 分支：`paper-boost/exp2-user-modeling-evaluation`
- V9 实现提交：`5927bbff03fda74eebaeb99e0c57203a644cfd74`
- V13.4 运行源码标识：目录后缀 `3dbcc4c`。
- V13.6 运行源码标识：目录后缀 `a358b28`。

交接后先确认两个远端同名分支指向同一 tip：

```bash
git ls-remote origin refs/heads/paper-boost/exp2-user-modeling-evaluation
git ls-remote upstream refs/heads/paper-boost/exp2-user-modeling-evaluation
```

## 3. 代码入口

- 主 Ours 管线：`src/experiments/realtalk_ours.py`
- V13 渐进管线：`src/experiments/realtalk_v13.py`
- V13 Schema：`src/experiments/realtalk_v13_schemas.py`
- 本地五项指标：`src/experiments/realtalk_local_metrics.py`
- 完整 GPT Judge：`src/experiments/realtalk_gpt_judge.py`
- 配对 Judge：`src/experiments/realtalk_paired_judge.py`
- 报告工具：`src/experiments/realtalk_v11_report.py`
- 测试：`tests/test_realtalk*.py`

## 4. 服务器环境

- 主目录：`/amax/xidian_ty/Ly/personaemp-exp2`
- Git worktree：`/amax/xidian_ty/Ly/personaemp-exp2/worktrees/realtalk-ours-v12-8bdc721`
- Python：`/amax/xidian_ty/Ly/personaemp-exp2/worktrees/realtalk-ours-90f8a43/.venv/bin/python`
- Ours 环境文件：`/amax/xidian_ty/Ly/personaemp-exp2/secrets/realtalk_ours.env`
- Judge 环境文件：`/amax/xidian_ty/Ly/personaemp-exp2/secrets/realtalk_judge.env`

环境文件只保存于服务器，不进入 Git。日志、manifest、报告均不得包含 API key。

## 5. 权威产物

V9：

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-evidencefix-flash-5927bbf`
- 本地指标：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-local-metrics-v1`
- GPT Judge：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-judge-resume-v1`

V13 诊断：

- V13.4：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v13-4-progressive-v1-3dbcc4c`
- V13.6：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v13-6-progressive-v1-a358b28`

详细 SHA256 在 `EXP2_REALTALK_ARTIFACT_MANIFEST.json`。不要覆盖、合并或移动这些目录。

## 6. 验证命令

本地代码验证：

```bash
python -m pytest -q tests -k realtalk
```

服务器产物最小核验：

```bash
wc -l /amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-evidencefix-flash-5927bbf/predictions.jsonl
wc -l /amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-judge-resume-v1/scored.jsonl
sha256sum /amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-evidencefix-flash-5927bbf/predictions.jsonl
```

预期分别为 519、519，以及预测哈希
`ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81`。

## 7. 完成定义

V9 的生成与评价组合已完成，但生成目录的旧 `run_manifest.json` 仍保留生成结束时的
`gpt_evaluation_status=pending`，因为 Judge 后来在独立目录补齐。不要修改旧 manifest。
组合完成证据是：

- predictions 519/519；
- local metrics 519/519；
- Judge 519/519，3,114/3,114 单元；
- Judge `status=complete`；
- unresolved 为 0。

## 8. 续跑边界

- 不运行论文基线，不训练或微调。
- 不改变 Table 8 Ca/Cb、前三个连续 Session、519 条样本和真实历史滚动规则。
- 新 Prompt 或 Schema 必须使用新协议名、新目录，并从小门禁重新开始。
- 不按具体人物、result ID 或 Ground Truth 定制 Prompt。
- V13 当前已停止；若继续，应建立 V13.x 或 V14，而不是覆盖 V13.4/V13.6。
- 全量比较继续采用 speaker macro mean 与 population std。

## 9. 已知限制

- 论文未公布 Persona Simulation 的基础模型和完整评测实现。
- 519 是协议重建数，不是论文公开的官方样本数。
- BERTScore 配置是标准重建，不是官方未公开代码的逐字复制。
- V9 有 3 项未超过论文逐列最优；不可宣称八项全部领先。
