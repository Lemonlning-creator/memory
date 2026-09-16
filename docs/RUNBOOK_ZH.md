# 操作手册

## 环境

建议 Python 3.11/3.12 + Linux；Windows 可以做数据校验与单测，原版 POSIX alarm 超时测试在 Windows 跳过。建议从仓库根目录执行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m release_tools.data --download
python -m release_tools.cli verify-code
python -m pytest -q
```

安装没有历史 Web UI、Milvus、Chroma 等服务依赖。冻结支持模块因导入关系保留，但不启动它们。完整本地指标需要额外安装：

```bash
python -m pip install -e '.[realtalk-eval]'
```

初次本地评价会下载 3 个分类器和 `roberta-large`，需要网络、磁盘与 CPU 内存，不需要训练/GPU。封装入口强制 `CUDA_VISIBLE_DEVICES` 为空。已缓存模型可由操作者设置 `HF_HOME`；离线时确保模型都存在再设置 `HF_HUB_OFFLINE=1`。

原 V9 记录了 bert-score 0.3.13、transformers 4.57.6。本包固定 bert-score，保留兼容版本范围；要贴近旧环境，可显式安装 `transformers==4.57.6`。它不是原机器的逐字节完整环境锁。

## 端点与密钥

在本地创建 `.env`，字段参考 `.env.example`。`REALTALK_OURS_*` 和 `REALTALK_JUDGE_*` 分开，禁止把真实密钥写到代码、README 或启动命令中。环境变量优先于 `.env`。

Ours 模型不能随意使用端点默认模型，入口固定 `deepseek-v4-flash`。模型若不可用，停止并报告，不能自动替换为 Pro、Qwen 或其他 alias。Judge 固定 `gpt-4o-mini`。Schema/接口不兼容不属于性能结论。

## 生成

```bash
python -m release_tools.cli generate --scope preflight --output results/preflight --new
python -m release_tools.cli generate --scope smoke --speaker Akib --output results/smoke-akib --new
python -m release_tools.cli generate --scope full --output results/v9-full --new
```

smoke 是所选人物每 session 中点一条，不是正式性能表。full 固定十人全量 519 条，不传 speaker 或抽样。上述是**复现命令**；本次归档不代表已重新启动这些收费调用。

重启同一配置、同一提交、同一目录：

```bash
python -m release_tools.cli generate --scope full --output results/v9-full --resume
```

checkpoint signature 绑定代码提交、输入哈希、Prompt、Schema 与配置。不能直接用本发布分支去续跑原 `5927bbf` 目录，即便核心代码相同；旧目录保留，复现用新目录。`--new` 拒绝非空目录，不暴露原版的危险覆盖行为。

每个结构化逻辑操作最多三次尝试，网络重试默认最多六次；这不是“总体只发三次 HTTP”。每次人工重新 resume 也可能再次尝试失败操作，不能当成永久预算上限。不要无界循环重启。

## 八项指标

本地五项，不调用 GPT：

```bash
python -m release_tools.cli local-metrics --predictions results/v9-full/predictions.jsonl --output results/v9-full-local
```

GPT 三项，真实调用并产生费用：

```bash
python -m release_tools.cli judge --predictions results/v9-full/predictions.jsonl --output results/v9-full-judge
```

每条 reference/candidate 各评 Reflectiveness、Grounding、Empathy 三次；全新 519 条 Judge 逻辑请求共 3,114 次，重试另计。当前冻结代码串行评测，checkpoint 支持同命令续跑。本包没有新增并发或跨运行参考标签复用，不改变旧行为。

新入口绑定预测 SHA、数据协议、代码与 Judge 模型，拒绝在另一组输入上沿用同一评价目录。只有 ID、真实答案和历史重建校验通过才能提交评价。

## 完整结果验收

```bash
python -m release_tools.report --generation results/v9-full --local results/v9-full-local --judge results/v9-full-judge --output results/v9-full-report
```

要求 519 条无重复、缺失；十人计数正确；历史无裁剪；零 unresolved；全部八项有效。生成 `summary.json`、`TABLE2.md` 与发布层 `PIPELINE_COMPLETE`。标志只表示计算齐全，**不表示分数超过论文**。

单独的原生成模块只写 `GENERATION_COMPLETE` 并声明 GPT 待评；本地模块另写评价 manifest。本包不会篡改这些历史标志，而是在报告层验证汇总。

## 冻结包校验

```bash
python -m release_tools.package_frozen --output /path/to/realtalk-v9-frozen-519-v1 --verify-only
```

如需从服务器现有原目录重新封装，必须使用未占用的新目标目录：

```bash
python -m release_tools.package_frozen --runs /amax/xidian_ty/Ly/personaemp-exp2/runs --dataset dataset --output /path/to/new-frozen-package
```

这条只读旧结果并复制，不调用模型。输入被固定预测哈希与提交约束，不可拿 V10/V14/Pro 结果冒充 V9。
