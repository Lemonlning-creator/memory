# PersonaEmp Exp1 运行说明

## 当前实验比较什么

当前适配器运行两组：

- `base_model`：读取 `extracted_memory + query`，直接生成回复。
- `ours`：读取同一份 `extracted_memory + query`，先派生五层画像和 Deep Empathy 状态，再生成回复。

两组最终回答共享同一个实验回复合同：与用户同语言、2--4 句、先给具体帮助再选择性追问、自然共情、不透露画像或记忆来源。生产系统的核心 Prompt 文件不做修改。

数据集中的 `persona`、`scenario/situation`、`category` 和 `conversation` 不提供给任何生成方法。它们只保留为数据元信息或供官方评审流程使用。

## 配置生成模型

模型通过环境变量配置。本阶段可用 Kimi 验证流程，正式实验再切换指定模型：

```powershell
$env:PERSONAEMP_GENERATOR_API_KEY="<local-secret>"
$env:PERSONAEMP_GENERATOR_BASE_URL="https://api.moonshot.cn/v1"
$env:PERSONAEMP_GENERATOR_MODEL="kimi-k2.6"
$env:PERSONAEMP_GENERATOR_ENABLE_THINKING="false"
```

不要把 API Key 写入代码、命令记录、运行清单或实验输出。

## 校验和运行

只校验数据：

```powershell
python -m src.experiments.personaemp.cli `
  --dataset path/to/English.json `
  --output-dir data/personaemp_exp1/validate `
  --validate-only
```

小规模生成：

```powershell
python -m src.experiments.personaemp.cli `
  --dataset tests/fixtures/personaemp_paper_case.json `
  --output-dir data/personaemp_exp1/paper_case_smoke `
  --dataset-provenance paper_case_pilot `
  --methods ours base_model `
  --limit 3
```

重复运行同一命令会从 checkpoint 恢复，不重复计算成功样本。

## 输出

- `results.jsonl`：逐样本回复、调用次数、Token 和耗时。
- `errors.jsonl`：失败记录。
- `run_manifest.json`：代码版本、数据指纹、模型、输入边界和 Prompt 指纹。
- `evaluation_dataset.json`：本次选中的官方格式数据。
- `predictions/ours.json` 与 `predictions/base_model.json`：供官方 Judge 使用。
- `summary.json`：成功率与成本汇总。

`summary.json` 将画像提取列为 `profile_preprocessing`，不把它混入在线回复速度。Ours 的在线成本只比较 alignment 与 response；同时保留完整端到端成本，便于工程分析。

## 官方评审

正式指标为 PersonaEmp 的 Resonation、Expression、Reception 与 Average。固定 criteria 后，分别交给论文指定的 Qwen 与 DeepSeek Judge；当前仓库通过固定官方提交的外部 checkout 调用 evaluator，不改写其评分实现。

只有数据 SHA-256 与登记的官方 Table 1 数据指纹一致时，清单才允许标记为可直接比较。论文公开案例或重新生成的数据只能作为流程验证或补充实验。
