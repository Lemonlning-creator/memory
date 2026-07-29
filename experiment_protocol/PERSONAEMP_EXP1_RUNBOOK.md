# PersonaEmp Exp1 运行说明

## 1. 校验数据

```powershell
python -m src.experiments.personaemp.cli `
  --dataset path/to/English.json `
  --output-dir data/personaemp_exp1/validate `
  --validate-only
```

## 2. 配置生成模型

在本地 `.env` 或当前终端设置：

```powershell
$env:PERSONAEMP_GENERATOR_API_KEY="<local-secret>"
$env:PERSONAEMP_GENERATOR_BASE_URL="https://api.moonshot.cn/v1"
$env:PERSONAEMP_GENERATOR_MODEL="kimi-k2.6"
$env:PERSONAEMP_GENERATOR_ENABLE_THINKING="false"
```

不要把 Key 写入命令历史、代码、运行清单或实验输出。

`ours` 与 `base_model` 共享相同的 `extracted_memory + query` 原始证据。
数据集的 persona 和 scenario 不提供给生成器，只供固定版本官方评测器使用。

## 3. 小规模生成

```powershell
python -m src.experiments.personaemp.cli `
  --dataset tests/fixtures/personaemp_paper_case.json `
  --output-dir data/personaemp_exp1/paper_case_smoke `
  --dataset-provenance paper_case_pilot `
  --methods ours base_model `
  --limit 3
```

输出包括：

- `results.jsonl`：逐样本结果和调用统计；
- `errors.jsonl`：失败记录；
- `run_manifest.json`：代码、数据、模型和 Prompt 指纹；
- `evaluation_dataset.json`：本次选择的官方格式数据；
- `predictions/ours.json`；
- `predictions/base_model.json`；
- `summary.json`。

重复运行同一命令会从 checkpoint 恢复，不会重复计算成功样本。

## 4. 生成固定 criteria

官方代码另行克隆并固定在提交
`b555447f267b8057039aab39a4be44725718ea7f`。

```powershell
$env:PERSONAEMP_CRITERIA_API_KEY="<local-secret>"
$env:PERSONAEMP_CRITERIA_BASE_URL="<deepseek-compatible-url>"
$env:PERSONAEMP_CRITERIA_MODEL="deepseek-v4-flash"

python -m src.experiments.personaemp.official_eval prepare-criteria `
  --official-repo D:\codex_workspace\.references\PersonalizedEmpathy-official `
  --dataset data/personaemp_exp1/paper_case_smoke/evaluation_dataset.json `
  --output data/personaemp_exp1/paper_case_smoke/criteria.json `
  --limit 9
```

三条查询、三个维度共九次 criteria 调用。

## 5. 运行 Judge

Qwen Judge：

```powershell
python -m src.experiments.personaemp.official_eval judge `
  --official-repo D:\codex_workspace\.references\PersonalizedEmpathy-official `
  --dataset data/personaemp_exp1/paper_case_smoke/evaluation_dataset.json `
  --predictions data/personaemp_exp1/paper_case_smoke/predictions/ours.json `
  --criteria data/personaemp_exp1/paper_case_smoke/criteria.json `
  --output data/personaemp_exp1/paper_case_smoke/evaluations/ours.qwen.json `
  --judge-name qwen `
  --env-prefix PERSONAEMP_QWEN_JUDGE
```

DeepSeek Judge 使用相同命令，将输出名、Judge 名和环境变量前缀改为：

```text
ours.deepseek.json
deepseek
PERSONAEMP_DEEPSEEK_JUDGE
```

每个 Judge 输出旁会生成 `.summary.json`，同时保留 1–5 原始均分和 0–1 归一化均分。
