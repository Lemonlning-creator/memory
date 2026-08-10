# Experiment 2: User Modeling Evaluation

本文件说明如何在 NVIDIA RTX 3090 机器上部署和运行实验二。

实验入口只有一个：

```text
src/experiments/exp2_user_modeling.py
```

推荐按以下顺序运行：

```text
prepare -> generate -> evaluate -> curves
```

其中 `prepare`、`generate` 和 `evaluate` 是主结果必需阶段；`curves` 是额外定性分析。

## 1. 实验协议

- 数据集：REALTALK 的 `dataset/Chat_*.json`。
- 划分：每个对话按 Session 时间顺序进行 9:1 划分。
- 前 90% Session：一次性抽取用户画像和智能体人设。
- 后 10% Session：生成智能体回复并进行评估。
- 测试历史：始终使用数据集中的真实双方回复，即 teacher forcing。
- 测试输入：真实历史、当前用户消息、训练集用户画像、训练集智能体人设。
- 评估对象：Ours 生成回复与数据集真实智能体回复。
- 测试阶段不使用真实目标回复或未来用户消息作为生成输入。

## 2. 主结果指标

最终表格沿用 REALTALK Table 2，并在原始两行后追加 `Ours`：

| 分组 | 指标 | 计算方式 | 方向 |
|---|---|---|---|
| Content Similarity | Lexical | ROUGE-L F1 | 越高越好 |
| Content Similarity | Semantic | BERTScore F1 | 越高越好 |
| Message-level EI | Reflective | 生成回复标签与真实回复标签的准确率 | 越高越好 |
| Message-level EI | Grounding | 生成回复标签与真实回复标签的准确率 | 越高越好 |
| Message-level EI | Sentiment | 生成回复标签与真实回复标签的准确率 | 越高越好 |
| Message-level EI | Emotion | 生成回复标签与真实回复标签的准确率 | 越高越好 |
| Message-level EI | Intimacy | 生成回复与真实回复的分数绝对差 | 越低越好 |
| Message-level EI | Empathy | 生成回复与真实回复的总分绝对差 | 越低越好 |

聚合时先计算每位目标 speaker 的平均分，再计算所有目标 speakers 之间的 `mean ± population std`。

## 3. 3090 环境部署

以下以 Linux + Bash 为主。在仓库根目录执行：

```bash
cd /path/to/memory
```

### 3.1 创建虚拟环境

推荐使用 Python 3.11 到 3.13：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如果服务器已经有本项目的 `.venv`，可以直接激活，不需要重新创建。

### 3.2 安装项目基础依赖

```bash
python -m pip install -e .
```

### 3.3 安装3090评估依赖

PyTorch 的 CUDA wheel 应根据服务器驱动选择。下面是 CUDA 12.8 的安装示例：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers bert-score
```

如果服务器驱动不适合 CUDA 12.8，请使用 PyTorch 官方安装选择器生成对应命令，不要安装 CPU-only wheel。

验证3090是否被 PyTorch 正确识别：

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

预期输出应包含：

```text
cuda= True
gpu= NVIDIA GeForce RTX 3090
```

## 4. API 与模型配置

仓库根目录需要 `.env`：

```dotenv
API_KEY=你的密钥
BASE_URL=OpenAI兼容接口地址
```

`config.ini` 中至少确认：

```ini
[API]
model = qwen3.6-flash
embedding_model = text-embedding-v4
enable_thinking = False
```

注意：

- 当前记忆向量维度固定为 1536。
- embedding 服务必须让 `text-embedding-v4` 返回 1536 维向量。
- 画像、人设和回复生成使用 `config.ini` 中的对话模型。
- Table 2 的 Reflective、Grounding 和 Empathy 官方评估模型是 `gpt-4o-mini`。
- 如果当前 `BASE_URL` 不支持 `gpt-4o-mini`，可以通过 `--judge-model` 指定兼容模型，但这会降低与原论文结果的严格可比性。

## 5. 单个对话完整运行

下面以 `Chat_1_Emi_Elise.json` 为例。四个阶段必须使用相同的 `--output-dir`、`--case` 和 `--train-ratio`。

### 5.1 阶段一：抽取用户画像和智能体人设

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase prepare \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling
```

主要输出：

```text
data/exp2_user_modeling/cases/chat_1_emi_elise__emi__to__elise/assets/
├─ agent_persona.json
├─ user_profile.json
├─ user_profile_runtime.json
└─ asset_manifest.json
```

- `user_profile.json`：固定字段的一次性用户画像，便于人工检查。
- `user_profile_runtime.json`：核心 Agent 运行时需要的包装格式。
- `agent_persona.json`：智能体人设。

### 5.2 阶段二：生成测试集回复

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling
```

主要输出：

```text
data/exp2_user_modeling/cases/chat_1_emi_elise__emi__to__elise/generations/predictions.jsonl
```

每条记录包含：

- 当前真实用户消息；
- Ours 生成回复；
- 数据集真实参考回复；
- 输入泄漏审计字段；
- 使用的 teacher-forcing 历史策略。

### 5.3 阶段三：计算 Table 2 指标

第一次执行该阶段时，Hugging Face 会下载评估模型。应提前确保服务器能够访问 Hugging Face，并为模型缓存预留空间。

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling \
  --eval-device cuda:0 \
  --eval-batch-size 16 \
  --judge-model gpt-4o-mini
```

评估模型包括：

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
cardiffnlp/twitter-roberta-large-emotion-latest
cardiffnlp/twitter-roberta-large-intimacy-latest
roberta-large                         # BERTScore
gpt-4o-mini                           # Reflective/Grounding/Empathy
```

主要输出：

```text
data/exp2_user_modeling/
├─ cases/<case_id>/evaluation/
│  ├─ table2_annotations.jsonl
│  └─ table2_scores.json
└─ evaluation/
   ├─ table2_main_results.json
   └─ table2_main_results.md
```

`table2_main_results.md` 即原论文两行加 `Ours` 的可直接检查表格。

### 5.4 阶段四：画像曲线

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase curves \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling
```

当前需要注意：现有 `prepare` 使用一次性画像抽取，不会生成逐 Session 的 `profile_snapshots`。因此曲线阶段目前还不能直接接在一次性画像后运行；主结果只需要完成 `prepare -> generate -> evaluate`。在画像进化数据来源最终确定前，不建议运行 `curves` 或 `--phase all`。

## 6. 批量运行

不传 `--case` 就会处理 `dataset` 下全部 `Chat_*.json`。

### 6.1 批量准备画像和人设

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase prepare \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling
```

### 6.2 批量生成回复

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling
```

### 6.3 批量评估并生成最终表格

```bash
./.venv/bin/python -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --train-ratio 0.9 \
  --output-dir data/exp2_user_modeling \
  --eval-device cuda:0 \
  --eval-batch-size 16 \
  --judge-model gpt-4o-mini
```

## 7. 断点续跑

生成和评估均支持断点续跑：

- `predictions.jsonl` 通过 `example_id` 避免重复生成。
- `table2_annotations.jsonl` 通过评测点、候选类型和评估器指纹避免重复标注。
- 已成功写入的记录会被跳过。
- 中断后使用完全相同的命令重新执行即可。

不要在不同实验协议之间复用同一个输出目录。修改模型、数据划分或角色方向时，推荐换一个新的目录，例如：

```bash
--output-dir data/exp2_user_modeling_run2
```

## 8. 3090显存不足时

如果 BERTScore 报 CUDA out of memory，先降低批大小：

```bash
--eval-batch-size 8
```

仍然不足时使用：

```bash
--eval-batch-size 4
```

Reflective、Grounding 和 Empathy 通过 API 评估，不占用3090显存。Sentiment、Emotion、Intimacy 和 BERTScore 使用本地 GPU。

## 9. 常见错误

### `ModuleNotFoundError: torch/transformers/bert_score`

说明尚未安装评估依赖。重新执行第3.3节中的安装命令。

### `torch.cuda.is_available() is False`

检查：

1. NVIDIA 驱动是否正常；
2. 是否误装了 CPU-only PyTorch；
3. PyTorch wheel 的 CUDA 版本是否与当前驱动兼容。

### `no generated replies ... run --phase generate first`

评估目录中没有该 case 的 `predictions.jsonl`。确认 `prepare`、`generate` 和 `evaluate` 使用了相同的 `--output-dir` 与 `--case`。

### Hugging Face 模型下载失败

确认3090服务器可以访问 `huggingface.co`。如果模型已提前下载到服务器缓存，Transformers 会直接复用缓存。

### `gpt-4o-mini` 调用失败

当前 `BASE_URL` 可能不支持该模型。严格复现实验时应配置支持 `gpt-4o-mini` 的评估 API；临时检查流程时可以传入当前接口支持的模型，但必须在结果中记录这一偏差。

### Milvus 向量维度不匹配

当前代码固定使用 1536 维 embedding。不要复用由1024维或其他维度创建的旧 Milvus 数据库；使用新的 `--output-dir` 重新运行。

### `missing profile snapshot` 或 curves 失败

这是当前一次性画像抽取与逐 Session 曲线数据之间尚未对齐造成的。它不影响 Table 2 主结果，先完成 `prepare -> generate -> evaluate`。

## 10. 当前实验范围说明

当前生成代码对每个 REALTALK 文件使用：

```text
speaker_1 = 被建模用户
speaker_2 = 目标智能体
```

因此现有全部文件只覆盖作为 `speaker_2` 出现的目标 speakers。Table 2 最终写作要求是覆盖全部目标 speakers；在正式全量结果前，还需要补齐相反角色方向。当前单 case 或单方向输出应标记为阶段性结果，不能直接作为最终论文表格。

## 11. 正式运行前检查清单

- [ ] 使用固定代码版本并记录 Git commit。
- [ ] 9:1 Session 划分未被修改。
- [ ] 使用新的、干净的输出目录。
- [ ] embedding 输出维度为1536。
- [ ] 画像和人设只使用训练 Session。
- [ ] 生成历史使用数据集真实回复。
- [ ] `predictions.jsonl` 中参考回复未进入 generation input audit。
- [ ] 3090被 `torch.cuda.is_available()` 正确识别。
- [ ] 记录 judge model 和三个 CardiffNLP 模型名称。
- [ ] 最终 `table2_main_results.md` 包含原论文两行和 `Ours`。
- [ ] 正式论文结果覆盖全部目标 speakers，而不是单个 case。

## 12. Windows PowerShell附录

如果3090机器运行Windows，可以使用下面的PowerShell命令。实验参数和Linux版本完全一致。

### 12.1 环境部署

```powershell
cd E:\01_Research\03_UserProfile\memory
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers bert-score
```

### 12.2 单个对话

```powershell
.\.venv\Scripts\python.exe -m src.experiments.exp2_user_modeling `
  --phase prepare `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --output-dir data/exp2_user_modeling

.\.venv\Scripts\python.exe -m src.experiments.exp2_user_modeling `
  --phase generate `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --output-dir data/exp2_user_modeling

.\.venv\Scripts\python.exe -m src.experiments.exp2_user_modeling `
  --phase evaluate `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --output-dir data/exp2_user_modeling `
  --eval-device cuda:0 `
  --eval-batch-size 16 `
  --judge-model gpt-4o-mini
```

### 12.3 批量运行

批量运行时去掉 `--case`，依次执行 `prepare`、`generate` 和 `evaluate` 即可。
