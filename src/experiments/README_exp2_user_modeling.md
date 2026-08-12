# Experiment 2: User Modeling Evaluation

本文件说明如何在 NVIDIA RTX 3090 机器上部署和运行实验二。

实验二有两个相互独立的入口：

```text
src/experiments/exp2_user_modeling.py              # Table 2 主结果
src/experiments/exp2_user_modeling_qualitative.py  # 画像进化与画像熵
```

主结果按以下顺序运行：

```text
prepare -> generate -> evaluate
```

Qualitative Analysis 使用独立脚本单独运行，不依赖 `generate` 或 `evaluate`。

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

### 2.1 指标含义

- **Lexical（词汇相似度）**：使用 ROUGE-L F1 比较生成回复与数据集真实回复的最长公共子序列，反映两者在用词和句子片段上的重合程度。取值通常为 0–1，越高表示表层表达越接近真实回复；它不直接代表回复是否自然或正确。
- **Semantic（语义相似度）**：使用 `roberta-large` 的 BERTScore F1 比较生成回复与真实回复的上下文语义，允许两者使用不同措辞表达相近含义。越高表示整体语义越接近真实回复。
- **Reflective（反思性语言一致性）**：分别判断生成回复和真实回复是否表现出自我观察、对自身反应的视角理解，或对行为动机和目标的解释，再计算两者标签的一致率。只陈述事实、偏好、决定或普通认可不算反思性语言。取值为 0–1，越高表示生成回复的反思性表达模式与真实回复越一致；它不是对“反思越多越好”的直接评分。
- **Grounding（共同理解建立行为一致性）**：分别判断生成回复和真实回复是否通过澄清问题、针对前文的相关追问或确认性检查来建立共同理解，再计算标签一致率。普通认可、单纯观察、无关提问或换话题式提问不算 Grounding。取值为 0–1，越高表示生成回复建立共同理解的方式越接近真实回复。
- **Sentiment（情感倾向一致性）**：使用 CardiffNLP 情感分类器分别标注生成回复和真实回复的正向、中性或负向倾向，再计算标签准确率。取值为 0–1，越高表示两者情感极性越一致。
- **Emotion（情绪类别一致性）**：使用 CardiffNLP 情绪分类器分别识别生成回复和真实回复的主要情绪，再计算标签准确率。取值为 0–1，越高表示两者表达的主要情绪越一致。
- **Intimacy（亲密度误差）**：使用 CardiffNLP intimacy 模型分别给生成回复和真实回复评分，然后计算两者的绝对差 `|generated - reference|`。越低表示生成回复的自我披露和亲密程度越接近真实回复。该列不是“亲密度本身”，因此不能解释为数值越高越亲密或越好。
- **Empathy（共情强度误差）**：按照 EPITOME 的 Emotional Reaction、Interpretation、Exploration 三个维度分别给出 0–2 分，总分范围为 0–6；指标取生成回复总分与真实回复总分的绝对差。越低表示共情强度越接近真实回复。过度共情和共情不足都会增大误差，因此该列不能解释为“共情越多越好”。

Table 2 衡量的是生成回复对真实回复的**模拟一致性**。高分表示生成回复在相应属性上更像数据集中的真实人物回复，并不等同于对回复通用质量、帮助性或安全性的独立评价。

### 2.2 逐样本评估方法

每个测试评测点包含同一条真实用户输入对应的两条候选回复：

```text
reference = REALTALK 数据集中的真实目标角色回复
generated = Ours 根据历史、用户画像和智能体人设生成的回复
```

评估时从该回复所在 Session 的开头按原顺序读取真实对话，截断在目标回复之前，然后分别把 `reference` 和 `generated` 放到完全相同的最后一个位置。这样两条候选回复共享相同的 Session 内历史，评估器不会看到目标回复之后的消息。两条回复分别标注后再进行比较，不把 reference 直接交给分类器或 judge 作为 generated 的评分提示。

| 指标 | 评估器 | 单条样本计算方法 |
|---|---|---|
| Lexical | 项目中的 ROUGE-L 实现 | 直接计算 `ROUGE-L_F1(reference, generated)` |
| Semantic | BERTScore，`roberta-large` | 计算 `BERTScore_F1(reference, generated)`；运行设备由 `--eval-device` 指定 |
| Reflective | LLM judge + `REALTALK_REFLECTIVE_EVALUATION_SYSTEM_PROMPT` | 在相同对话历史下分别将 reference 和 generated 标为 `True/False`；标签相同记 1，否则记 0 |
| Grounding | LLM judge + `REALTALK_GROUNDING_EVALUATION_SYSTEM_PROMPT` | 在相同对话历史下分别将 reference 和 generated 标为 `True/False`；标签相同记 1，否则记 0 |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 分别预测两条回复的最高概率情感倾向标签；标签相同记 1，否则记 0 |
| Emotion | `cardiffnlp/twitter-roberta-large-emotion-latest` | 分别预测两条回复的最高概率情绪标签；标签相同记 1，否则记 0 |
| Intimacy | `cardiffnlp/twitter-roberta-large-intimacy-latest` | 分别取得模型输出分数，计算 `abs(score_reference - score_generated)` |
| Empathy | LLM judge + `REALTALK_EMPATHY_EVALUATION_SYSTEM_PROMPT` | 分别评估 ER、IN、EX 三项，每项 0–2 分；先求两侧总分，再计算 `abs(total_reference - total_generated)` |

三个 CardiffNLP 分类器只接收候选回复本身；Reflective、Grounding 和 Empathy judge 接收“相同的 Session 内历史 + 当前候选回复”。分类器输入使用 `truncation=True, max_length=512`，LLM judge 使用 `temperature=0.0`。阶段性调试可以通过 `--judge-model qwen-plus` 运行；严格复现 REALTALK 时应使用论文采用的 `gpt-4o-mini`。

三个 LLM judge prompt 以 REALTALK Appendix C.1–C.3 的定义、正反例和 EPITOME 分档为依据，并适配为当前代码使用的“System prompt + Session 内历史及最后候选回复”输入格式。这里没有复用项目中旧的通用 EI prompt，因为旧 prompt 的直接 0–2 对比方式及附加 appropriateness 字段不符合 Table 2 的标注与聚合协议。适配只明确判定边界和结构化输出：普通认可不自动算 Reflective，任意提问不自动算 Grounding，事实追问不自动算 Empathy Exploration；不改变 Table 2 的指标定义。

离散属性的逐样本分数为：

```text
score_i = 1[label_reference == label_generated]
```

Intimacy 和 Empathy 使用绝对误差：

```text
error_i = abs(value_reference - value_generated)
```

逐样本标注写入：

```text
cases/<case_id>/evaluation/table2_annotations.jsonl
```

每个对话的八项平均结果写入：

```text
cases/<case_id>/evaluation/table2_scores.json
```

### 2.3 聚合方法

最终结果不是把所有消息直接做微平均，也不是先把十个对话等权平均，而是：

1. 合并属于同一目标 speaker 的全部测试样本。
2. 对每位目标 speaker 分别计算每个指标的样本平均值。
3. 对所有目标 speakers 的均值做宏平均，得到表中的 `mean`。
4. 对目标 speakers 的均值计算总体标准差，得到表中的 `population std`。

因此表中的 `mean ± std` 表示“目标角色之间的平均表现及角色差异”，不是多次随机运行的均值和误差，也不是置信区间。完整聚合协议为：

```text
mean per target speaker -> macro mean and population std across target speakers
```

## 3. 3090 环境部署

以下以 Linux + Bash 为主。在仓库根目录执行：

```bash
cd /path/to/memory
```

### 3.1 安装uv并同步项目环境

如果服务器还没有uv，先安装并确认版本：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

使用uv安装Python 3.11，并根据仓库中的 `pyproject.toml` 和 `uv.lock` 创建或同步 `.venv`：

```bash
uv python install 3.11
uv sync --locked --python 3.11
```

不需要手动执行 `source .venv/bin/activate`。基础实验通过 `uv run` 使用项目环境；第3.3节单独安装评估依赖后，评估命令必须使用 `uv run --no-sync`。

### 3.2 同步项目基础依赖

```bash
uv sync --locked
```

### 3.3 安装3090评估依赖（Driver 470 / CUDA 11.x）

当前3090服务器的 NVIDIA Driver 为 `470.256.02`，`nvidia-smi` 显示 CUDA 11.4。该驱动属于CUDA 11.x兼容范围，因此使用PyTorch的CUDA 11.8 wheel，不使用CUDA 12.x wheel：

```bash
uv pip install "torch==2.7.1" --torch-backend=cu118
uv pip install --reinstall "transformers==4.57.6" "tokenizers==0.22.2" "huggingface-hub==0.36.2" "bert-score==0.3.13"
```

这里不要求系统额外安装完整的CUDA 11.8 Toolkit；PyTorch wheel会携带所需CUDA运行时。不要在这台Driver 470机器上安装 `cu12x` wheel。

当前 `torch`、`transformers` 和 `bert-score` 尚未写入项目锁文件，因此这里通过 `uv pip` 安装到uv管理的项目 `.venv`。不要在安装后直接执行普通的 `uv run`：它会根据当前锁文件重新同步 Chroma 的传递依赖，并可能把兼容版本回滚为 `tokenizers 0.23.1` 和 `huggingface-hub 1.16.1`。评估统一使用 `uv run --no-sync`。如果之后执行了 `uv sync`，需要重新执行本节。

评估前先检查依赖一致性和实际版本：

```bash
uv pip check
uv pip show transformers tokenizers huggingface-hub torch bert-score
```

预期至少包括：

```text
transformers      4.57.6
tokenizers        0.22.2
huggingface-hub   0.36.2
torch             2.7.1+cu118
bert-score        0.3.13
```

验证3090是否被 PyTorch 正确识别：

```bash
uv run --no-sync python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

预期输出应包含：

```text
torch_cuda= 11.8
cuda_available= True
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

下面以 `Chat_1_Emi_Elise.json` 为例。主实验的 `prepare`、`generate`、`evaluate` 必须使用相同的 `--output-dir`、`--case`、`--train-ratio` 和 `--prompt-version`。定性曲线脚本不生成智能体回复，因此不接收 `--prompt-version`。

### 5.1 阶段一：抽取用户画像和智能体人设

```bash
uv run python -m src.experiments.exp2_user_modeling \
  --phase prepare \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
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
uv run python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
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

当前3090已经缓存四个本地评估模型，因此下面使用离线模式，避免每次启动都访问 Hugging Face：

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
cardiffnlp/twitter-roberta-large-emotion-latest
cardiffnlp/twitter-roberta-large-intimacy-latest
roberta-large
```

在一台没有这些缓存的新机器上，第一次运行需要联网并暂时去掉 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；缓存完整后再使用下面的离线命令。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
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
uv run python -m src.experiments.exp2_user_modeling_qualitative \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --output-dir data/exp2_user_modeling
```

该脚本与主结果完全独立：

- 用户画像从空白状态开始，不读取 `prepare` 生成的一次性画像。
- 只按时间顺序回放前 90% Session 的 REALTALK 真实双方消息。
- 不生成智能体回复，不读取后 10% 测试 Session，也不运行 Table 2 指标。
- 直接调用现有 `MemoryOSLocal` 记忆流水线和 `bayesian_online` 画像更新。
- 从零开始时预置与一次性画像抽取完全相同的 5 层 21 个固定字段；动态更新只能修改字段内容和置信度，不能新增、删除、改名或移动字段。
- 在初始状态及每个训练 Session 结束后保存一次画像快照。

单案例输出位于：

```text
data/exp2_user_modeling/cases/<case_id>/qualitative/
├─ profile_runtime.json
├─ user_profile.json
├─ memory/memory.db
├─ profile_snapshots/
├─ profile_trajectory.json
└─ trajectory_manifest.json
```

`profile_runtime.json` 是算法内部使用的贝叶斯画像，包含置信度和证据等更新元数据。最终对外使用 `user_profile.json`：其结构与 `dataset/lsy_user.json` 一致，只包含 5 层、21 个固定字段及画像内容。

曲线输出位于：

```text
data/exp2_user_modeling/qualitative_figures/
├─ profile_curves.json
├─ profile_evolution_curve.png
└─ profile_entropy_curve.png
```

两条曲线始终使用同一组固定字段。画像进化是 21 个字段中已经填入稳定画像内容的比例；画像熵是这 21 个字段的平均二元熵，尚无内容的字段按最大不确定性 1.0 计算。横轴是前 90% 数据中的时间顺序 Session 编号，包含 Session 0 的空画像起点。

## 6. 批量运行

不传 `--case` 就会处理 `dataset` 下全部 `Chat_*.json`。

### 6.1 批量准备画像和人设

```bash
uv run python -m src.experiments.exp2_user_modeling \
  --phase prepare \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
  --output-dir data/exp2_user_modeling
```

### 6.2 批量生成回复

```bash
uv run python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
  --output-dir data/exp2_user_modeling
```

### 6.3 批量评估并生成最终表格

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
  --output-dir data/exp2_user_modeling \
  --eval-device cuda:0 \
  --eval-batch-size 16 \
  --judge-model gpt-4o-mini

# 远程连接测试
tmux new -s exp2_v4

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -u -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v4_task_reframed \
  --output-dir data/exp2_qwen_plus_v4 \
  --judge-config-section EvaluationAPI \
  --eval-device cuda:0 \
  --eval-batch-size 16 \
  --judge-model gpt-4o-mini \
  2>&1 | tee data/exp2_qwen_plus_v4_evaluate.log
```

### 6.4 连续执行生成和评估

已经准备好训练集画像和人设后，可以用 `generate-evaluate` 在一个进程中先生成全部选中 case 的回复，再立即计算 Table 2 指标。原来的 `generate` 和 `evaluate` 阶段仍然保留，可以继续单独运行。组合阶段同样支持断点续跑：如果生成已经完成而评估中断，重新执行相同命令会跳过已有 generation，再从评估缓存继续。

批量运行 v4：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase generate-evaluate \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v4_task_reframed \
  --output-dir data/exp2_qwen_plus_v4 \
  --judge-config-section EvaluationAPI \
  --eval-device cuda:0 \
  --eval-batch-size 16 \
  --judge-model gpt-4o-mini
```

只运行一个对话时，在相同命令中增加：

```bash
--case Chat_1_Emi_Elise.json
```

组合命令要求本地评估依赖和模型缓存已经可用。`--no-sync` 只禁止 uv 在启动时重新同步环境，不会阻止生成阶段调用 `[API]`，也不会阻止评估阶段调用 `[EvaluationAPI]`。

### 6.5 批量生成画像进化与画像熵曲线

不传 `--case` 即从零开始分别回放全部对话的前 90% Session：

```bash
uv run python -m src.experiments.exp2_user_modeling_qualitative \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --output-dir data/exp2_user_modeling
```

## 7. 断点续跑

生成和评估均支持断点续跑：

- `predictions.jsonl` 通过 `example_id` 避免重复生成。
- `table2_annotations.jsonl` 的缓存 ID 同时绑定评测点、候选类型、评估器指纹、候选回复 SHA256 和完整评估上下文 SHA256；回复或上下文变化后不会误用旧标注。
- 旧格式缓存只有在候选回复、上下文哈希和评估器指纹全部一致时才会迁移到新 ID，不会重新调用 Judge。
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

### `tokenizers>=0.22.0,<=0.23.0 ... found tokenizers==0.23.1`

这是普通 `uv run` 根据锁文件同步 Chroma 传递依赖造成的版本回滚。重新执行第3.3节的两个 `uv pip install` 命令，确认 `uv pip check` 通过，然后使用 `uv run --no-sync` 启动评估，不要再用普通 `uv run`。

### `torch.cuda.is_available() is False`

检查：

1. NVIDIA 驱动是否正常；
2. 是否误装了 CPU-only PyTorch；
3. PyTorch wheel 的 CUDA 版本是否与当前驱动兼容。

### `expanded size of the tensor ... 533 ... existing size 514`

这是回复超过 RoBERTa 上下文上限、但 tokenizer 未自动截断造成的错误。当前评估代码已显式使用 `truncation=True, max_length=512`；同步最新的 `src/experiments/exp2_user_modeling.py` 后，用相同评估命令继续即可，已写入的 JSONL 标注会自动复用。

### `no generated replies ... run --phase generate first`

评估目录中没有该 case 的 `predictions.jsonl`。确认 `prepare`、`generate` 和 `evaluate` 使用了相同的 `--output-dir` 与 `--case`。

### Hugging Face 模型下载失败

如果模型已经在本地缓存，使用文档中的 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，避免 Transformers 先联网执行 HEAD 请求。若离线模式提示找不到文件，则缓存并不完整，需要先在可联网环境下载上述四个模型。

### `gpt-4o-mini` 调用失败

当前 `BASE_URL` 可能不支持该模型。严格复现实验时应配置支持 `gpt-4o-mini` 的评估 API；临时检查流程时可以传入当前接口支持的模型，但必须在结果中记录这一偏差。

### Milvus 向量维度不匹配

当前代码固定使用 1536 维 embedding。不要复用由1024维或其他维度创建的旧 Milvus 数据库；使用新的 `--output-dir` 重新运行。

### Qualitative Analysis 中断

完整的 qualitative trajectory 会直接复用，不会重复调用模型。若运行在中途失败，脚本不会从非空画像或非空 Milvus 数据库继续，以免破坏“从零开始”的协议；请换一个新的 `--output-dir` 重新运行。该问题不影响 `prepare -> generate -> evaluate` 主结果。

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
- [ ] `uv pip check` 显示所有依赖兼容。
- [ ] 评估使用 `uv run --no-sync`，没有触发依赖回滚。
- [ ] 四个 Hugging Face 评估模型已缓存，离线加载正常。
- [ ] 记录 judge model 和三个 CardiffNLP 模型名称。
- [ ] 最终 `table2_main_results.md` 包含原论文两行和 `Ours`。
- [ ] 正式论文结果覆盖全部目标 speakers，而不是单个 case。

## 12. Windows PowerShell附录

如果3090机器运行Windows，可以使用下面的PowerShell命令。实验参数和Linux版本完全一致。

### 12.1 环境部署

```powershell
cd E:\01_Research\03_UserProfile\memory
uv python install 3.11
uv sync --locked --python 3.11
uv pip install "torch==2.7.1" --torch-backend=cu118
uv pip install --reinstall "transformers==4.57.6" "tokenizers==0.22.2" "huggingface-hub==0.36.2" "bert-score==0.3.13"
```

### 12.2 单个对话

```powershell
uv run python -m src.experiments.exp2_user_modeling `
  --phase prepare `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --config config.qwen-plus.ini `
  --prompt-version v3_realtalk_aligned `
  --output-dir data/exp2_user_modeling

uv run python -m src.experiments.exp2_user_modeling `
  --phase generate `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --config config.qwen-plus.ini `
  --prompt-version v3_realtalk_aligned `
  --output-dir data/exp2_user_modeling

$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
uv run --no-sync python -m src.experiments.exp2_user_modeling `
  --phase evaluate `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --config config.qwen-plus.ini `
  --prompt-version v3_realtalk_aligned `
  --output-dir data/exp2_user_modeling `
  --eval-device cuda:0 `
  --eval-batch-size 16 `
  --judge-model gpt-4o-mini

uv run python -m src.experiments.exp2_user_modeling_qualitative `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --config config.qwen-plus.ini `
  --output-dir data/exp2_user_modeling
```

### 12.3 批量运行

批量运行时去掉 `--case`，依次执行 `prepare`、`generate` 和 `evaluate` 即可。

## 13. Prompt 版本、独立 Judge API 与 turn 级曲线

### 13.1 不可混用的版本

主实验现在同时记录三类版本：

- `protocol_version`：状态传递、teacher forcing、训练/测试划分等实验协议。
- `generation_prompt_version`：回复生成与 empathy alignment 使用的 prompt bundle。
- `evaluation_prompt_version`：固定的 REALTALK Table 2 judge prompt。

可用生成 prompt 版本：

```text
v1_baseline          已有旧结果对应的基线提示词，不更新 current_state
v2_state_update      上一轮共情状态 + 每轮 current/projected state 更新
v3_realtalk_aligned  在 v2 基础上适配 REALTALK 对话行为与 EI 判定边界（默认）
v4_task_reframed     完全重写的任务优先提示词；独立校准角色、内容和对话行为，不继承 v1-v3 文本
```

`v4_task_reframed` 不是在 v3 后追加规则：回复 prompt 与 alignment prompt 均从零重写，但保持核心算法所需的并行执行方式以及 `understanding`、`prediction`、`empathy_state`、`state_update` 输出契约。v4 暂不设为默认版本，必须显式传入并使用独立输出目录。

每条 prediction 和最终 manifest 都保存 prompt 版本及 SHA256。不同版本必须使用不同的 `--output-dir`，程序会拒绝把不同版本追加到同一个 `predictions.jsonl`。

示例：

```bash
uv run python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
  --output-dir data/exp2_user_modeling/v3_realtalk_aligned
```

已有旧 predictions 没有 prompt metadata 时只按 `v1_baseline` 处理。不要把旧目录用于 v2、v3 或 v4。

v4 单独运行示例。为只比较 prompt 版本，应复用同一次 `prepare` 产生的用户画像和智能体人设，不要重新抽取。但只能复制每个 case 的 `assets/`，不能复制旧版本的 `generations/`、`states/`、`evaluation/` 或 `memory/`。

源目录可以是 prepare-only 目录，也可以是已经跑过 generate 的旧版本目录，但**不能复制旧的 `user_profile_runtime.json`**，因为其中可能带有旧版本结束时的用户状态。下面的脚本只复用不可变的 `agent_persona.json`、`user_profile.json` 和 `asset_manifest.json`，并根据静态画像为 v4 重建全新的 runtime profile：

```bash
ASSET_SOURCE=data/exp2_shared_assets
V4_DIR=data/exp2_qwen_plus_v4

uv run --no-sync python - "$ASSET_SOURCE" "$V4_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
source_cases = source / "cases"

if not source_cases.is_dir():
    raise SystemExit(f"asset source does not exist: {source_cases}")

old_predictions = sorted(target.glob("cases/*/generations/predictions.jsonl"))
if old_predictions:
    joined = "\n".join(str(path) for path in old_predictions)
    raise SystemExit(
        "target already contains predictions; use a new target or move it to a backup:\n"
        + joined
    )

copied = 0
for source_case in sorted(path for path in source_cases.iterdir() if path.is_dir()):
    source_assets = source_case / "assets"
    required = [
        source_assets / "agent_persona.json",
        source_assets / "user_profile.json",
        source_assets / "asset_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            f"incomplete assets for {source_case.name}: " + ", ".join(missing)
        )

    target_assets = target / "cases" / source_case.name / "assets"
    target_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(required[0], target_assets / "agent_persona.json")
    shutil.copy2(required[1], target_assets / "user_profile.json")

    with required[1].open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    runtime = {
        "state_axis": {
            "static_profile": profile,
            "current_state": {},
            "projected_state": {},
        },
        "context_axis": {},
    }
    with (target_assets / "user_profile_runtime.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(runtime, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with required[2].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["profile_path"] = str((target_assets / "user_profile.json").resolve())
    manifest["runtime_profile_path"] = str(
        (target_assets / "user_profile_runtime.json").resolve()
    )
    manifest["persona_path"] = str((target_assets / "agent_persona.json").resolve())
    with (target_assets / "asset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    copied += 1

print(f"prepared clean assets for {copied} cases in {target}")
PY
```

复制后可以检查目标目录；不应出现旧的 `predictions.jsonl`：

```bash
find "$V4_DIR/cases" -path '*/generations/predictions.jsonl' -print
```

当前版本的 `generate` 会自动为每个 case 创建新的 `memory/`、`generations/`、`states/` 和 `evaluation/` 目录；复制 assets 时不需要手工创建这些目录。若服务器尚未同步该修复、运行时出现 `Open local milvus failed, dir: .../memory not exists`，可以先用下面的兼容命令创建空目录，再重新执行完全相同的生成命令：

```bash
for CASE_DIR in "$V4_DIR"/cases/*; do
  [ -d "$CASE_DIR" ] || continue
  mkdir -p "$CASE_DIR"/{memory,generations,states,evaluation}
done
```

该命令只创建缺失目录，不会修改或删除画像、人设、已有生成结果或评估结果。同步当前版本代码后不再需要执行它。

如果目标目录包含旧 predictions，其中记录的 `prompt_version` 或 `prompt_sha256` 与 v4 不同，程序会主动报错，防止不同版本结果混写。不要手工把旧记录的版本字段改成 v4。如果没有干净的 prepared assets，才对新的 v4 输出目录重新运行一次 `--phase prepare`。

保留生成和评估分开执行时，生成命令为：

```bash
uv run python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v4_task_reframed \
  --output-dir data/exp2_user_modeling/v4_task_reframed
```

也可以将上面的 `--phase generate` 改为 `--phase generate-evaluate`，并补充评估参数，使生成完成后直接开始评估。

### 13.2 用户状态时序

第 `t` 轮回复 prompt 在当前 alignment 启动前冻结，因此使用第 `t-1` 轮已经完成的 `current_state` 和 `previous_empathy_state`。当前回复与第 `t` 轮 alignment 并行；alignment 完成后把固定结构的 `current_state` 与 `projected_state` 写入 runtime profile，供第 `t+1` 轮使用。长期静态用户画像在主实验测试阶段保持不变。

### 13.3 固定智能体人设

智能体人设必须与 `dataset/lx_agent.json` 的层级和字段语义一致，schema 版本为 `lx_agent_v1`；JSON 键和字段内容统一使用英文。旧的 `name/personality/tone/...` 或 `meta_info/strategy_layer/...` 人设文件不能与新结果混用；请换新的输出目录重新执行 `prepare`。

### 13.4 GPT-4o-mini 使用独立中转 API

`config.ini` 中的 `[API]` 只用于算法生成；`[EvaluationAPI]` 只用于 Reflective、Grounding 和 Empathy judge。当前评估后端配置为：

```ini
[EvaluationAPI]
model = gpt-4o-mini
backend = zhizengzeng
base_url = https://api.zhizengzeng.com/v1
api_key_env = EVAL_API_KEY
```

运行前只在环境变量中提供密钥，不要把密钥写入仓库：

```bash
export EVAL_API_KEY="your-key"
```

评估命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v3_realtalk_aligned \
  --output-dir data/exp2_user_modeling/v3_realtalk_aligned \
  --judge-config-section EvaluationAPI \
  --eval-device cuda:0 \
  --eval-batch-size 16
```

Judge 缓存指纹包含模型、API 后端、base URL、评估 prompt 和本地分类器名称；缓存 ID 另外包含候选回复与完整上下文哈希，因此不会误用此前模型、prompt、回复或上下文不同的标注。

### 13.5 turn 级 qualitative 曲线

qualitative 脚本仍只重放前 90% 的真实数据集对话，不生成智能体回复。现在每个合并后的连续同 speaker dialogue bubble 记为一个 turn，并在每个 turn 后记录原始 completeness、entropy、profile hash 和 profile version。完整画像只在画像真实变化时保存快照。

单对话底层横轴使用真实 `turn_index`；多对话聚合图使用归一化训练 turn 进度 `0%–100%`。图中淡线为真实 per-turn 均值，主线为只使用当前及历史数据的 causal EWMA，可视化平滑不改变原始指标；单对话图额外标出真实画像更新点和 Session 边界。
