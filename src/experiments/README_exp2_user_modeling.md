# Experiment 2：User Modeling Evaluation

本文件说明实验二的设计、3090 环境部署、主实验运行、Table 2 评估、Prompt 版本实验和定性曲线。

当前有两个相互独立的入口：

```text
src/experiments/exp2_user_modeling.py              # 主实验与 Table 2
src/experiments/exp2_user_modeling_qualitative.py  # 画像进化与画像熵
```

主实验流程：

```text
前 90% Session：prepare 用户画像和智能体人设
后 10% Session：generate 智能体回复 → evaluate Table 2 指标
```

定性分析不生成智能体回复，也不计算 Table 2；它从空画像开始重放前 90% 的真实对话。

---

## 1. 当前推荐命令

### 1.1 V13 干净表述 V2 复验（推荐先跑）

`v13_grounding_precision_clean` 的首次复验失败：24 条中 23 条包含问句，Grounding 为 `0.44`，Empathy 为 `1.65`。该版本及其结果继续保留，不覆盖。

`v13_grounding_precision_clean_v2` 以原始 `v13_grounding_precision` 为母版，只删除 `V7`、`Grounding`、指标和调参过程等实验性措辞，保留全部有效行为约束：默认不追问、回答可以结束当前轮、仅在必要且符合角色习惯时问一个具体问题、禁止泛化情绪探索。三个版本共享同一 response user Prompt、V5 alignment、状态更新、模型配置和评估 Prompt。

先在 `diagnostic3` 的同一 24 条回复上运行干净版：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
SWEEP_ROOT=data/exp2_v13_clean_v2_diagnostic3 \
CASE_SET=diagnostic3 \
VERSIONS_CSV=v13_grounding_precision_clean_v2 \
GENERATE_WORKERS=3 \
bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

结果位于：

```text
data/exp2_v13_clean_v2_diagnostic3/v13_grounding_precision_clean_v2/evaluation/table2_main_results.md
data/exp2_v13_clean_v2_diagnostic3/prompt_sweep_summary.md
```

旧 V13 在这 24 条上的 Grounding 为 `0.6210`，混淆计数为 `TN/FP/FN/TP = 8/7/2/7`。建议仅当干净版 Grounding 不低于 `0.59`、假阳性不超过 8、假阴性不超过 3，并且 Semantic 不低于 `0.82`、Empathy 不高于 `1.25` 时，再运行全量 10 个对话：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
SWEEP_ROOT=data/exp2_v13_clean_v2_full \
CASE_SET=all \
VERSIONS_CSV=v13_grounding_precision_clean_v2 \
GENERATE_WORKERS=3 \
bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

### 1.2 V11–V15 定向 Prompt 实验

当前 V5 的用户画像和人设已经准备完成，V6–V10 也已经完成。运行 V11–V15 时不需要重新执行 `prepare`：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
SWEEP_ROOT=data/exp2_prompt_sweep_v11_v15_directed \
CASE_SET=diagnostic3 \
bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

`diagnostic3` 只运行 Chat 1、Chat 4 和 Chat 9，共 24 条测试回复。该子集是根据完整 V7 的逐样本结果选择的：四个待优化指标、问句/Emoji 比例以及 Reflective/Grounding 假阳性率均接近全量 117 条，同时保留 elise、Paola 和 Fahim Khan 三种差异明显的目标角色。脚本依次运行 V11、V12、V13、V14 四个专项版，最后运行集成版 V15。中断后重新执行同一命令即可续跑。

结果位于：

```text
data/exp2_prompt_sweep_v11_v15_directed/prompt_sweep_summary.md
data/exp2_prompt_sweep_v11_v15_directed/prompt_sweep_summary.json
data/exp2_prompt_sweep_v11_v15_directed/<prompt-version>/evaluation/table2_main_results.md
data/exp2_prompt_sweep_v11_v15_directed/<prompt-version>.log
```

### 1.3 服务器后台运行

为了避免 VS Code、SSH 或局域网断开后实验停止，使用 `tmux`：

```bash
cd ~/fxw/memory
tmux new -s exp2-sweep
export PATH="$HOME/.local/bin:$PATH"
```

在 tmux 中执行实验命令。按 `Ctrl+B`，再按 `D` 退出；重新进入：

```bash
tmux attach -t exp2-sweep
```

---

## 2. 实验设计

### 2.1 Research Question

> RQ2：Does explicit user modeling enable better personalized interactions?

### 2.2 数据与切分

- 数据集：REALTALK 的 `dataset/Chat_*.json`。
- 每个对话按 Session 时间顺序进行 9:1 划分。
- 前 90% Session：一次性抽取被建模用户的用户画像，以及目标智能体的人设。
- 后 10% Session：生成目标智能体回复并评估。
- 用户画像和人设只读取训练 Session。
- 主实验测试阶段不在线更新长期静态画像，但每轮更新短期 `current_state` 和 `projected_state`。

当前代码在每个 REALTALK 文件中使用：

```text
speaker_1 = 被建模用户
speaker_2 = 目标智能体
```

因此当前结果覆盖作为 `speaker_2` 出现的目标角色。如果论文最终要求两个方向都作为目标角色，需要另行补充相反方向；单方向结果不能自动解释为双向全覆盖。

### 2.3 测试时交互

测试使用 teacher forcing：

1. 当前用户输入来自 REALTALK 数据集。
2. 目标智能体根据真实历史、训练集用户画像、训练集智能体人设和上一轮状态生成回复。
3. 为保证所有方法获得相同历史，进入下一轮时写入数据集中的真实目标角色回复，而不是 Ours 刚生成的回复。
4. Ours 的回复只用于评估，不会污染后续生成历史。

生成模型不能看到当前目标回复和未来用户消息。每条 prediction 中的 `generation_input_audit` 会记录泄漏审计信息。

### 2.4 状态时序

第 `t` 轮遵循以下时序：

```text
用第 t-1 轮已经完成的状态生成第 t 轮回复
                 ∥
根据第 t 轮用户消息运行 alignment
                 ↓
将第 t 轮状态保存给第 t+1 轮
```

回复生成与当前轮 alignment 并行。当前 alignment 不会反向影响已经开始生成的当前回复；其结果供下一轮使用。这是核心算法的既定并行设计。

### 2.5 固定画像与人设结构

用户画像采用 5 层、21 个固定字段，与 `dataset/lsy_user.json` 对齐；字段名固定，内容由训练对话抽取。

智能体人设采用固定英文 schema，与 `dataset/lx_agent.json` 对齐：

```text
core_layer
capability_layer
expression_layer
```

当前 schema 版本为 `lx_agent_v3_behavior_calibrated_no_catchphrases`。`expression_layer.catchphrases` 已删除，避免少量历史句子被机械复用。生成前会校验字段结构、schema 版本和抽取 Prompt SHA256。

---

## 3. Table 2 评估

最终表格复用 REALTALK Table 2 的两条论文基线，并增加 `Ours`：

| 分组 | 指标 | 评估方法 | 方向 |
|---|---|---|---|
| Content Similarity | Lexical | ROUGE-L F1 | 越高越好 |
| Content Similarity | Semantic | BERTScore F1，`roberta-large` | 越高越好 |
| Message-level EI | Reflective | reference/generated 二分类标签一致率 | 越高越好 |
| Message-level EI | Grounding | reference/generated 二分类标签一致率 | 越高越好 |
| Message-level EI | Sentiment | CardiffNLP 情感标签一致率 | 越高越好 |
| Message-level EI | Emotion | CardiffNLP 情绪标签一致率 | 越高越好 |
| Message-level EI | Intimacy | 两侧分数绝对差 | 越低越好 |
| Message-level EI | Empathy | 两侧 EPITOME 总分绝对差 | 越低越好 |

### 3.1 指标含义

- **Lexical**：生成回复与真实回复的最长公共子序列重合程度。它衡量表层措辞接近程度，不等同于自然性或正确性。
- **Semantic**：BERTScore 衡量上下文语义接近程度，允许不同措辞表达相近意思。
- **Reflective**：判断回复是否包含对说话者自身感受、动机或行为模式的真实自我观察。普通事实、偏好或意见不自动算反思。该指标衡量两侧是否一致，不是反思越多越好。
- **Grounding**：判断回复是否通过具体澄清、确认或紧接前文的相关追问建立共同理解。任意问题或普通认可不自动算 Grounding。
- **Sentiment**：生成回复和真实回复的正向、中性、负向标签是否一致。
- **Emotion**：生成回复和真实回复的主要情绪类别是否一致。
- **Intimacy**：两条回复亲密度分数的绝对误差。它不是生成回复本身的亲密度；过高或过低都会增加误差。
- **Empathy**：分别从 Emotional Reaction、Interpretation、Exploration 三项给出 0–2 分，总分范围 0–6，再计算两侧总分绝对差。过度共情和共情不足都会增加误差。

Table 2 衡量的是 Ours 对真实人物回复属性的**模拟一致性**，不是通用回复质量、帮助性或安全性的独立评分。

### 3.2 逐样本评估

每个评测点包含：

```text
reference = REALTALK 中的真实目标角色回复
generated = Ours 对同一用户输入生成的回复
```

两条候选共享从当前 Session 开头到目标回复之前的相同真实历史。评估器不会看到目标回复之后的消息，也不会把 reference 作为 generated 的评分提示。

| 指标 | 评估器 | 单样本计算 |
|---|---|---|
| Lexical | 项目 ROUGE-L 实现 | `ROUGE-L_F1(reference, generated)` |
| Semantic | BERTScore / `roberta-large` | `BERTScore_F1(reference, generated)` |
| Reflective | GPT judge | 两侧分别标为 `True/False`，相同记 1 |
| Grounding | GPT judge | 两侧分别标为 `True/False`，相同记 1 |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 最高概率标签相同记 1 |
| Emotion | `cardiffnlp/twitter-roberta-large-emotion-latest` | 最高概率标签相同记 1 |
| Intimacy | `cardiffnlp/twitter-roberta-large-intimacy-latest` | `abs(reference - generated)` |
| Empathy | GPT judge / EPITOME | `abs(total_reference - total_generated)` |

离散指标：

```text
score_i = 1[label_reference == label_generated]
```

连续误差指标：

```text
error_i = abs(value_reference - value_generated)
```

Reflective、Grounding 和 Empathy 使用固定的 REALTALK 适配评估 Prompt，位于：

```text
src/prompts/eval_templates_en.py
```

评估 Prompt 依据 REALTALK Appendix C.1–C.3 和 EPITOME 分档定义编写，不复用项目中旧的通用 EI Prompt。生成 Prompt 的版本实验不会修改评估 Prompt。

### 3.3 聚合方法

最终结果不是简单平均 10 个对话，也不是对全部消息直接微平均，而是：

1. 合并同一目标 speaker 的全部测试样本。
2. 对每位目标 speaker 分别计算每项指标的样本均值。
3. 对所有目标 speaker 做宏平均。
4. 计算目标 speaker 均值之间的总体标准差。

```text
mean per target speaker → macro mean and population std across speakers
```

因此 `mean ± std` 表示目标角色之间的平均表现和角色差异，不是多次随机运行的误差或置信区间。

---

## 4. 3090 环境部署

以下命令以 Linux、Bash 和 Python 3.11 为准。

### 4.1 创建 uv 环境

```bash
cd ~/fxw/memory
uv python install 3.11
uv sync --locked --python 3.11
```

不需要手动激活 `.venv`。

### 4.2 安装评估依赖

当前服务器为 NVIDIA Driver `470.256.02`，系统显示 CUDA 11.4，因此使用 PyTorch CUDA 11.8 wheel，不要安装 CUDA 12.x wheel：

```bash
uv pip install "torch==2.7.1" --torch-backend=cu118
uv pip install --reinstall \
  "transformers==4.57.6" \
  "tokenizers==0.22.2" \
  "huggingface-hub==0.36.2" \
  "bert-score==0.3.13"
```

当前已验证组合：

```text
torch             2.7.1+cu118
transformers      4.57.6
tokenizers        0.22.2
huggingface-hub   0.36.2
bert-score        0.3.13
```

这些评估依赖尚未完全由项目锁文件固定。普通 `uv run` 可能重新同步传递依赖并把 Tokenizers 改回不兼容版本；涉及评估的命令必须使用：

```bash
uv run --no-sync ...
```

如果之后执行了 `uv sync`，重新执行本节的安装命令。

### 4.3 环境检查

```bash
uv pip check
uv pip show torch transformers tokenizers huggingface-hub bert-score

uv run --no-sync python -c \
  "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

服务器上预期：

```text
2.7.1+cu118
11.8
True
NVIDIA GeForce RTX 3090
```

### 4.4 Hugging Face 模型

评估使用四个本地模型：

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
cardiffnlp/twitter-roberta-large-emotion-latest
cardiffnlp/twitter-roberta-large-intimacy-latest
roberta-large
```

3090 当前已经缓存这些模型，因此正式评估使用：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ...
```

新机器第一次下载模型时暂时去掉这两个环境变量；缓存完整后再使用离线模式。

---

## 5. API 与模型配置

候选回复生成和 Table 2 Judge 使用相互独立的配置：

```ini
[API]
model = qwen-plus
embedding_model = text-embedding-v4
enable_thinking = False

[EvaluationAPI]
model = gpt-4o-mini
backend = zhizengzeng
base_url = https://api.zhizengzeng.com/v1
api_key_env = EVAL_API_KEY
```

- `[API]`：用户画像、人设、alignment 和 Ours 回复生成。
- `[EvaluationAPI]`：只用于 Reflective、Grounding 和 Empathy Judge。
- `--judge-model` 可以覆盖 `[EvaluationAPI].model`。
- `--judge-workers` 控制远程 LLM-as-Judge 并发数，默认 `6`；不改变评估 Prompt、标签定义或缓存指纹。
- 严格与 REALTALK 对比时使用 `gpt-4o-mini`；换成其他 Judge 必须记录为实验偏差。
- 当前记忆 embedding 固定为 1536 维，embedding 服务也必须返回 1536 维向量。

运行前设置密钥，不要提交到仓库：

```bash
export EVAL_API_KEY="your-key"
```

生成 API 的密钥和 base URL 按 `config.qwen-plus.ini` 当前字段配置。

---

## 6. 主实验运行

### 6.1 参数一致性

同一实验版本的 `prepare`、`generate` 和 `evaluate` 必须保持以下参数一致：

```text
--train-ratio
--case（如果指定）
--config
--prompt-version
--output-dir
```

不同 Prompt 版本必须使用不同输出目录。每条 prediction 都记录 Prompt 版本和 SHA256；程序会拒绝把不同版本写进同一个目录。

### 6.2 单个对话：分阶段运行

以下以 V5 和 `Chat_1_Emi_Elise.json` 为例。

准备训练 assets：

```bash
uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase prepare \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v5_relationship_calibrated \
  --output-dir data/exp2_single_v5 \
  --generate-workers 1
```

生成测试回复：

```bash
uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase generate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v5_relationship_calibrated \
  --output-dir data/exp2_single_v5
```

计算 Table 2：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase evaluate \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v5_relationship_calibrated \
  --output-dir data/exp2_single_v5 \
  --judge-config-section EvaluationAPI \
  --judge-model gpt-4o-mini \
  --judge-workers 6 \
  --eval-device cuda:0 \
  --eval-batch-size 16
```

### 6.3 全部对话：从头运行

不传 `--case` 就会处理 `dataset/Chat_*.json` 的全部对话。

```bash
V5_DIR=data/exp2_qwen_plus_v5_clean

uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase prepare \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v5_relationship_calibrated \
  --output-dir "$V5_DIR"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -m src.experiments.exp2_user_modeling \
  --phase generate-evaluate \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --prompt-version v5_relationship_calibrated \
  --output-dir "$V5_DIR" \
  --generate-workers 3 \
  --judge-config-section EvaluationAPI \
  --judge-model gpt-4o-mini \
  --judge-workers 6 \
  --eval-device cuda:0 \
  --eval-batch-size 16
```

`generate-evaluate` 只是依次执行生成和评估。原来的 `generate`、`evaluate` 仍可单独运行。

### 6.4 生成与评估并发

`--generate-workers` 按完整 conversation 并发生成，默认值为 `3`。同一 conversation 内的 turn 仍严格按数据集顺序执行；每个 turn 的回复生成和 alignment 继续并行。因此 `--generate-workers 3` 的生成阶段峰值约为 6 个远程 API 请求。

每个生成进程使用独立的 case 目录和 Milvus 数据库：

```text
cases/<case_id>/memory/memory.db
```

实验在构造 Agent 时直接注入该数据库，不再先打开共享的 `data/milvus_memory.db`。普通 App 未显式传入 memory manager 时仍保持原有行为。

- 单个 `--case` 实际只启动 1 个生成进程，即使参数为 3。
- 推荐3090批量实验使用 `--generate-workers 3`。
- 如果生成API出现频繁限流，将其降为 `2` 或 `1`。
- 不要并发同一 conversation 的turn；后续turn依赖上一轮状态和记忆。
- 中断后重新运行相同命令，各case根据已有prediction独立续跑。
- `--generate-workers` 只改变调度，不改变Prompt、缓存指纹或评估协议。

`--judge-workers` 只控制远程 Reflective、Grounding 和 Empathy Judge，默认值为 `6`。本地GPU分类器和BERTScore仍由单进程执行。

所有阶段会输出适合 `tmux`、`tee` 和日志文件的文本进度条，例如：

```text
[Prepare]        [##############--------------] 5/10 ( 50.0%) ...
[Generate]       [#################-----------] 72/117 ( 61.5%) ...
[Judge]          [######################------] 18/24 ( 75.0%) ...
[Evaluate cases] [#########################---] 9/10 ( 90.0%) ...
```

生成进度以成功写入prediction和state并完成该turn记忆步骤为准。断点续跑时会先读取已有有效缓存，因此进度会从已有数量开始，不会重新从0计算。由于多个生成进程还有token、memory等诊断日志，进度条采用“一次完成输出一行”，不使用会污染日志的单行覆盖动画。

### 6.5 主要输出

```text
data/<experiment>/
├── split_manifest.json
├── run_manifest.json
├── cases/<case_id>/
│   ├── assets/
│   │   ├── user_profile.json
│   │   ├── user_profile_runtime.json
│   │   ├── agent_persona.json
│   │   └── asset_manifest.json
│   ├── generations/predictions.jsonl
│   ├── states/user_understanding.jsonl
│   ├── memory/memory.db
│   └── evaluation/
│       ├── table2_annotations.jsonl
│       └── table2_scores.json
└── evaluation/
    ├── table2_main_results.json
    └── table2_main_results.md
```

`table2_main_results.md` 是 REALTALK 两条原始基线加 `Ours` 的最终表格。

---

## 7. Prompt 版本管理

实验同时记录三类版本：

- `protocol_version`：数据切分、teacher forcing 和状态传递协议。
- `generation_prompt_version`：回复生成与 alignment Prompt bundle。
- `evaluation_prompt_version`：固定的 REALTALK Table 2 Judge Prompt。

### 7.1 V1–V5

| 版本 | 作用 |
|---|---|
| `v1_baseline` | 旧基线，不更新 current state |
| `v2_state_update` | 增加上一轮 empathy state 和每轮状态更新 |
| `v3_realtalk_aligned` | 适配 REALTALK 行为与 EI 边界；当前代码默认值 |
| `v4_task_reframed` | 从零重写任务导向 Prompt，不继承 V1–V3 文本 |
| `v5_relationship_calibrated` | 按关系距离和角色行为频率校准；当前可复用 assets 来源 |

V5 改变过智能体人设抽取协议，因此 V5 的 assets 不能由 V1–V4 assets 冒充。当前标准 V5 assets 位于：

```text
data/exp2_qwen_plus_v5_clean
```

### 7.2 V6–V10 已完成实验

V6–V10 共用 V5 assets、V5 alignment、状态更新、teacher forcing 和评估 Prompt，只改变最终回复 Prompt。

| 版本 | 控制变量 |
|---|---|
| `v6_last_topic_plain` | 只回应最后活跃话题 |
| `v7_recent_style_imitation` | 模仿近期真实目标角色回复的表层风格 |
| `v8_frequency_hard_gate` | rare/occasional 行为硬门控 |
| `v9_evidence_bound_persona` | 严格限制人设事实和个人经历使用 |
| `v10_balanced_surface_act` | 综合话题、风格、动作和证据规则 |

五版均完整运行 10 个对话、117 条回复。主要结果：

| 版本 | Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V5 | 0.1047 | 0.8320 | 0.7175 | 0.4842 | 0.5345 | 0.4694 | 0.0774 | 1.2668 |
| V7 | 0.1012 | 0.8302 | 0.7171 | 0.5661 | 0.6511 | 0.4539 | 0.0703 | 1.1079 |

V7 是该组最佳综合版本。V10 虽把 Reflective 提升到 0.7231，但 Empathy error 恶化到 1.6758，说明把多种全局规则叠在一起并不稳定。

完整结果：

```text
data/exp2_prompt_sweep_v6_v10/prompt_sweep_summary.md
```

如需重新运行旧 sweep：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
SWEEP_ROOT=data/exp2_prompt_sweep_v6_v10 \
CASE_SET=all \
bash scripts/run_exp2_prompt_sweep.sh
```

### 7.3 V11–V15 当前实验

V11–V15 直接针对完整 V7 中仍未达到论文结果的四个指标。每个专项版都以 V7 的具体错误方向为依据，而不是泛化地增加风格规则：

| 版本 | 专项目标 | V7 错误方向 | Prompt 调整 | 保护指标 |
|---|---|---|---|---|
| `v11_lexical_fidelity` | Lexical | 语义正确但引入新实体、新经历和自由联想 | 减少无依据具体内容；自然使用当前对话词汇；保留有证据的自我披露和换题 | Semantic、Sentiment |
| `v12_reflective_placement` | Reflective | 14 条假阳性、12 条假阴性；总量接近但位置错误 | 总体频率不做单向增减；只纠正哪些轮次应出现真正自我观察 | Empathy、Intimacy |
| `v13_grounding_precision` | Grounding | 38 条假阳性、11 条假阴性；V7 问句 78 条而 reference 37 条 | 明确减少习惯性追问；仅保留必要澄清、确认和有近期行为证据的具体 follow-up | Reflective、Empathy |
| `v13_grounding_precision_clean` | 首次干净复验（失败，保留） | “问题可选”的软约束导致 23/24 条回复追问 | 不再用于后续实验 | 历史结果保留 |
| `v13_grounding_precision_clean_v2` | V13 语义等价复验 | 只删除实验痕迹，不删除有效行为约束 | 默认不追问；回答可以结束；追问必须必要且符合角色；禁止泛化情绪探索 | 与旧 V13 全指标对照 |
| `v14_emotion_calibration` | Emotion | joy 82 条而 reference 65 条；Emoji 66 条而 reference 2 条 | 按角色和当前场景校准情绪；减少无依据 joy、Emoji 和装饰性积极表达，不全局压低 Paola 等真实高 joy 角色 | Sentiment、Empathy |
| `v15_metric_integrated` | 四项集成 | 同时存在上述四类错误 | 使用“内容→反思→Grounding→情绪”的短决策流，不拼接四份专项 Prompt | Semantic、Sentiment、Intimacy、Empathy |

五版继续共享相同的 response user Prompt、V5 alignment、状态更新、teacher forcing、模型参数和评估 Prompt。V11–V14 只改变各自的定向策略；V15 从同一 V7 审计中预先集成四个方向，但不预设四个专项都会有效。

V13 的两个 clean 版本均不会覆盖旧 V13，且必须使用独立 `SWEEP_ROOT`。脚本的 `VERSIONS_CSV` 可只选择一个或多个版本，例如：

```bash
VERSIONS_CSV=v13_grounding_precision_clean_v2 bash scripts/run_exp2_prompt_sweep_v11_v15.sh
VERSIONS_CSV=v13_grounding_precision_clean_v2,v14_emotion_calibration bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

当前默认测试子集是：

```text
Chat_1_Emi_Elise.json    9 条
Chat_4_Emi_Paola.json    8 条
Chat_9_Fahim_Akib.json   7 条
合计                    24 条
```

该子集与 V7 全量结果的关键对照：

| 统计 | diagnostic3 | V7 全量 |
|---|---:|---:|
| Lexical | 0.0981 | 0.1012 |
| Reflective | 0.7308 | 0.7171 |
| Grounding | 0.5470 | 0.5661 |
| Emotion | 0.4815 | 0.4539 |
| Sentiment | 0.6508 | 0.6511 |
| Grounding 假阳性率 | 33.3% | 32.5% |
| Reflective 假阳性率 | 12.5% | 12.0% |
| 生成问句率 | 70.8% | 66.7% |
| 生成 Emoji 率 | 58.3% | 56.4% |

运行：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
SWEEP_ROOT=data/exp2_prompt_sweep_v11_v15_directed \
CASE_SET=diagnostic3 \
bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

其他可选子集：

```bash
CASE_SET=fast2 bash scripts/run_exp2_prompt_sweep_v11_v15.sh       # Chat 2 + Chat 10，23 条
CASE_SET=balanced3 bash scripts/run_exp2_prompt_sweep_v11_v15.sh  # Chat 2 + Chat 5 + Chat 10，47 条
```

只有在定向子集确认 Prompt 值得继续后，再运行全部 10 个对话：

```bash
CASE_SET=all bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

自定义 case：

```bash
CASE_LIST=Chat_1_Emi_Elise.json,Chat_5_Nicolas_Nebraas.json \
bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

### 7.4 V18：Reflective与Grounding联合门控

`v18_reflective_grounding_joint_gate`以V7为唯一生成基线，只替换最终回复的system Prompt。以下输入和流程保持不变：

- 完整的固定字段`user_profile.json`，不筛选画像；
- 原固定字段`agent_persona.json`，不追加行为统计；
- V7使用的response user Prompt；
- V5 alignment、上一轮状态传递、teacher forcing和评估Prompt；
- 生成模型、温度和max tokens。

V7的Reflective错误主要是出现位置不对，不能单向增加反思；Grounding主要是假阳性和习惯性追问过多。V18因此先在内部互斥选择一种主要回复动作：真实自我观察、具体Grounding或普通回复。证据不足时选择普通回复；不会把V12和V13两段Prompt直接拼接。

全量运行：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
BEST_FULL_DIR=data/exp2_v16_full/v16_v7_selective_followup \
SWEEP_ROOT=data/exp2_v18_reflective_grounding \
CASE_SET=all \
VERSIONS_CSV=v18_reflective_grounding_joint_gate \
GENERATE_WORKERS=3 \
bash scripts/run_exp2_prompt_sweep_v11_v15.sh
```

论文目标为Reflective `0.77 ± 0.09`、Grounding `0.62 ± 0.08`。脚本报告继续同时列出论文两行、V7、V7之后已接受的全量最好版本和V18。

### 7.5 V19–V22：历史定向实验

V19–V22是已经完成的V7定向实验，版本定义必须与已生成结果保持一致，不再回写或改名。四版分别测试Reflective触发、Grounding具体性、两类动作独立判断和近期动作模仿。全量结果表明继续在V7后追加规则不能稳定合并V7与V18的优势，因此后续不再以这些Prompt作为叠加基础。

| 版本 | 定向变化 | 主要目的 |
|---|---|---|
| `v19_reflective_trigger_recall` | 明确Reflective触发条件 | Reflective专项诊断 |
| `v20_grounding_specificity_gate` | 问句绑定用户具体细节 | Grounding专项诊断 |
| `v21_independent_act_decisions` | 独立判断Reflective和Grounding | 两类动作联合诊断 |
| `v22_recent_act_imitation` | 根据近期真实角色回复选择动作 | 行为证据诊断 |

把已完成的V18全量结果作为专项参考行，运行四版全量实验：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
BEST_FULL_DIR=data/exp2_v16_full/v16_v7_selective_followup \
FULL_REFERENCE_DIRS_CSV=data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate \
SWEEP_ROOT=data/exp2_prompt_sweep_v19_v22 \
CASE_SET=all \
GENERATE_WORKERS=3 \
bash scripts/run_exp2_prompt_sweep_v19_v22.sh
```

V19–V22包装脚本已经默认把正确的联合门控V18作为全量参考；显式设置`FULL_REFERENCE_DIRS_CSV`可以覆盖该路径。需要保留多个全量专项强版本时，用逗号分隔目录。最终汇总会按以下顺序保留：论文两行、V7、`BEST_FULL_DIR`、V18、当前V19–V22。最后一次汇总会包含本批全部版本，而不再只显示最后运行的一版。

Sweep运行期间，每完成一个版本都会重写一次`prompt_sweep_summary.md`。中间报告会累计列出本次命令中所有已经产生`table2_main_results.json`的版本，而不是只保留刚完成的一版；因此即使后续版本失败或任务中断，前面已经完成的结果行也会保留。

### 7.6 V23–V24：独立择优Prompt

V23和V24不通过`V7 + addendum`或`V18 + addendum`构造，两个system Prompt均为独立完整文本：

| 版本 | 保留内容 | 唯一差异 |
|---|---|---|
| `v23_selected_style_joint_gate` | 近期角色表面风格、一个当前回应点、Reflective/Grounding/Ordinary三选一 | 最小择优版本 |
| `v24_selected_gate_empathy_independence` | 与V23相同的择优结构 | 明确所选动作不决定共情强度，上一轮Empathy状态仅为弱历史信息 |

两版继续使用与V7/V18相同的response user Prompt、完整固定用户画像、原固定人设、V5 alignment和状态更新协议。最终回复不接收显式`relationship_distance`。

全量运行：

```bash
ASSET_SOURCE=data/exp2_qwen_plus_v5_clean \
BASELINE_DIR=data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation \
BEST_FULL_DIR=data/exp2_v16_full/v16_v7_selective_followup \
SWEEP_ROOT=data/exp2_prompt_sweep_v23_v24 \
CASE_SET=all \
GENERATE_WORKERS=3 \
bash scripts/run_exp2_prompt_sweep_v23_v24.sh
```

包装脚本默认把正确的联合门控V18加入汇总参考行。V23和V24必须使用各自的新目录，不复用V19–V22生成结果。

### 7.7 Sweep 的 asset 和缓存隔离

Sweep helper 只复用不可变文件：

```text
agent_persona.json
user_profile.json
asset_manifest.json
```

它会为每个版本重建干净的 runtime profile，并使用独立的 Milvus、states、generations 和 evaluation 目录。不会复制旧生成回复或旧 generated 标注。

默认 `REUSE_REFERENCE_CACHE=1` 只复用 `variant=reference` 的 Judge 标注。缓存同时绑定评估器指纹、候选文本哈希和完整上下文哈希；Prompt、模型、候选回复或上下文发生变化时不会误命中。

---

## 8. 定性分析：画像进化与画像熵

定性分析使用独立脚本：

```bash
uv run --no-sync python -m src.experiments.exp2_user_modeling_qualitative \
  --case Chat_1_Emi_Elise.json \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --output-dir data/exp2_qualitative
```

不传 `--case` 会处理全部对话。

该流程：

- 从空画像开始，不读取主实验 `prepare` 的一次性画像。
- 只回放前 90% Session 的真实双方消息。
- 不生成智能体回复，不读取后 10%，不运行 Table 2。
- 调用现有 `MemoryOSLocal` 和 `bayesian_online` 更新逻辑。
- 固定使用与一次性抽取相同的 5 层、21 个字段，不能新增、删除、改名或移动字段。
- 每个合并后的连续同 speaker dialogue bubble 记作一个 turn。
- 每个 turn 后记录 completeness、entropy、profile hash 和 profile version。

输出：

```text
data/exp2_qualitative/
├── cases/<case_id>/qualitative/
│   ├── profile_runtime.json
│   ├── user_profile.json
│   ├── memory/memory.db
│   ├── profile_snapshots/
│   ├── profile_trajectory.json
│   └── trajectory_manifest.json
└── qualitative_figures/
    ├── profile_curves.json
    ├── profile_evolution_curve.png
    └── profile_entropy_curve.png
```

- **画像进化**：21 个字段中已有稳定内容的比例。
- **画像熵**：21 个字段的平均二元熵；空字段按最大不确定性 1.0 计算。
- 单对话横轴使用真实 `turn_index`。
- 多对话聚合横轴使用归一化训练进度 `0%–100%`。
- 淡线为原始 per-turn 均值，主线为只使用当前和历史点的 causal EWMA。
- 单对话图会标记真实画像更新点和 Session 边界。

如果 trajectory 已完整生成，会直接复用；如果中途失败，为保证“从零开始”的协议，应换新的输出目录重跑。

---

## 9. 断点续跑与结果隔离

主实验和 sweep 均支持断点续跑：

- `predictions.jsonl` 通过 `example_id` 跳过已有连续前缀。
- `user_understanding.jsonl` 与 predictions 必须保持一致。
- `table2_annotations.jsonl` 保存 reference/generated Judge 缓存。
- 已完成的生成和 Judge 标注不会重复调用 API。
- 中断后使用完全相同的命令重新执行。

以下改动必须使用新输出目录：

- Prompt 版本或 Prompt 文本变化；
- 数据切分或 case 方向变化；
- 人设/画像 schema 或抽取 Prompt 变化；
- embedding 维度变化；
- 希望进行完全独立的新实验运行。

不要手工修改已有 prediction 的 `prompt_version` 或 `prompt_sha256`。

实验输出目录已被 `.gitignore` 忽略，不应提交到 GitHub。

---

## 10. 常见错误

### 10.1 `tokenizers ... found tokenizers==0.23.1`

普通 `uv run` 或 `uv sync` 重新同步了不兼容的传递依赖。修复：

```bash
uv pip install --reinstall \
  "transformers==4.57.6" \
  "tokenizers==0.22.2" \
  "huggingface-hub==0.36.2" \
  "bert-score==0.3.13"
uv pip check
```

之后评估统一使用 `uv run --no-sync`。

### 10.2 `torch.cuda.is_available() is False`

确认：

1. `nvidia-smi` 能识别 GPU；
2. 安装的是 `torch==2.7.1+cu118`，不是 CPU 或 CUDA 12 wheel；
3. Python 输出 `torch.version.cuda == 11.8`。

### 10.3 `training assets missing ... run prepare first`

当前输出目录没有该 case 的完整 assets。主实验应先在同一目录运行 `prepare`；Prompt sweep 应确认 `ASSET_SOURCE` 指向完整 V5 目录。

### 10.4 `existing predictions were generated with a different ... prompt version`

目标目录中已有其他 Prompt 版本的 predictions。不要修改 metadata；为新版本使用新目录，或使用对应 sweep 脚本自动创建隔离目录。

### 10.5 `no generated replies ... run --phase generate first`

`evaluate` 指向的目录没有 predictions。检查 `--output-dir`、`--case`、`--train-ratio` 和 `--prompt-version` 是否与 generate 完全一致。

### 10.6 Hugging Face 离线模式找不到模型

缓存不完整。第一次在联网状态去掉：

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

完成四个模型的下载后再恢复离线模式。

### 10.7 `gpt-4o-mini` 调用失败

检查 `[EvaluationAPI]`、`EVAL_API_KEY`、中转站余额和模型支持情况。临时换 Judge 可以验证流程，但不能视为严格复现 REALTALK。

### 10.8 Milvus 向量维度不匹配

当前 embedding 固定为 1536 维。不要复用由 1024 维或其他维度建立的旧 Milvus 数据库；使用新输出目录。

### 10.9 CUDA out of memory

依次降低：

```text
--eval-batch-size 16 → 8 → 4
```

Reflective、Grounding 和 Empathy 使用远程 Judge，不占本地 GPU；三个 CardiffNLP 模型和 BERTScore 使用 GPU。

远程 Judge 会按候选回复并发执行，默认最多同时处理 6 个候选。每个候选内部仍按相同 Prompt 分别得到 Reflective、Grounding 和 Empathy；结果返回后由主线程顺序写入 `table2_annotations.jsonl`，因此断点缓存不会被多个线程同时写入。本地 Sentiment、Emotion、Intimacy 和 BERTScore 不随 `--judge-workers` 并发，避免争抢单张 3090。若中转站返回频繁的 `429`，将其降为 `4` 或 `2`；不要通过同时启动多个完整 `evaluate` 进程来提高并发。

### 10.10 RoBERTa pooler warning

BERTScore 加载基础 `roberta-large` 时可能提示 pooler 权重未初始化。BERTScore 使用 token embeddings，不使用该 pooler；这是预期警告，不表示评估失败。

### 10.11 Milvus `too_many_pings`

本地 gRPC 偶尔会输出 `GOAWAY ... too_many_pings` 并自动调整 keepalive。只要后续生成和评估继续运行，它就是非致命提示。

---

## 11. 正式运行前检查

- [ ] 固定并记录 Git commit。
- [ ] `--train-ratio 0.9` 未改变。
- [ ] 画像和人设只读取前 90% Session。
- [ ] 测试历史使用 REALTALK 真实回复 teacher forcing。
- [ ] 目标回复和未来消息没有进入 generation input。
- [ ] 人设 schema 为 `lx_agent_v3_behavior_calibrated_no_catchphrases`。
- [ ] embedding 输出为 1536 维。
- [ ] 不同 Prompt 版本使用不同输出目录。
- [ ] `torch.cuda.is_available()` 为 `True`。
- [ ] `uv pip check` 通过。
- [ ] 评估使用 `uv run --no-sync`。
- [ ] 四个 Hugging Face 模型能在离线模式加载。
- [ ] 候选生成模型和 Judge 模型分别记录。
- [ ] `table2_main_results.md` 包含论文两行和 `Ours`。
- [ ] 正式结果覆盖论文声明的全部目标角色方向。

---

## 12. Windows PowerShell 附录

Linux/3090 是推荐环境。Windows 本地调试时，参数保持不变，仅续行符改为反引号：

```powershell
uv run --no-sync python -m src.experiments.exp2_user_modeling `
  --phase generate `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --config config.qwen-plus.ini `
  --prompt-version v5_relationship_calibrated `
  --output-dir data/exp2_single_v5 `
  --generate-workers 1

$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

uv run --no-sync python -m src.experiments.exp2_user_modeling `
  --phase evaluate `
  --case Chat_1_Emi_Elise.json `
  --train-ratio 0.9 `
  --config config.qwen-plus.ini `
  --prompt-version v5_relationship_calibrated `
  --output-dir data/exp2_single_v5 `
  --judge-config-section EvaluationAPI `
  --judge-model gpt-4o-mini `
  --judge-workers 6 `
  --eval-device cuda:0 `
  --eval-batch-size 16
```

批量运行时去掉 `--case`。夜间 Prompt sweep 脚本是 Bash 脚本，应在 Linux 3090 上运行。
