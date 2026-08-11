# REALTALK Task 1 Ours Qwen3-8B 实施锚点

状态：已实现并完成真实 Qwen3-8B 两样本预检；等待运行环境注入 API 密钥后执行全量  
方法锚点：`OURS_METHOD_ANCHOR_ZH.md`  
协议审计：`EXP2_REALTALK_INPUT_OUTPUT_AUDIT_ZH.md`  
最终协议：`EXP2_REALTALK_OURS_FINAL_PROTOCOL_ZH.md`

## 1. 唯一目标

在 REALTALK Task 1 Persona Simulation 上只运行 Ours，不训练、不微调、不运行论文基线、Task 2、Memory Probing 或未来用户理解任务。论文 Table 2 的两行直接作为已发表参考结果，待 Ours 八项指标完整后增加一行。

全部 Ours 调用固定为：

```text
model: qwen3-8b
thinking: disabled
```

论文没有公开 Persona Simulation 基础模型，因此结果只能称为 `protocol-aligned comparison`，不能称为相同运行时的严格复现。

## 2. 数据与测试点

- 固定官方 REALTALK 仓库 commit：`b903e06a9770bf4e5fe9018c3e132889666d3b4a`。
- 严格使用论文 Table 8 的十组 speaker-specific Ca/Cb。
- 同一 Session 内连续同说话者气泡合并，不跨 Session 合并。
- Ca 前 3 个 Session 用于一次性建立目标人物 Self Domain。
- Cb 前 3 个 Session 中，每条目标人物合并消息形成一个预测点。
- 每个预测点只读取该目标之前的真实历史；模型输出不回灌。
- 公开数据重建数量为 10 人、519 条；论文没有公布其运行时准确样本数。

逐人目标数固定为：

```text
Emi 37, Nicolas 117, Kevin 25, Akib 37, Muhhamed 37,
Nebraas 51, Paola 23, Vanessa 116, elise 36, Fahim Khan 40
```

## 3. Ours 调用顺序

### 3.1 Self Domain

每位目标人物在进入 Cb 前调用一次。输入是其 Ca 前 3 个 Session 的双方真实对话，但只有目标人物自己的发言能支持目标人物属性。输出包括身份信息、人格和表达风格、行为策略先验、硬约束与不确定项。进入 Cb 后固定，不读取测试答案更新。

### 3.2 每条目标消息

1. `User Domain Update`：只使用当前目标之前新观察到的 Cb 伙伴发言；无新伙伴证据时复用，不调用模型。
2. `State + Alignment + Policy`：一次输出 Current/Future User State、显式动态 `lambda_t` 和唯一 Behavior Policy。
3. `Generation`：使用真实历史和上述状态生成目标人物下一条消息。

`lambda_t` 越高表示越适配当前伙伴，但 Self Domain 始终是身份硬约束。它不能把目标人物变成通用高共情助手。禁用 Omega、多候选策略、Verification 和语义重写。

最终系统约束固定为：

```text
You are {speaker}. Continue the conversation.
Output only the message, not the speaker name.
```

不强制共情、建议、追问、积极语气或固定句数。

## 4. Schema、重试与因果审计

- Self Domain、User Domain、Alignment 使用严格 JSON Schema；Generation 为纯文本。
- DashScope Qwen 使用 required tool schema。
- 每个逻辑结构化操作最多 3 次：首次生成和最多两次格式修复。
- 修复 Prompt 携带验证错误和原输出，明确禁止增加证据、事实、状态解释或改变策略。
- User Domain 的每条事实必须引用已经观察到的伙伴 turn ID。
- User State 不得引用未来、目标人物或未观察到的 turn ID。
- 3 次失败后写入 `unresolved_errors.json`；正式生成完成要求零 unresolved。
- 检查点按 Prompt、Schema、数据、模型和配置哈希绑定，签名不一致拒绝复用。
- API 密钥只从环境变量读取，不进入源码、日志、manifest 或报告。
- Self Domain 的身份与兴趣事实仅作为稳定背景；若当前 Cb 历史没有直接支持，不得被写成当前活动、计划或轶事。
- 单个问候或礼貌问句不得被扩写为 User Domain 的 Core、Identity 或稳定 Cognition；证据稀疏时允许五层为空。
- 无伙伴证据的首条消息采用确定性冷启动语义：`lambda_t=0`、高不确定性、只允许简短问候或通用 check-in。

固定解码：

| Stage | Temperature | Top-p | Max tokens |
|---|---:|---:|---:|
| Self Domain | 0.2 | 0.9 | 1800 |
| User Domain | 0.2 | 0.9 | 1800 |
| Alignment | 0.2 | 0.9 | 1600 |
| Generation | 0.6 | 0.9 | 300 |

## 5. 输出

主要产物：

- `dataset_manifest.json`：Table 8 映射、源文件哈希和逐人数目。
- `self_domains.json`：十位目标人物固定 Self Domain。
- `predictions.jsonl`：519 条预测、真值、User Domain、State、lambda 和 Policy。
- `raw_responses.jsonl`：每次逻辑尝试的原始输出与用量，不含密钥。
- `checkpoint.json`：逐阶段原子检查点和失败记录。
- `run_manifest.json`：模型、解码、Prompt/Schema/数据哈希与调用成本。
- `REPORT_PARTIAL.md`、`table2_partial.json`：论文两行加 Ours 当前可用指标。

完成标志：

- `GENERATION_COMPLETE`：目标记录齐全且零 unresolved。
- `LOCAL_METRICS_COMPLETE`：本地五项指标完成。
- `GPT_EVALUATION_PENDING.json`：三项官方 GPT 指标仍待评。
- 在真实 `gpt-4o-mini` 三项评价完成前，禁止生成 `PIPELINE_COMPLETE`。

## 6. 评价

当前先计算：

1. ROUGE-L 重建值。
2. BERTScore F1，固定标准 English `roberta-large` 配置。
3. Sentiment Accuracy，固定 CardiffNLP 模型与 revision。
4. Emotion Accuracy，固定 CardiffNLP 模型与 revision。
5. Intimacy Absolute Difference，固定 CardiffNLP 模型与 revision。

获得真实 `gpt-4o-mini` 端点后，沿用 REALTALK 官方 Appendix C Prompt 补充：Reflectiveness Accuracy、Grounding Accuracy、Empathy Absolute Difference。

Table 2 的 Ours 行按每位说话者先求均值，再对十位说话者报告 macro mean 与 population standard deviation；message-micro 仅作诊断。

## 7. 运行方式

只在进程环境配置秘密：

```bash
export REALTALK_OURS_API_KEY='...'
export REALTALK_OURS_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export REALTALK_DATASET_DIR='dataset'
export REALTALK_OURS_OUTPUT_DIR='data/realtalk_ours_qwen3_8b'
```

模型预检：

```bash
python -m src.experiments.realtalk_ours --preflight-only
```

单人两条结构预检：

```bash
python -m src.experiments.realtalk_ours \
  --speaker Emi \
  --max-eval-points-per-speaker 2 \
  --skip-local-metrics \
  --output-dir data/realtalk_ours_qwen3_8b_pilot
```

完整运行：

```bash
bash tools/run_realtalk_ours_qwen3_8b.sh
```

运行脚本固定 `CUDA_VISIBLE_DEVICES` 为空，只运行本协议入口。
