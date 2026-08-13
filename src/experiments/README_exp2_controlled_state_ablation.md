# Exp2 受控状态消融测试

这个测试用于回答一个单独的问题：V18 最终回复表现的变化，究竟来自回复 Prompt，还是来自每次重跑时发生变化的用户状态轨迹。

它不会修改主实验，也不会重新运行 alignment 或 Milvus。脚本读取一份已经完整跑完的 V18 目录，逐条冻结以下输入：

- 当前 REALTALK 用户消息；
- teacher-forcing 历史与 `relevant_memory`；
- 训练集用户画像与智能体人设；
- 来源 V18 的短期 `current_state` 轨迹；
- V18 的 response system/user Prompt。

三个条件唯一的差别是传给最终回复 Prompt 的上一轮状态：

| 条件 | `previous_empathy_state` 实际内容 |
|---|---|
| `full_state` | 来源 V18 保存的完整对象，包括数值、tone 和 response guidance |
| `scores_only` | 只保留 `emotional_reaction`、`interpretation`、`exploration` 三个数值 |
| `no_state` | 空对象 `{}` |

第一条回复的 `current_state` 按主实验初始化协议设为空对象；后续回复使用来源 V18 上一个评测点保存的 `core_current_state`。三个条件不会各自更新状态，因此不存在新的随机状态分叉。

## 一键运行全部 117 条

以下命令默认使用温度 0，先完成三个条件的生成，再执行原 Table 2 评估并生成并列表：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -u -m src.experiments.exp2_controlled_state_ablation \
  --phase all \
  --source-dir data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate \
  --output-dir data/exp2_controlled_state_ablation_v18 \
  --train-ratio 0.9 \
  --config config.qwen-plus.ini \
  --source-prompt-version v18_reflective_grounding_joint_gate \
  --conditions full_state,scores_only,no_state \
  --temperature 0 \
  --generate-workers 3 \
  --judge-config-section EvaluationAPI \
  --judge-model gpt-4o-mini \
  --judge-workers 6 \
  --eval-device cuda:0 \
  --eval-batch-size 16
```

脚本支持断点续跑。已有 prediction 的 `example_id` 会跳过；但只要来源文件哈希、Prompt、模型、温度、条件或 case 集合发生变化，就会拒绝混用旧目录，要求换一个新的 `--output-dir`。

## 先跑单个对话验证

```bash
uv run --no-sync python -u -m src.experiments.exp2_controlled_state_ablation \
  --phase generate \
  --source-dir data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate \
  --output-dir data/exp2_controlled_state_ablation_v18_chat1 \
  --config config.qwen-plus.ini \
  --case Chat_1_Emi_Elise.json \
  --temperature 0 \
  --generate-workers 1
```

确认生成后再评估同一个目录：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -u -m src.experiments.exp2_controlled_state_ablation \
  --phase evaluate \
  --source-dir data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate \
  --output-dir data/exp2_controlled_state_ablation_v18_chat1 \
  --config config.qwen-plus.ini \
  --case Chat_1_Emi_Elise.json \
  --judge-config-section EvaluationAPI \
  --judge-model gpt-4o-mini \
  --judge-workers 6 \
  --eval-device cuda:0 \
  --eval-batch-size 16
```

## 输出位置

```text
data/exp2_controlled_state_ablation_v18/
├── full_state/
├── scores_only/
├── no_state/
├── controlled_state_ablation_summary.json
└── controlled_state_ablation_summary.md
```

每个条件都有独立的 prediction、annotation、Table 2 分数和条件 manifest。总报告除官方聚合结果外，还会按相同 `example_id` 给出 `scores_only/no_state` 相对 `full_state` 的 wins、losses、ties；对 Intimacy 和 Empathy 已先反转方向，因此报告中的正向 delta 始终表示改善。

注意：历史 V18 行只用于提供已有结果背景。真正的随机对照是本次同时生成的 `full_state`，因为它与另外两组共享模型、温度和冻结输入。
