# Persona Simulation Benchmark (实验2a)

## 概述

基于 RealTalk 数据集的 Persona Simulation 任务，验证情感共情机制的有效性。

## 实验设计

### 评估任务
在对话的 session 边界处，给定前 N 个 session 的对话历史，让系统生成 agent speaker 的下一条回复，并与 ground truth 对比。

### 方法变体

1. **baseline_llm**: 零样本 LLM baseline，仅使用对话历史
2. **profile_only**: 使用用户画像 + persona，无记忆检索
3. **full_agent**: 完整系统（画像 + 记忆 + persona）

### 评估指标

复用 RealTalk 论文的 Style Similarity 指标：
- **Reflectiveness** (0-2): 自我反思程度
- **Grounding** (0-2): 澄清/追问行为
- **Sentiment Score** (0-2): 情感极性匹配
- **Emotion Score** (0-2): 情绪类别匹配
- **Intimacy Score** (0-2): 亲密度匹配
- **Empathy Score** (0-2): 共情程度匹配

## 运行方式

```bash
# 基础运行
cd /Users/ln123/Document/pythonfile/memory
.venv/bin/python3 -m src.experiments.persona_simulation

# 自定义参数
.venv/bin/python3 -m src.experiments.persona_simulation \
  --dataset-dir dataset \
  --output-dir data/persona_simulation_eval \
  --min-context-sessions 2 \
  --max-eval-points 10 \
  --methods baseline_llm profile_only full_agent
```

## 前置条件

1. RealTalk 数据文件在 `dataset/` 目录下（格式：`Chat_*_*.json`）
2. 用户画像文件在 `user/` 目录下（格式：`{user}_{agent}_profile.json`）
3. Agent persona 文件在 `agent/` 目录下（格式：`{agent}_persona.json`）

如果画像或 persona 不存在，需要先运行：
```bash
# 生成用户画像
.venv/bin/python3 -m src.experiments.user_profile_generation --realtalk dataset/Chat_1_Emi_Elise.json

# 生成 agent persona
.venv/bin/python3 -m src.experiments.agent_persona_generation --realtalk dataset/Chat_1_Emi_Elise.json
```

## 输出

- `data/persona_simulation_eval/persona_simulation_results.json`: 详细结果（每条评估的生成文本和 EI 分数）
- `data/persona_simulation_eval/persona_simulation_summary.json`: 汇总结果（各方法的平均分数）

## 扩展方向

### 待实现的变体

- **w/o Empathy Reasoning**: 需要实现 unified reasoning prompts（当前 `agent_ablate_dimension.py` 中引用的 `UNIFIED_REASONING_*` prompts 尚未定义）
- **w/o Memory**: 在 FullAgent 中禁用记忆检索
- **Fine-tuned Baseline**: 需要在目标 speaker 的历史数据上微调 LLM

### 实现 unified reasoning

1. 在 `src/prompts/templates.py` 中添加 `UNIFIED_REASONING_SYSTEM_PROMPT` 和 `UNIFIED_REASONING_USER_PROMPT_TEMPLATE`
2. 参考 `agent_ablate_dimension.py` 中的推理流程
3. 在 `persona_simulation.py` 中添加 `EmpathyReasoningAgent` 变体

## 参考

RealTalk 论文: "REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation"
- Persona Simulation 任务定义: Section 6.1
- EI 评估框架: Section 4
- 基线结果: Table 2, Table 8
