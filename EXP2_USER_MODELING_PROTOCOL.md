# 实验二：用户建模评测协议

实验一和实验二位于同一个 `memory` 仓库，共用数据结构、画像实现、模型客户端、
断点和评价基础设施。实验二在独立开发分支/worktree 中修改，目的是防止未验证代码
破坏已经收束的实验一；稳定后再合回共同主线，并非建立第二个仓库。

后续生成与结构化判断统一使用百炼中国区
`qwen3-30b-a3b-instruct-2507`。DeepSeek 暂不启用，历史 Kimi Pilot 仅作为旧记录。

## 1. 数据与因果边界

- 使用 REALTALK Table 8 的说话人级跨对话划分。
- `Ca` 前 3 个 session 只用于建立固定五层画像。
- `Cb` 前 3 个 session 按“合并后的同说话人消息”逐点测试。
- 当前理解轨道可以看到当前真实用户消息。
- 未来理解轨道只能看到目标消息之前的真实历史，不能看到目标消息。
- 历史默认保留所选 `Cb` 片段中目标之前的全部 turn；超过
  `--max-context-chars` 时从最旧完整 turn 开始删除。

## 2. 两个轨道

### 当前用户理解

这是论文实验设计新增的扩展任务，不是 REALTALK Table 2 原任务。

- `realtalk_zero_shot`：历史 + 当前消息。
- `ours`：相同历史 + 当前消息 + 由 `Ca` 独立生成的五层画像。
- 主指标：Emotion Accuracy、Sentiment Accuracy。
- Topic Consistency 保留为扩展指标，因为 REALTALK 没有 topic 标准答案。

### 未来用户理解

严格采用 REALTALK persona simulation 的消息级任务：根据目标前历史生成该说话人的
下一条消息，再将生成消息与真实消息都交给同一套固定 EI 分类器。

新版设计稿写作“下一 Session”，但 REALTALK Table 2 的可复现单位是下一条合并后的
说话人消息。正式实现采用论文单位，否则不能与 Table 2 构成同一实验协议。

- `realtalk_zero_shot`：使用论文原始任务句
  `You are {speaker}. Continue the conversation.`
- `ours`：相同任务与历史，额外加入由 `Ca` 生成的五层画像。
- 主指标：Emotion Accuracy、Intimacy Absolute Difference。
- 原始生成消息、真实消息、Sentiment 和词面相似度一并保存，便于后续补算
  REALTALK Table 2 的其余指标。

REALTALK 的 `w/ fine-tune` 需要逐说话人训练模型。当前 API 模型不能等价复现，
因此 runner 只接受外部逐样本预测文件；没有提供时明确标记为 unavailable，不用
提示词伪造微调结果。论文 Table 2 数值仅作为独立参考表保存，不与本次结果拼接。

## 3. 输出

- `results.jsonl`：每个测试点的原始消息、标签、两轨道输出和逐项分数。
- `summary.json`：speaker-macro 主结果、micro 诊断与官方参考表。
- `run_manifest.json`：模型、分类器 revision、Schema、prompt hash、切分和配置。
- `checkpoint.json`：按输入哈希缓存的断点数据；失败不会记为 0 分。

Persona Consistency 使用 REALTALK Table 8 已发布的跨对话诊断值，并与方法主排名分开。
每位说话人的固定五层画像熵也会保存，为后续画像进化曲线提供原始数据；本阶段不启用
Bayesian Updating。

运行示例：

```powershell
uv run python -m src.experiments.user_modeling.runner `
  --dataset-dir dataset `
  --speaker Emi `
  --max-eval-points-per-speaker 2 `
  --output-dir data/exp2_user_modeling_smoke
```
