# REALTALK persona simulation — Ours (V9 + Rule A+B) r2

一个冻结版本：**在 V9 的 Actor 阶段追加 Rule A+B 约束**，其余（自域、用户域、决策、数据、判分口径）全部保持与 V9 一致。
全量 519 条（10 位说话者），完整 8 项指标，与论文 Table 2 并列。

## 结果一览

| Method | Lexical ↑ | Semantic ↑ | Reflective ↑ | Grounding ↑ | Sentiment ↑ | Emotion ↑ | Intimacy ↓ | Empathy ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o fine-tune | 0.14 ± 0.04 | 0.76 ± 0.08 | 0.62 ± 0.13 | 0.40 ± 0.13 | 0.53 ± 0.22 | 0.43 ± 0.22 | 0.06 ± 0.01 | 1.80 ± 0.55 |
| w/ fine-tune | 0.14 ± 0.05 | 0.78 ± 0.04 | 0.77 ± 0.09 | 0.62 ± 0.08 | 0.59 ± 0.18 | 0.46 ± 0.21 | 0.07 ± 0.01 | 1.24 ± 0.12 |
| **Ours (V9 + Rule A+B) r2** | **0.14 ± 0.04** | **0.86 ± 0.01** | **0.77 ± 0.07** | **0.62 ± 0.10** | **0.65 ± 0.18** | **0.52 ± 0.22** | 0.07 ± 0.01 | **1.08 ± 0.15** |

**7/8 项达到或超过论文逐列最优**；未达标：Intimacy。

> `±` 为 **10 位说话者间的样本标准差 (n−1)**，与论文 Table 2 的口径相同：
> 论文自身从未定义 `±`；本口径由论文 Appendix E.2 的逐说话者 Table 8 反推确认——
> 用 Table 8 的 10 个逐人值计算样本标准差，16 个 `±` 全部精确复现。

## 本实验到底做了什么

V9 的生成链路是：`Self Domain（冻结）→ User Domain（冻结）→ Decision（冻结）→ Actor（本轮唯一改动）`。
本版本**只重跑 Actor**，输入的自域、用户域、每条的 Decision 结果全部复用 V9 冻结产物。

Actor 收到的额外约束（仅在满足触发条件时追加）见 `method/rule_clause.txt`：

- **触发条件**：`partner_asked == False`（伙伴上一句不含问号）。519 条中 348 条命中。
- **Rule A 部分**：禁止第一人称描述自身感受 / 偏好 / 意图 / 动机 / 自我观察。
- **Rule B 部分**：不新增信息、不换话题、不给建议、不做自我延伸，也不确认或复述伙伴已说的内容；只给一个简短反应。

## 配置（固定）

| 项 | 值 |
|---|---|
| 生成模型 | `deepseek-v4-flash`（thinking 关闭） |
| 解码 | temperature 0.6 / top_p 0.9 / max_tokens 300 / 无 seed |
| 数据 | REALTALK Table 8 的 10 组 Ca/Cb，各前 3 个 session；519 条目标消息 |
| 判分 prompt | 论文 Appendix C 对齐版（`paper-align`） |
| 判分模型 | `gpt-4o-mini-2024-07-18`（固定快照） |
| 聚合 | 先按说话者平均，再对 10 人取宏平均；2 位小数 |

## 目录

```
method/
  rule_clause.txt      规则原文（Rule A 与 Rule A+B 全文）
  actor_replay.py      生成实现（含 RULE_A_CLAUSE / RULE_AB_CLAUSE 与触发条件）
  judge_arm.py         判分实现（paper-align prompt + 固定快照）
  run.sh               本次运行的完整启动脚本
results/
  TABLE2.md            8 项对比表
  per_speaker.json     宏平均、样本标准差(n-1)、逐说话者结果
  predictions.jsonl    519 条生成结果（唯一正式预测文件）
  judge_checkpoint.json  1,557 个 judge 判定
  judge_summary.json     判分汇总
  local_annotations.json Cardiff 分类器标注（Sentiment/Emotion/Intimacy）
  bertscore_f1.json      BERTScore F1
provenance/
  MANIFEST.json        全部文件 SHA256 + 配置快照 + 运行时间
```

## 复现方式

```bash
# 生成（需要 REALTALK_OURS_API_KEY / REALTALK_OURS_BASE_URL）
python method/actor_replay.py --frozen <V9冻结包> --output-dir <out> \
    --arm ruleAB --ids-file <519个result_id的json> --workers 12

# 判分（需要 REALTALK_JUDGE_API_KEY / REALTALK_JUDGE_BASE_URL）
python method/judge_arm.py --gen <out>/generated.jsonl --out <judge-out> \
    --label ruleAB --workers 16 --model gpt-4o-mini-2024-07-18
```

## 必须一起交接的限制

1. **Intimacy 未达标**（0.0747 对论文逐列最优 0.06）。该指标由固定 Cardiff 分类器给出，与 judge 配置无关，
   论文自身两行也只差 0.01（等于其标准差）。
2. **判分模型未与论文对齐到同一快照**：论文未披露 judge 版本；本版使用 `gpt-4o-mini-2024-07-18` 作为可复现的近似。
3. **生成模型与论文不一致**：Ours 用 `deepseek-v4-flash`，论文未披露其基础模型。
4. 无固定 seed，同代码重跑不保证得到相同文本。
