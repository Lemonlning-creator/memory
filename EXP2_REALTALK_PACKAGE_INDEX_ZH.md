# Exp2 REALTALK 封装索引

## 当前结论

Exp2 已完成一组可复核的 REALTALK Persona Simulation 全量实验：Ours V9 在固定
Table 8 Ca/Cb 映射、每段对话前三个连续 Session、10 位目标人物和 519 条重建测试
消息上完成生成与八项评价。V9 超过论文逐列最优值中的 5 项，尚未超过
Reflectiveness、Grounding 和 Intimacy AD。

V13 是后续渐进优化实验，目前只在固定 6 条困难集完成 Gate 1。它用于诊断，不得替代
V9 全量结果，也不得作为论文主表结果。

## 阅读顺序

1. `EXP2_REALTALK_FINAL_REPORT_ZH.md`：完整实验结论、论文对比与限制。
2. `EXP2_REALTALK_HANDOFF_ZH.md`：代码入口、服务器产物、验证命令和续跑边界。
3. `EXP2_REALTALK_ARTIFACT_MANIFEST.json`：机器可读的路径、哈希、数量和状态。
4. `EXP2_REALTALK_OURS_FINAL_PROTOCOL_ZH.md`：数据划分和方法协议锚点。
5. `EXP2_REALTALK_V9_FULL519_RESULTS_ZH.md`：V9 原始全量结果摘要。
6. `EXP2_REALTALK_OURS_V13_GATE1_REPORT_ZH.md`：V13 Gate 1 诊断结论。

## 权威层级

| 层级 | 文档或产物 | 用途 |
|---|---|---|
| 主结果 | V9 519 条生成、完整本地指标、完整 GPT Judge | Exp2 当前正式结果 |
| 优化诊断 | V13.4 / V13.6 Gate 1 | 分析后续优化方向 |
| 协议锚点 | Final Protocol、3-Session Execution Anchor | 解释数据与因果边界 |
| 历史诊断 | V2-V12 计划、预检和局部结果 | 保留研究过程，不进入主结论 |

## 不应混用

- `519` 是按公开 REALTALK 数据、Table 8 和论文规则重建的样本数；论文未公布该准确数字。
- V9 与论文属于 protocol-aligned comparison，不是相同基础模型运行时的严格复现。
- V13 的 6 条是有意选择的困难样本，不可与论文全量 Table 2 数字直接横比。
- BERTScore 是论文 Table 2 指标；当前实现参数是透明重建，不是论文未公开的官方代码。
- 任何旧版小样本“八项全部领先”结论均被 V9 全量结果取代。

## Git 分支

- Fork：`Nobody-ly/memory` 的 `paper-boost/exp2-user-modeling-evaluation`。
- 师姐仓库：`Lemonlning-creator/memory` 的同名 Exp2 分支。
- 两个分支应指向同一提交；交接时以远端分支 tip 为准。
