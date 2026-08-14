# REALTALK Ours V12 单动作自然表达锚点

V12 从冻结的 V9 完整 519 条结果重放 Response Actor。Self Domain、User Domain、Situation、动态 lambda、Behavior Policy、历史和模型均不重新生成。

核心约束是保留 V9 的单动作纪律，同时允许 V11 式自然措辞。Actor 只读取精简后的结构化动作；反思权限严格来自 `self_revelation_mode`，问题权限严格来自 `primary_move` 和 `continuation_move`。生成阶段不读取完整 User Domain、lambda、评价指标或 Ground Truth，不启用 thinking、Verification、多候选或重写。

开发集为 30 条：排除 V11 120 条后，每位人物每个 Session 选择一个确定性中点。独立测试集为 80 条：进一步排除开发集，每位人物选择 8 条，按剩余 Session 规模使用最大余数法分配并在各单元内 full-span 取样。

开发 30 条通过后冻结 Prompt。独立 80 条使用论文 Appendix C Judge 和相同本地指标；只有八项原始值全部严格超过论文逐列最优，才允许运行完整 519 条。
