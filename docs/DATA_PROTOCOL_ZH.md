# 数据、划分与因果边界

数据来自 REALTALK 官方预处理的 10 个 `data/Chat_*.json`，不是 PersonaEmp、AlpsBench 或 WildChat。固定源提交和逐文件 SHA256 见 `provenance/dataset_manifest.json`。

本版是在公开数据上按论文描述实现 Persona Simulation 协议。官方仓库该任务入口在所固定版本中仍标为 Coming soon；**这不是把作者完整官方 Persona Simulation 脚本直接运行一遍**。数据来源、论文规则、我们的实现选择需区分。

## 十个目标人物

Ca 用于建立目标人物 Self；Cb 用于测试。两者均取按 session 数字排序的前三个 session。表中的文件编号对应 `Chat_<编号>_*.json`，完整文件名以 manifest 为准。

| 目标人物 | Ca 文件 | Cb 文件 | Cb 目标消息数 |
|---|---|---|---:|
| Emi | Chat_4_Emi_Paola.json | Chat_1_Emi_Elise.json | 37 |
| Nicolas | Chat_5_Nicolas_Nebraas.json | Chat_6_Vanessa_Nicolas.json | 117 |
| Kevin | Chat_3_Kevin_Paola.json | Chat_2_Kevin_Elise.json | 25 |
| Akib | Chat_9_Fahim_Akib.json | Chat_8_Akib_Muhhamed.json | 37 |
| Muhhamed | Chat_10_Fahim_Muhhamed.json | Chat_8_Akib_Muhhamed.json | 37 |
| Nebraas | Chat_5_Nicolas_Nebraas.json | Chat_7_Nebraas_Vanessa.json | 51 |
| Paola | Chat_4_Emi_Paola.json | Chat_3_Kevin_Paola.json | 23 |
| Vanessa | Chat_7_Nebraas_Vanessa.json | Chat_6_Vanessa_Nicolas.json | 116 |
| elise | Chat_2_Kevin_Elise.json | Chat_1_Emi_Elise.json | 36 |
| Fahim Khan | Chat_10_Fahim_Muhhamed.json | Chat_9_Fahim_Akib.json | 40 |
| 合计 | | | **519** |

这是逐人物划分，不是所有文件都双向跑一遍；也不根据当前运行的 EI 分数重新选择 Ca/Cb。一个文件可以被不同目标人物以不同身份使用，但画像、状态与缓存按目标人物隔离。

## 519 条如何产生

1. 读取官方预处理字段 `clean_text`，去掉首尾空白，忽略空消息。
2. 按 `session_1, session_2, session_3` 等数字顺序读取，而非另按时间戳重排。
3. **同一个 session 内**连续同说话者气泡用换行合并，保存所有原始 message index 和 dia_id；不跨 session 合并。
4. Ca 前三个 session 的双方合并消息用于 Self 编译。
5. Cb 前三个 session 中，每个目标人物合并消息建立一个测试点。
6. 测试输入是该点之前的全部真实 Cb 合并消息；隐藏答案是当前目标合并消息。

因此“519”是本实现和冻结文件重建得到的样本数，不是论文给出的官方精确计数。1076 个原始气泡不是本版评分单位。合并对非空 clean_text 使用换行保留；“完整历史”指该预处理后的范围，不能宣称 XLSX 原始字节完全未加工。

## Session 与历史继承

Session 2 能看到真实 Session 1；Session 3 能看到真实前两个 session。预测出来的话不会回灌。模型不能看到当前或未来真实答案；上一条真实目标人物消息可以出现在后续预测的历史中。

Self 固定，User 仅在完成 Session 1 和 Session 2 后更新。既不是随机 train/test，也没有梯度训练。代码变量 `train_chat` 的实际用途是 Ca 建模资料。

生成历史是跨 session 累积；冻结 Judge 的上下文则仅为当前 session 的真实前缀。二者不同，是明确保留的实现选择，不应在整理时统一改掉。

## 校验与文件获取

`python -m release_tools.data --download` 只获取指定 commit 下的十个官方 JSON，每个文件先检验 SHA256 再写入。不会下载新的上游语料或重新调用 LLM 生成数据。

离线使用 `--source-repo /path/to/REALTALK` 从 Git 对象读取固定提交；避免 Windows CRLF checkout 改变哈希。`--source /path/to/json-directory` 要求字节已经一致，遇到差异拒绝覆盖。

准备后生成本地 `dataset/prepared_manifest.json` 和 `dataset/sample_manifest.json`；后者记录每条 result_id、人物、session、历史哈希与真值哈希。公开 Git 分支不重新发布原始聊天；完整数据副本随独立结果包保存在受控本地/服务器位置。
