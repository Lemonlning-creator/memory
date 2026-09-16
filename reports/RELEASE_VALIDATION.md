# V9 发布整理验证

日期：2026-09-16。所有验证只做离线计算、文件复制与 Git 操作；未调用生成模型、未调用 Judge、未启动训练。

## 已通过

- 28 个原始 Python 文件与冻结提交 `5927bbf` 逐字节一致，包括原 V9 Prompt、Schema 与支持模块。
- Prompt/Schema 哈希与服务器原 `run_manifest.json` 一致。
- 10 个官方数据文件 SHA256 一致，重建得到 519 条与 10 人固定计数；真实历史/答案哈希逐项匹配冻结预测。
- 原正式 predictions SHA256：`ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81`。
- 新入口固定 DeepSeek Flash、关闭 thinking；测试防止误用 Qwen 默认值、覆盖非空目录或复用不同预测的评价缓存。
- 从逐样本重新汇总八项分数，六位小数与原始汇总一致。
- 完整包：519 条生成、519 条本地指标、519 条 GPT 评分记录、3,114 个固定判断、10 份 Self Domain，零 unresolved。
- 结果包 48 个文件在服务器封装后校验，再下载本地复验，哈希全部一致。
- 本地 Windows 单测：42 passed、1 skipped；跳过项为原版 POSIX alarm 超时测试。
- 凭据模式扫描：没有发现 API key、GitHub token 或私钥块；真实 .env、缓存和对话数据不进入 Git。

## 结果包

```text
realtalk-v9-frozen-519-v1.tar.gz
SHA256: 63ce79bb33b22bd8e83310cb867deae43fd2fe4a4c8d3793d5043bbbf4073d51
```

本地位置和服务器位置见 README。代码发布测试不会改动结果包、V9 原目录或师姐仓库已有分支。

## 未声称通过

- 没有重新支付 API 成本跑一遍 519 条或重新 Judge。
- 没有在本轮重新下载所有评价权重并执行五项模型推理；验证使用已存在的完整指标记录与原始 mock 单测。
- 并非作者运行环境逐字节复刻；BERTScore 权重未锁定官方未披露的 revision，完整依赖环境也不是历史机器镜像。
- 独立公开仓库分支仅封装冻结版本，不代表八项指标全优于论文。
