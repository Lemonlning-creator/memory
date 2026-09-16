# Deep Empathy · REALTALK Ours V9

本分支只封装已完成 **519 条、10 人、八项指标** 的冻结 V9。它是当前保留的完整基线，不含 V10 以后优化，也不运行实验一、实验三或其他 baseline。

**实际模型是 `deepseek-v4-flash`，不是 Qwen。所有 Ours 阶段关闭 thinking。本版没有训练或微调。**

| 固定项 | 内容 |
|---|---|
| 原始实现提交 | `5927bbff03fda74eebaeb99e0c57203a644cfd74` |
| 官方数据源 | [danny911kr/REALTALK](https://github.com/danny911kr/REALTALK)，提交 `b903e06a9770bf4e5fe9018c3e132889666d3b4a` |
| 方法来源 | [师姐的 memory 仓库](https://github.com/Lemonlning-creator/memory)；本分支为实验二 V9 独立归档 |
| 任务 | 扮演指定真实人物，预测下一条合并消息，不是通用助手回复 |
| Ca / Cb | Table 8 逐人物指定；各取前 3 个 session，无随机重划分 |
| 评价 | 5 项本地指标 + `gpt-4o-mini` 的 3 项 Judge 指标 |
| 结果定位 | protocol-aligned exploratory comparison，不宣称模型运行条件与论文完全相同 |

## 先看哪里

- [方法、Self/User Domain、λ 和完整调用链](docs/METHOD_ZH.md)
- [数据来源、十人 Ca/Cb、519 条如何构造](docs/DATA_PROTOCOL_ZH.md)
- [安装、预检、完整测试、续跑和评价命令](docs/RUNBOOK_ZH.md)
- [训练与微调：本版实际做了什么](docs/TRAINING_STATUS_ZH.md)
- [冻结结果、论文对比和已知边界](docs/RESULTS_ZH.md)
- [交接、文件职责和复核清单](docs/HANDOFF_ZH.md)
- [完整 Prompt 和 JSON Schema](contracts/index.json)，由原代码导出，不是重新编写。

## 快速开始

Python 3.11+；建议 Linux 执行正式实验。以下前四步不调用付费模型：

```bash
git clone --single-branch --branch release/exp2-realtalk-v9-clean https://github.com/Lemonlning-creator/memory.git memory-v9
cd memory-v9
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m release_tools.data --download
python -m release_tools.cli verify-code
python -m pytest -q
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。已有官方 Git 仓库可离线准备：

```bash
python -m release_tools.data --source-repo /path/to/REALTALK
```

在本地将 `.env.example` 复制为 `.env`，填入自己的端点与密钥。该文件已忽略。下面会产生 API 费用：

```bash
# 仅模型端点预检；单独目录，不作为正式生成目录。
python -m release_tools.cli generate --scope preflight --output results/preflight --new
# Akib 每个 session 中点一条，共 3 条；只用于流程检查。
python -m release_tools.cli generate --scope smoke --speaker Akib --output results/smoke-akib --new
# 完整 10 人、519 条。正式测试不传任何抽样参数。
python -m release_tools.cli generate --scope full --output results/v9-full --new
```

后续本地五项、Judge 三项、最终报告命令见 [RUNBOOK](docs/RUNBOOK_ZH.md)。本轮整理没有重新生成或重新 Judge 已有 V9。

## 干净结果放在哪里

整理后的代码：本地 `D:\codex_workspace\memory-exp2-v9-release\`；服务器 `/amax/xidian_ty/Ly/personaemp-exp2/releases/exp2-realtalk-v9-code/`。

完整结果包独立于代码，原始实验目录保持不动：

- 本地：`D:\codex_workspace\deliveries\realtalk-v9-frozen-519-v1\`
- 服务器：`/amax/xidian_ty/Ly/personaemp-exp2/releases/realtalk-v9-frozen-519-v1/`
- 服务器同目录有 `realtalk-v9-frozen-519-v1.tar.gz`，便于转交。
- GitHub 只保存 [冻结汇总](reports/frozen/)、[来源证明](provenance/) 和 [结果说明](docs/RESULTS_ZH.md)。完整对话、画像、预测、Judge 检查点不复制进公开 Git 历史。

包内 `README.md` 解释所有产物，`PACKAGE_MANIFEST.json` 校验全部文件。运行：

```bash
python -m release_tools.package_frozen --output /path/to/realtalk-v9-frozen-519-v1 --verify-only
```

唯一的正式预测文件是包内 `generation/predictions.jsonl`，SHA256：

```text
ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81
```

## 目录

```text
release_tools/    干净入口：准备数据、锁定 V9 配置、评测、封装和校验
src/             原 V9 实现及导入依赖，逐字节保留
tests/           原 V9 测试 + 发布封装测试
contracts/       Self/User/Decision/Actor/Judge Prompt 和 Schema 导出
provenance/      原始提交、原始运行 manifest、数据哈希、源代码哈希
reports/frozen/  原 519 条结果汇总，不是新实验
docs/            方法、数据、操作、结果、训练状态、交接
dataset/         本地下载的 10 个官方 JSON；不上传
results/         新复现实验的输出；不上传
```

## 冻结原则

`src/` 保留历史名称和默认值，以便核对原始源码。不要直接使用其默认 CLI；统一使用 `release_tools`，它显式固定 DeepSeek Flash、关闭 thinking，并拒绝覆盖已有新运行目录。

旧内部 protocol 名仍为 `realtalk_task1_ours_agentic_v8_low_specificity_continuity`，部分 Schema 名仍含 v4、docstring 仍含 V2；判定 V9 要看提交、模型、Prompt/Schema 哈希和预测文件哈希，不能仅看字符串中的版本数字。

未来任何 Prompt、Schema、数据或模型修改都另开版本。本分支是独立干净归档分支，不用于直接合并替换师姐仓库 main/experiment 的整棵目录。
