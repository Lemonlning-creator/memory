# 新版实验工作区

## 基线

- 上游仓库：`Lemonlning-creator/memory`
- 上游分支：`experiment`
- 基线提交：`491145961e5ce51ae144c3bb5d84f62d027e8df8`
- Fork：`Nobody-ly/memory`
- 开发分支：`paper-boost/revised-experiment-suite`

本分支用于实现新版 Exp1–Exp3，与此前实验修正分支和本地测试数据隔离。

## Prompt 与行为保护原则

新版实验优先复用现有系统的核心生成链路，不重新设计角色：

1. 不改写现有陪伴智能体的人设、语气、互动原则和共情行为。
2. 不为提高单项实验分数而在核心 System Prompt 中加入答案暗示。
3. 实验需要的画像、历史、状态等信息通过现有模板字段或独立实验输入传入。
4. 输出 Schema、Judge Prompt 和指标提取 Prompt 属于评价协议，可以独立实现，但不得反向进入待评价回复的生成 Prompt。
5. 若官方复现必须使用固定 Prompt，将其作为单独 Baseline 适配器保存，不覆盖 Ours 的核心 Prompt。
6. 上游未来确实修改核心 Prompt 时，先单独审查语义差异，再显式更新基线，不能静默接受。

以下文件视为人设和行为基线：

- `src/prompts/templates.py`
- `src/prompts/templates_en.py`
- `src/prompts/prompt_loader.py`
- `src/agent.py`
- `src/memory_os_local.py`

运行以下命令可检查核心 Prompt 和行为链路是否被意外修改：

```powershell
python tools/verify_core_prompt_baseline.py
```

基线哈希保存在 `experiment_protocol/core_prompt_baseline.json`。

## 实验开发边界

- 新实验实现优先放在新的实验模块中。
- 共享的重试、缓存、checkpoint 和运行清单可以抽成独立基础设施。
- 原始数据、API Key、缓存和完整实验输出不得提交。
- 调参只在开发集或小规模 Pilot 上进行；协议锁定后再运行正式测试集。
