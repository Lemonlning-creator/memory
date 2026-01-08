# Memory — 对话代理的记忆模块

这是一个用于对话代理（conversational agent）的轻量级“记忆”子系统，负责存储、检索与管理会话记忆项，以便在后续对话中使用。

## 主要功能
- 管理结构化的记忆条目（读取、写入、序列化到 JSONL）
- 基于向量/句向量检索（使用 Sentence-Transformers 模型）
- 支持数据更新、信任评估与噪声检测模块

## 快速开始

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. 安装项目（使用项目中的 `pyproject.toml`）：

```bash
pip install .
```

3. 运行主程序（示例）：

```bash
python src/main.py
```

> 注意：项目默认依赖列在 `pyproject.toml` 中，确保系统环境满足深度学习模型运行所需的依赖与资源。

## 项目结构（简要）

- `src/` — 源代码目录，主要模块：
	- `src/main.py`：程序入口与示例运行脚本。
	- `src/memory_store.py`：存储层，负责读写 `output/memory_store.jsonl`。
	- `src/memory_builder.py`：构建记忆条目的工具与转换逻辑。
	- `src/memory_structures.py`：记忆数据模型与类型定义。
	- `src/llm_client.py`：与大模型/外部 LLM 的接口封装。
	- `src/prompt.py`：提示模板与生成工具。
	- `src/trust.py`：信任评估逻辑，用于打分角色的信任值。
	- `src/noise_detector.py`：噪声检测/清洗模块。
	- `src/logger.py`：日志封装。
	- `src/voice/`：语音相关工具（ASR/TT S/播放/录音）

- `models/` — 预训练或离线模型文件（例如 Sentence-Transformers 子模型）
- `data/` — 初始与示例数据（`init/`, `trust/`, `update/`）
- `output/` — 运行时生成的输出，如 `memory_store.jsonl`
- `logs/` — 日志目录

## 配置
- 项目使用 `pyproject.toml` 定义依赖与元数据。可根据需要新增配置文件或环境变量（例如 OpenAI API key）。

常见配置点：
- 环境变量（通过 `.env` 或外部 secret 管理）：API keys、模型选择、路径配置等。

## 模型与数据说明
- 使用 `models/paraphrase-multilingual-MiniLM-L12-v2/` 目录下的句向量模型进行语义检索与嵌入计算。
- 初始数据与信任基准见 `data/init/` 和 `data/trust/`。

## 运行示例
- 生成或更新记忆：运行 `src/main.py`，程序会示范如何从输入构建记忆并存入 `output/memory_store.jsonl`。
- 若需自定义流程，可调用 `src/memory_builder.py` 中的构建函数并使用 `src/memory_store.py` 的存储 API。

## 开发与调试
- 日志：查看 `logs/` 下的输出以排查运行问题。
- 本地模型：若使用大型模型或 ONNX 模型，请确保 `models/` 内相应文件完整并已配置正确的推理后端。

建议开发流程：
1. 在虚拟环境中安装依赖并运行单元/示例脚本。
2. 修改 `src/` 中模块并通过示例数据验证行为。

## 输出与持久化
- 运行后，记忆条目和检索结果会写入 `output/memory_store.jsonl`，该文件为 JSONL 格式，每行一个记忆对象。

## 贡献
欢迎提交 issue 或 PR。请在贡献前先在本地复现问题并附上最小可重现示例。

## 许可
当前仓库未在项目中包含明确许可信息。若需开源发布，请在仓库根目录添加适当的 `LICENSE` 文件。

---

如需我把 README 翻译为英文、添加更详细的使用示例或补充接口文档（例如函数签名、参数说明），告诉我你想要的深度和风格，我可以继续完善。

