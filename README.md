# 小具动态记忆构建系统

这是一个用于对话型智能体的记忆构建与语音交互示例项目（中文说明）。
项目把记忆构建、记忆存储、噪声检测、要素提取与 TTS/ASR 语音能力整合在一起，便于做原型验证或二次开发。

## 主要功能
- 记忆动态构建与主题管理（持久化为 JSONL）
- 噪声检测（使用 LLM 判断临时噪声）

## 目录结构（简要）
```
./
├─ pyproject.toml          # 项目依赖声明
├─ src/
│  ├─ main.py              # 程序入口（CLI）
│  ├─ main_voice.py        # 语音交互主逻辑（ChatSession / VoiceManager）
│  ├─ memory_builder.py    # 记忆构建器
│  ├─ memory_store.py      # 记忆存取（JSONL 持久化）
│  ├─ memory_structures.py # 记忆数据结构
│  ├─ noise_detector.py    # 噪声检测逻辑
│  ├─ llm_client.py        # LLM 调用封装（流式/非流式）
│  ├─ prompt.py            # 各类提示词模板
│  │  └─ config.py         # 配置（请勿在公共仓库中提交真实密钥）
│  └─ voice/               # 语音相关实现（可选模块）
│     ├─ tts/              # 多种 TTS 实现（kdxf, nailong, edge, local 等）
│     └─ asr/              # ASR 实现（如 tencent_asr）
```

## 安装与依赖
建议使用虚拟环境（venv / conda）来隔离依赖。

1. 克隆仓库并进入目录：
```bash
git clone https://github.com/Lemonlning-creator/memory.git
cd memory
```

2. 创建并激活虚拟环境：
```bash
uv sync
source .venv/bin/activate
```

## 配置
- 配置文件：`src/config.py` 包含默认值和示例配置（LLM 提供商、模型、路径等）。
- 强烈建议不要把真实的 API Key 或凭证直接提交到仓库。可用环境变量或 `.env` 文件配合 `python-dotenv` 加载敏感信息。

## 运行
```bash
uv run src/main.py
```

或运行语音交互入口（如果你只想测试语音一体化模块）：

```bash
uv run src/main_voice.py
```

## 日志与持久化
- 记忆会按主题以 JSONL 追加保存到 `src/config.py` 指定的 `MEMORY_JSONL_PATH`（默认 `./output/memory_store.jsonl`）
- 日志文件路径与级别同样在 `src/config.py` 中配置，默认输出到 `./logs/` 目录