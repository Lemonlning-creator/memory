# 小具 — 动态用户画像与智能体人设系统

基于大语言模型的对话智能体，核心特性是**动态用户画像**和**动态智能体人设**，两者均按时间维度（过去 / 现在 / 将来）建模，并通过主题记忆模块实现跨会话的上下文感知。

---

## 核心架构

### 1. 用户画像三维度

| 维度 | 含义 | 存储位置 |
|------|------|----------|
| 过去 | 累积的稳定画像（行为模式、偏好、语言风格） | `data/user_domain.json` |
| 现在 | 当前对话中识别的情绪 / 意图 / 情境 | 运行时动态生成 |
| 将来 | 结合过去画像 + 相关记忆预测用户走向及风险 | 运行时动态生成，写入记忆 |

### 2. 智能体人设三维度

| 维度 | 含义 | 存储位置 |
|------|------|----------|
| 过去 | 稳定不变的人设（性格、表达风格、原则） | `data/self_domain.json` |
| 现在 | 当前是否与用户达成情感共情 | 运行时动态生成 |
| 将来 | 基于人设风格决定是否主动引导用户及引导方式 | 运行时动态生成，写入记忆 |

### 3. 记忆模块

- 按**话题**自动分段存储对话内容（JSONL）
- 每条记忆附带该话题期间累积的用户预测、风险、智能体共情与引导
- 检索时结合用户画像的预测/风险做**语义增强查询**，实现跨话题关联（如"今天躺着" → 召回"每天健身"的记忆）
- 噪声过滤：临时无意义对话不触发记忆存储

---

## 目录结构

```
memory/
├── src/
│   ├── app.py               # Flask Web 后端（SSE 流式接口）
│   ├── main.py              # CLI 入口
│   ├── domain.py            # 用户域 / 自我域 / 三维度分析
│   ├── memory_builder.py    # 记忆构建器（话题分段 + 三维度累积）
│   ├── memory_store.py      # 记忆持久化与语义检索
│   ├── memory_structures.py # 数据结构（Memory / UserProfile / AgentPersona）
│   ├── prompt.py            # 所有 LLM 提示词模板
│   ├── llm_client.py        # LLM 调用封装（流式 / 非流式）
│   ├── noise_detector.py    # 噪声检测
│   ├── config.py            # 配置（API Key、路径、模型等）
│   └── voice/               # 可选：TTS / ASR 语音模块
├── web/
│   └── index.html           # Web 前端（聊天界面 + 记忆侧边栏）
├── data/
│   ├── user_domain.json     # 用户画像持久化
│   └── self_domain.json     # 智能体人设持久化
└── output/
    └── memory_store.jsonl   # 记忆存储
```

---

## 安装

```bash
git clone https://github.com/Lemonlning-creator/memory.git
cd memory
uv sync
source .venv/bin/activate
```

---

## 配置

编辑 `src/config.py`，填写 LLM 相关配置：

```python
LLM_PROVIDER = "openai"
LLM_MODEL    = "gpt-4o"
LLM_API_KEY  = "your-api-key"
LLM_BASE_URL = "https://..."   # 可选，使用代理或国内镜像时填写
```

> 不要将真实 API Key 提交到仓库。

---

## 运行

**Web 界面（推荐）**

```bash
python src/app.py
# 浏览器打开 http://localhost:5000
```

Web 界面左侧为聊天区（含可折叠的思考过程面板），右侧为记忆库侧边栏。

**CLI**

```bash
uv run src/main.py
```

---

## 数据流

```
用户输入
  │
  ├─ [并行]
  │   ├── 激活用户域（过去画像）
  │   ├── 激活自我域（过去人设）
  │   └── 三维度分析 ──► 用户现在/将来 + 智能体现在/将来
  │
  ├─ 用预测/风险增强记忆检索
  │
  ├─ 生成回复（整合六个维度）
  │
  └─ 记忆构建（累积三维度 → 话题切换时写入 JSONL）
```

---

## 记忆字段说明

| 字段 | 说明 |
|------|------|
| `topic` | 话题主题 |
| `content` | 对话内容摘要 |
| `keywords` | 关键词列表 |
| `user_prediction` | 该话题期间累积的用户走向预测 |
| `user_risk` | 该话题期间累积的用户风险 |
| `agent_empathy` | 该话题期间智能体的共情内容 |
| `agent_action` | 该话题期间智能体的引导行动 |
