# 用户画像对话 Agent

这是一个个性化对话系统。系统会在对话过程中读取用户画像、生成个性化回复，并异步更新画像中的长期稳定信息和近期状态。

## 当前系统架构

项目由 5 个核心部分组成：

1. `app.py`：Flask 服务入口，负责页面和后端 API。
2. `frontend/`：聊天界面和画像展示界面。
3. `src/agent.py`：对话、记忆和画像接入逻辑。
4. `src/profile_batch_updater.py`：原始对话批处理、独立画像模型调用、格式校验和字段级合并。
5. `user/*_profile.json`：本地用户画像及其待处理队列。

## 用户画像更新

画像保持现有五层结构：`core`、`regulation`、`cognition`、`identity`、`behavior`。每层包含一条 `summary`，其他属性只能来自该层的既有字段白名单。

更新只读取本批原始对话，不使用被压缩的中期或长期记忆。默认累计 8 条用户消息或等待 15 分钟，任一条件满足即在后台处理。待处理消息写入画像同目录的 `.pending.json` 文件，只有画像成功校验、合并并保存后才会移出队列。

模型输出必须包含完整五层结构。服务端会校验层名、字段名、置信度和本批证据消息 ID，再按字段合并，禁止整块覆盖画像。失败时在同一任务上下文中带校验错误重试一次；任务结束后不保留纠错对话。

## 环境要求

- Python 3.11+
- 可访问的 OpenAI 兼容模型服务
- Windows、macOS、Linux

## 配置

主聊天模型继续使用：

```text
API_KEY=主聊天模型密钥
BASE_URL=主聊天模型接口地址
```

独立画像模型使用：

```text
PROFILE_API_KEY=画像模型密钥
PROFILE_BASE_URL=https://api.moonshot.cn/v1
PROFILE_MODEL=kimi-k2.6
PROFILE_BATCH_MESSAGES=8
PROFILE_BATCH_SECONDS=900
```

不要把真实密钥提交到仓库。项目已忽略 `.env`。

## 启动

1. 安装依赖：`pip install -r requirements.txt`
2. 在本地 `.env` 配置主聊天模型和画像模型。
3. 检查或编辑初始画像。
4. 运行：`python app.py`
5. 访问 `http://127.0.0.1:18201`

## 测试

运行离线单测：

```bash
python -m unittest tests.test_profile_batch_updater -v
```

单测覆盖五层格式约束、字段级合并、任务内纠错重试，以及失败时不推进待处理队列。
