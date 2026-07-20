# 用户画像对话 Agent

这是一个个性化对话系统。系统会在对话过程中读取用户画像、生成个性化回复，并根据用户输入尝试更新画像中的长期信息和近期状态。

## 当前系统架构

当前项目由 4 个核心部分组成：

1. `app.py`
   Flask 服务入口，负责提供页面和后端 API。
2. `frontend/`
   前端静态页面，包含聊天界面和画像展示界面。
3. `agent.py`
   用户画像 Agent 的核心逻辑，包括：
   - 加载/保存用户画像
   - 组装提示词并调用大模型
   - 基于模型分析更新画像
   - ...
4. `user_profile.json`
   本地用户画像数据文件，系统运行过程中会持续读取和更新。

## 环境要求

- Python `3.10+`，建议 `3.11`
- 可访问的 OpenAI 兼容模型服务
- Windows、macOS、Linux 均可运行

## 启动步骤

1. 安装依赖：`pip install -r requirements.txt`
2. 修改 `agent.py` 中的 `api_key`、`base_url`、`MODEL`
3. 检查或编辑 `user_profile.json` 的初始画像
4. 运行：`python app.py`
5. 打开浏览器访问 `http://127.0.0.1:18201`