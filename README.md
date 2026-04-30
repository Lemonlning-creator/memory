# 用户画像对话 Agent

这是一个个性化对话系统。系统会在对话过程中读取用户画像、生成个性化回复，并根据用户输入尝试更新画像中的长期信息和近期状态。

## 当前系统架构

当前项目由 4 个核心部分组成：

1. `app.py`
   Flask 服务入口，负责提供页面和后端 API。
2. `agent.py`
   用户画像 Agent 的核心逻辑，包括：
   - 加载/保存用户画像
   - 组装提示词并调用大模型
   - 基于规则和模型分析更新画像
3. `frontend/`
   前端静态页面，包含聊天界面和画像展示界面。
4. `user_profile.json`
   本地用户画像数据文件，系统运行过程中会持续读取和更新。

## 用户画像结构

当前画像主要分为两部分：

- `stable_profile`
  相对稳定、长期有效的信息

## 环境要求

- Python `3.10+`，建议 `3.11`
- 可访问的 OpenAI 兼容模型服务
- Windows、macOS、Linux 均可运行

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 配置说明

### 1. 大模型配置

当前代码中，大模型客户端初始化写在 `agent.py` 的顶部，主要配置项包括：

- `api_key`
- `base_url`
- `MODEL`

当前默认使用的是 OpenAI 兼容接口写法：

```python
client = OpenAI(
    api_key="你的API Key",
    base_url="你的兼容接口地址",
)

MODEL = "你的模型名"
```

如果你要切换模型服务，需要修改 `agent.py` 中这几个值。

### 2. `user_profile.json`

这是系统真实生效的用户画像文件，启动前可直接手动编辑：

- `stable_profile` 用于保存长期稳定信息
- `dynamic_state` 用于保存近期状态

系统运行中会自动写回这个文件。

### 3. `config.ini` 与 `.env` 的说明

项目中目前存在 `config.ini` 和 `.env`，但按照当前代码实现：

- `app.py` 还没有读取 `config.ini`
- `agent.py` 也还没有读取 `.env`

因此它们当前更接近“配置样例”或“后续可接入的配置文件”，并不是现阶段的真实生效入口。

如果只按当前系统运行，优先关注：

- `agent.py` 中的大模型配置
- `user_profile.json` 中的画像初始数据

## 启动方式

### 方式一：启动 Web 系统

在项目根目录执行：

```bash
python app.py
```

浏览器访问：

```text
http://127.0.0.1:5000
```

即可打开聊天界面和用户画像面板。

### 方式二：启动命令行聊天模式

如果只想在终端中测试 Agent，可以执行：

```bash
python agent.py
```

这会进入一个简单的命令行对话模式，不依赖前端页面。

## 推荐启动步骤

1. 安装依赖：`pip install -r requirements.txt`
2. 修改 `agent.py` 中的 `api_key`、`base_url`、`MODEL`
3. 检查或编辑 `user_profile.json` 的初始画像
4. 运行：`python app.py`
5. 打开浏览器访问 `http://127.0.0.1:5000`