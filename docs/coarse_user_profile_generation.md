# 五层粗粒度用户画像生成说明

## 1. 这个功能是做什么的

这个功能从一份双人对话记录中识别人类用户，并让大模型生成一份简短的用户画像。

它只保留用户画像最外层的五个维度：

| 字段 | 描述 |
| --- | --- |
| `core` | 用户较稳定的恐惧、愿望、价值观、依恋倾向和意义来源 |
| `regulation` | 用户面对压力、冲突和不确定性时的调节与应对方式 |
| `cognition` | 用户的表达、信息处理、情绪显露、社交距离和决策风格 |
| `identity` | 用户明确表达或可谨慎推断的身份、关系、生活条件和环境 |
| `behavior` | 用户的内容兴趣、消费或娱乐偏好、习惯和长期行为模式 |

与完整画像不同，这五层下面不再生成 `fears`、`values`、`habits` 等子字段，也不输出置信度和证据。每一层直接对应一段或两段中文概括。

## 2. 开始前需要准备什么

建议使用 Python 3.11。项目依赖应已安装到项目根目录的 `.venv` 虚拟环境中。如果没有 `.venv`，启动脚本会尝试使用系统 `PATH` 中的 `python`。

模型连接信息来自项目根目录的 `.env`：

```text
API_KEY=你的密钥
BASE_URL=OpenAI兼容接口地址
```

模型名称和是否启用思考模式来自 `config.ini` 的 `[API]` 部分：

```ini
[API]
model = qwen3.6-flash
enable_thinking = False
```

## 3. 输入格式

输入是一个 UTF-8 编码的 JSON 文件。推荐格式如下：

```json
{
  "name": {
    "speaker_1": "Emi",
    "speaker_2": "elise"
  },
  "session_1": [
    {
      "speaker": "Emi",
      "clean_text": "我最近一直在规划自己的研究方向。"
    },
    {
      "speaker": "elise",
      "clean_text": "你最想优先解决哪一部分？"
    },
    {
      "speaker": "Emi",
      "clean_text": "我更看重长期积累，不想只做短期见效的事情。"
    }
  ],
  "session_2": [
    {
      "speaker": "Emi",
      "clean_text": "遇到复杂问题时，我通常会先拆解再做决定。"
    }
  ]
}
```

输入规则：

1. `name.speaker_1` 表示需要生成画像的人类用户。
2. `name.speaker_2` 表示对话伙伴。
3. 会话字段必须命名为 `session_数字`，例如 `session_1`、`session_2`。
4. 每个 session 必须是消息数组。
5. 每条有效消息需要同时包含非空的 `speaker` 和 `clean_text`。
6. session 会按照数字顺序读取，而不是按照 JSON 中出现的先后顺序读取。
7. 双方发言都会保留，以便模型理解问题、回应和上下文，但模型只分析 `speaker_1`。

语音转文字后，如果只有一段对话，全部消息放到 `session_1` 即可。建议在转写阶段区分说话人，并保证 `speaker` 的值与 `name` 中的名字一致。

原始消息中即使含有 `dia_id`、时间戳、评分或其他字段也不会影响运行，因为发送给模型之前只会保留：

```json
{
  "speaker": "说话人",
  "clean_text": "转写文本"
}
```

空文本、没有说话人的消息，以及 `session_数字` 以外的内容都会被忽略。

## 4. 输出格式

输出是 UTF-8 编码的 JSON 文件，固定只有五个顶层字段：

```json
{
  "core": "用户重视长期成长和具有持续价值的成果，倾向于通过不断学习与创造获得意义。对于停滞或无法发挥潜力可能较为敏感。",
  "regulation": "面对复杂问题时，用户更倾向于主动分析、拆解问题并建立秩序，而不是立即回避。压力下可能通过理性解释和持续优化获得确定感。",
  "cognition": "用户偏好结构化、信息充分的沟通方式，决策前通常会比较多个方案。情绪表达相对克制，更多呈现思考过程。",
  "identity": "对话显示用户处在持续发展专业能力的阶段，并重视长期合作。关于年龄、家庭和经济情况的信息有限，不作进一步推断。",
  "behavior": "用户长期关注研究、知识创造和效率工具，习惯围绕同一问题进行多轮讨论并反复改进已有成果。"
}
```

程序会检查输出：

- 五个字段必须全部存在；
- 不允许出现第六个字段；
- 每个值必须是非空字符串；
- 不允许返回嵌套的 `value`、`confidence` 或 `evidence`。

“一至两段”主要通过提示词约束。程序负责检查字段和数据类型，不会机械限制句数或字数。

如果不指定输出路径，文件默认保存为：

```text
user/{speaker_1}_{speaker_2}_coarse_profile.json
```

例如：

```text
user/Emi_elise_coarse_profile.json
```

## 5. 如何运行

在项目根目录打开 PowerShell，然后运行：

```powershell
.\scripts\run_coarse_user_profile_generation.ps1 `
  -Realtalk dataset\Chat_1_Emi_Elise.json
```

指定输出文件：

```powershell
.\scripts\run_coarse_user_profile_generation.ps1 `
  -Realtalk dataset\Chat_1_Emi_Elise.json `
  -Output dataset\output\user\emi_elise_coarse_profile.json
```

调整送给模型的对话规模：

```powershell
.\scripts\run_coarse_user_profile_generation.ps1 `
  -Realtalk dataset\Chat_1_Emi_Elise.json `
  -MaxUtterances 240 `
  -MaxChars 32000
```

如果 PowerShell 因执行策略拒绝运行脚本，可以仅对这一次命令使用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_coarse_user_profile_generation.ps1 `
  -Realtalk dataset\Chat_1_Emi_Elise.json
```

也可以绕过 PowerShell 脚本，直接调用 Python 模块：

```powershell
python -m src.experiments.coarse_user_profile_generation `
  --realtalk dataset\Chat_1_Emi_Elise.json
```

## 6. 启动参数

| PowerShell 参数 | 是否必填 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `-Realtalk` | 是 | 无 | 输入对话 JSON 路径 |
| `-Output` | 否 | 自动生成 | 输出画像 JSON 路径 |
| `-Config` | 否 | `config.ini` | 模型配置文件路径 |
| `-MaxUtterances` | 否 | `180` | 最多选取多少条双方消息 |
| `-MaxChars` | 否 | `24000` | 清洗后对话 JSON 的最大字符数 |

相对路径统一以项目根目录为基准，因此可以从其他目录调用启动脚本。

## 7. 生成策略

整个生成过程分为以下步骤：

1. **读取输入**：以 UTF-8 读取对话 JSON。
2. **识别人物**：将 `name.speaker_1` 作为目标用户，将 `name.speaker_2` 作为对话伙伴。
3. **清洗消息**：读取所有 `session_数字`，删除空消息，每条消息只保留 `speaker` 和 `clean_text`。
4. **压缩长对话**：消息数超过 `MaxUtterances` 时，在整个时间范围内均匀取样，避免只看到开头或结尾。
5. **控制字符数**：如果清洗后的 JSON 仍超过 `MaxChars`，会继续稀疏取样，并始终保留最近一条消息。单条消息本身过长时才截断该消息文本。
6. **调用模型**：使用温度 `0.2`，降低随机性；最多生成 1600 tokens。
7. **约束画像**：提示模型只总结目标用户的稳定或重复特征，证据不足时保守表达，避免编造身份和心理信息。
8. **校验结果**：解析模型返回的 JSON，并严格检查五层字段和字符串类型。
9. **保存文件**：使用 UTF-8 和中文不转义的方式写入 JSON。

这是一种“一次性总体概括”策略，不会像完整画像脚本那样逐轮建立短期记忆、中期记忆、长期记忆，再持续更新细分画像。因此它速度更快、输出更简单，适合需要一个大致用户印象的场景；但它不会提供逐属性证据、置信度或增量更新能力。

## 8. 使用了哪些 Python 文件

运行链路如下：

```text
scripts/run_coarse_user_profile_generation.ps1
  └─ src/experiments/coarse_user_profile_generation.py
       ├─ src/llm_client.py
       ├─ src/utils.py
       └─ src/experiments/agent_persona_generation.py
```

各文件职责：

- `src/experiments/coarse_user_profile_generation.py`
  - 本功能的主程序；
  - 定义粗粒度画像提示词；
  - 清洗、采样对话；
  - 调用模型并校验、保存结果；
  - 提供命令行参数。
- `src/llm_client.py`
  - 读取接口配置；
  - 调用 OpenAI 兼容的聊天模型；
  - 处理接口重试和 token 使用统计。
- `src/utils.py`
  - 读取、保存 UTF-8 JSON；
  - 从模型文本中解析 JSON。
- `src/experiments/agent_persona_generation.py`
  - 复用其中的 session 排序逻辑；
  - 复用 `speaker_2` 对话伙伴识别逻辑。
- `tests/test_coarse_user_profile_generation.py`
  - 离线检查输入清洗、长对话压缩和输出格式校验；
  - 测试不会调用真实模型。

原来的 `src/experiments/user_profile_generation.py` 是设计参考，但不是新脚本的运行依赖。原脚本会驱动完整记忆系统并生成细分画像；新脚本刻意保持轻量，只生成最外层五层概括。

## 9. 常见错误

### 找不到 Python

```text
Python runtime not found
```

确认项目根目录存在 `.venv\Scripts\python.exe`，或者系统已经安装 Python 并加入 `PATH`。

### 找不到输入文件

```text
Input dialogue file not found
```

检查 `-Realtalk` 路径。相对路径以项目根目录为基准。

### 没有有效消息

```text
No valid dialogue messages found
```

确认至少存在一个 `session_数字` 数组，并且其中至少一条消息同时含有非空的 `speaker` 和 `clean_text`。

### 模型接口调用失败

检查 `.env` 中的 `API_KEY`、`BASE_URL`，以及 `config.ini` 中的模型名称。模型客户端最多自动重试三次。

### 模型输出格式不正确

```text
Invalid profile layers
```

这表示模型没有严格返回五层字符串结构。程序不会保存格式错误的画像，重新运行通常可以解决；如果频繁出现，应检查当前模型是否具有稳定的 JSON 输出能力。

## 10. 如何运行离线测试

测试不需要 API 密钥，也不会调用模型：

```powershell
python -m unittest discover -s tests -p "test_coarse_user_profile_generation.py" -v
```

