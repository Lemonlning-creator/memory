# PersonaEmp Kimi K2.6 小测记录

## 测试性质

- 日期：2026-07-29
- 模型：`kimi-k2.6`
- 接口：中国区 Kimi 官方 OpenAI-compatible API
- 数据：论文案例构造的 1 个 session、3 个 query
- 方法：`ours`、`base_model`
- 定位：链路与输出质量 Pilot，不是论文正式 Benchmark 结果

## 运行结果

- 6/6 个生成结果成功。
- 最终错误日志为空。
- Ours 三条回复长度分别为 246、192、281 字符。
- Base Model 三条回复长度分别为 1540、1444、1475 字符。
- Ours 保持原仓库 Direct Response Prompt 的简短对话风格。
- Base Model 没有额外限长，因此给出更长、更完整的建议列表。

人工快速检查显示，Ours 能使用亲密小圈子、低社交能量和偏安静沟通等
长期信息，三条回答均与问题相关，没有明显事实冲突或异常格式。

## 调用统计

下面只统计成功结果内的 alignment/response 调用：

| 方法 | Token | 阶段延迟合计 | API Attempts |
|---|---:|---:|---:|
| Ours | 12,340 | 249.19 秒 | 7 |
| Base Model | 1,318 | 119.54 秒 | 3 |

首次画像提取在一次被中断的运行中已成功并写入旧版缓存，因此本次记录无法恢复
其精确 Token 和延迟。代码现已把画像生成 usage 写入缓存；后续正式运行会按唯一
画像单独汇总，不会因断点恢复漏算，也不会按 query 重复计费。

## 运行中发现的问题

1. 国际区 `api.moonshot.ai` 不接受中国区 Key；中国区应使用
   `https://api.moonshot.cn/v1`。
2. Kimi K2.6 非思考模式必须显式传
   `thinking={"type":"disabled"}`，并使用固定温度 `0.6`。
3. 服务曾返回 `engine_overloaded_error`。提高重试窗口并复用画像缓存后完成运行。
4. 401 等确定性客户端错误现已改为立即失败，避免无意义重复请求。

## 尚未执行

本 Pilot 未生成官方 fixed criteria，也未运行 Qwen/DeepSeek 双 Judge，因此没有
Resonation、Expression、Reception 分数。正式评测还需要 PersonaEmp 正式数据、
固定 criteria 和两个 Judge 的可用接口。
