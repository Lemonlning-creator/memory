# 方法

## 流程

```
V9 冻结产物（Self Domain / User Domain / Decision）
                ↓  仅重跑这一步
        Actor（原合同 + Rule A+B 条件约束）
                ↓
        目标人物的下一条消息（纯文本）
```

## 触发条件

```python
applied = args.arm in ("ruleA", "ruleAB") and not asked
```

`asked` = 伙伴上一句是否含问号。不满足时，追加 Rule A+B 约束。

## 规则原文

见 `method/rule_clause.txt`（逐字从 `method/actor_replay.py` 提取）。
