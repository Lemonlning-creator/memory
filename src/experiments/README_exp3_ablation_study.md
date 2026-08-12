# Experiment 3：Deep Empathy 消融实验

入口：`src/experiments/exp3_ablation_study.py`

模拟用户与画像发现评估：`src/experiments/exp3_user_simulator.py`

## 实验协议

| 子实验 | 切分 | 主要条件 | 交互方式 | 评估重点 |
|---|---:|---|---|---|
| Exp3-A 显式用户建模 | 90/10 | Explicit / w/o Explicit | 固定 REALTALK 回复 | 八项回复指标与严格配对差值 |
| Exp3-B Adaptive Exploration | 50/50 | Adaptive；可选 Fixed omega | 隐藏画像模拟用户 | 新增画像、正确性、轮数、完善曲线 |
| Exp3-C Bayesian Updating | 50/50 | Bayesian Online / Static | 固定 REALTALK 时序回放 | 八项回复指标及 Early/Middle/Late 配对差值 |

默认条件 key：

```text
explicit_user_modeling
wo_explicit_user_modeling
adaptive_exploration
fixed_exploration
bayesian_online
static_profile
```

## Exp3-A：显式用户建模

- 前 90% Session 抽取五层用户画像及目标 Agent persona。
- 后 10% Session 固定回复评估。
- 两组使用完全相同的真实历史、模型、persona 和探索策略。
- 两组测试期间均冻结 `static_profile`，避免混入 Bayesian Updating 的作用。
- `wo_explicit_user_modeling` 不读取持久用户画像，只根据当前消息、近期历史和 Agent persona 做临时推断。
- 数据集真实回复进入记忆；模型生成回复只用于评估。
- 比较前严格验证两组 `example_id` 完全一致，再计算逐样本配对差值。

## Exp3-B：Adaptive Exploration

### 隐藏目标构造

1. 前 50% Session 得到初始画像 `P0`。
2. 全部 Session 使用同一个一次性画像抽取流程得到参考画像 `P*`。
3. 仅从后 50% 的真实用户消息抽取带 `dia_id` 的稳定证据 claim。
4. 对 `P*` 中每条 atomic claim 相对 `P0` 做语义审计：`known`、`new`、`refinement`、`contradiction`、`unsupported`。
5. 隐藏目标 `H` 仅保留具有后 50% 用户消息直接证据的 `new` 或 `refinement`；矛盾与无证据内容不进入目标集合，但保存在审计记录中。

这里比较的是 claim 语义，而不是简单比较字段是否为空。同一个字段新增了更具体且兼容的信息时，记为 `refinement`。

### 两阶段模拟用户

- 开场生成器只看 `P0` 和风格示例，不得披露隐藏目标。
- 私有披露控制器根据问题相关性、信任、疲劳、敏感度和重复提问决定 `disclose / withhold / refuse / none`。
- 每轮最多批准两条尚未披露的隐藏 claim。
- 可见回复生成器只能看到本轮获批 claim，无法访问其余隐藏画像。
- 所有模型 JSON 必须精确符合 schema；缺字段、多字段、非法 ID 或越界数值都会立即报错，不做静默默认或截断。

### 评估

设隐藏目标为 `H`，截至第 `T` 轮实际披露的目标为 `R_T`，画像更新后学到的目标为 `L_T`：

- Elicitation：`|R_T| / |H|`
- Uptake：`|L_T ∩ R_T| / |R_T|`；当 `R_T` 为空时输出 `null`，不写成 0
- End-to-end discovery：`|L_T| / |H|`
- 新增画像正确率及 unsupported novel claim rate
- 首次发现轮次、每发现一条画像所需轮数、每轮发现效率
- 逐轮累计披露/学习曲线及 Coverage AUC
- 五层画像的 target、revealed、learned 覆盖率
- 每条隐藏 claim 的原始证据、首次披露轮次和首次学习轮次
- 用户负担、拒绝率和探索提问率

只运行 `adaptive_exploration` 时，结果文件标记为 `capability_only`，只能证明模型具备探索能力，不能证明 Adaptive Exploration 相对更优。只有显式选择比较条件时才输出比较性结论。

## Exp3-C：Bayesian Updating

- 前 50% Session 为两组建立完全相同的初始画像。
- 后 50% Session 按时间顺序正常回放。
- `bayesian_online` 从长期记忆持续更新画像；`static_profile` 冻结初始画像。
- 每轮模型生成回复只用于评估，数据集真实 Agent 回复才进入记忆，确保后续历史完全相同。
- 除整体八项回复指标外，必须按每个 case 的时间进度分为 Early、Middle、Late，并输出两个条件逐样本、逐阶段的配对差值。

## 运行

先生成不调用 API 的计划：

```bash
python -m src.experiments.exp3_ablation_study --phase plan
```

单独运行一条轨道和一个 case：

```bash
python -m src.experiments.exp3_ablation_study \
  --phase prepare \
  --track exploration \
  --condition adaptive_exploration \
  --case <case-id>

python -m src.experiments.exp3_ablation_study \
  --phase generate \
  --track exploration \
  --condition adaptive_exploration \
  --case <case-id> \
  --sim-rounds 20 \
  --sim-seeds 1

python -m src.experiments.exp3_ablation_study \
  --phase evaluate \
  --track exploration \
  --condition adaptive_exploration \
  --case <case-id>
```

建议先用一个 case、一个 seed 做端到端验证，再扩大规模。旧协议缓存不会被静默复用；协议版本、数据哈希、条件、seed 或 schema 不一致时会直接报错，需使用新的输出目录。

## 主要输出

```text
data/exp3_ablation_study/
├── experiment_plan.json
├── split_explicit_90_10.json
├── split_online_50_50.json
├── shared/
│   ├── explicit_90_10/
│   ├── online_50_50/
│   ├── hidden_profiles/
│   │   └── cases/<case>/
│   │       ├── hidden_user_profile.json
│   │       ├── hidden_claim_manifest.json
│   │       └── simulator_context.json
│   └── simulator_scenarios/
└── tracks/
    ├── explicit/
    ├── exploration/
    │   └── conditions/<condition>/cases/<case>/simulations/<seed>/
    │       ├── dialogue.jsonl
    │       ├── profile_snapshots/
    │       ├── generation_summary.json
    │       └── evaluation.json
    └── bayesian/
```
