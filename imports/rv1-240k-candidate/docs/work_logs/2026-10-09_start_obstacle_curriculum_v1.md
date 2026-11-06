# 起点课程之后逐级解锁随机障碍：startobscurr_v1

## 实验目的与改动范围

当前四个 `startcurr_v1` 选中策略在原起点 Stage 0 达到 80/80，而随机 Stage 3 只有 14/80；seed 45 的同一策略在 Full 固定五树地图为 20/20，在随机五树地图为 5/20。新实验检验：**保留已验证的起点课程，然后训练随机障碍 Stage 1–3，能否减少入目标引导区前的碰撞**。奖励、TD3、h8、K1、rayfix、圆盘动作表示、成功球与原 36 树训练地图保持不变。`startcurr_v1` 及四个冻结 checkpoint 不改动。新模式从头训练，不能把旧 checkpoint 当作完整续训：当前 checkpoint 不含 replay 与课程状态。由于评估地图和阶段化 checkpoint 选择也必须随目标改变，这不是严格的“只改一行障碍阶段”的单变量消融；本轮固定策略网络、奖励与物理动作链，报告时应明确这几处配套训练机制变化。

新增模式：`upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1`。训练起初完全沿用 `startcurr_v1` 的 L0–L4 起点距离课程，障碍固定 Stage 0（一棵挡路树）。达到 L4（全部原 7 m 起点）后，障碍晋级只统计**该阶段主难度、原起点**训练回合；短起点回合与 20% 较简单障碍混合回合可以进入 replay，却不参与晋级。至少 100 个合格回合且最近 100 局达到如下门槛才晋级：

| 当前阶段 | 路线挡路树 | 晋级成功率 | 碰撞率上限 | 下阶段 |
| --- | ---: | ---: | ---: | --- |
| Stage 0 | 1 | 80% | 15% | Stage 1 |
| Stage 1 | 2 | 75% | 20% | Stage 2 |
| Stage 2 | 3 | 70% | 20% | Stage 3 |
| Stage 3 | 5 | 持续记录 | 持续记录 | 最终阶段 |

这些是首轮可检验门槛，不是已证明最优的训练参数。Stage 3 没有自动“毕业”判断。障碍阶段达到目标后不会因短期回退自动降级；因此保留模型选择与周期评估。

## 验证与选模型口径

周期评估从原 7 m 起点出发，使用新增 `procedural_forest_multitree_validation_random_v1`：与训练地图相同的 36 棵树、随机挡路树生成规则以及 1/2/3/5 数量，但**完全关闭简单阶段混合**。评估环境跟随训练端当前障碍阶段；每 4000 步对固定环境 seeds 0–19 做 20 局。旧 `fixed_safe_five` 地图不再参与本实验的 checkpoint 选择；它可在训练后作单独迁移对照。新地图虽标为 validation，仍是同一随机生成器和地图族，不能当成跨域泛化证明。正式复核应另用 seeds 20–39，并报告各训练 seed 而非只报告平均。

选模型采用阶段优先的稳定窗口：跨阶段的评估不能拼成一个窗口；在同样满足单次成功率 ≥60%、碰撞率 ≤20% 的两次连续评估中，优先选更高障碍阶段，再比较同阶段成功率、碰撞率和回报。`model_peak_validation` 另保存最高阶段的单次峰值以供诊断；`model_final_selection.json` 单独标记选中网络所属阶段，即使训练结束时环境已推进到更高阶段。为避免新阶段刚解锁、成功率暂降时原 convergence guard 提前中断，本轮关闭 guard，但仍周期保存和恢复选中 checkpoint。训练上限设为 240k 步，不代表一定会到 Stage 3；若卡在某一级，应按合格训练回合与对应随机验证成功率判断失败点，不跳级。

训练后在未参与周期选择的环境 seeds 20–39 上，对各选中 checkpoint 分别做 K1 Stage 0–3 的 20 局正式评估；目标是至少 3/4 个训练 seed 在随机 Stage 3 达到 ≥50% 成功，并逐 seed 报告碰撞与入 1.2 m 比例。达到该门槛后，再把同一批策略零样本放到 Full Stage 3；只有 Full 明显低于 K1 时才单独做动力学微调。这个 ≥50% 是本项目的首轮工程门槛，不是任务已解决的理论标准。

核心日志：`env/start_curriculum_level`（达到 4 才开始障碍晋级计数）、`env/obstacle_curriculum_stage`、`env/obstacle_curriculum_episode_stage`、`env/obstacle_curriculum_episode_primary`、`env/obstacle_curriculum_stage_episodes`、`env/obstacle_curriculum_success_rate`、`env/obstacle_curriculum_collision_rate`、`env/route_blocking_tree_count` 与 `env/curriculum_mixed_difficulty`。周期评估看 `eval/curriculum_stage`、成功/碰撞/timeout、1.2 m 引导区覆盖及碰撞 A/C 诊断。每次晋级附近重点检查性能回退，不把最后训练网络等同于选中模型。

## 运行

先单独跑曾经失败的 seed 43，观察是否从 L4 晋级 Stage 1，再决定是否顺序运行其余训练 seed：

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1 --seed 43 --total_steps 240000
```

其余训练 seed 可逐条运行，避免覆盖，输出分别带 `_seed42`、`_seed44`、`_seed45`：

```bash
for seed in 42 44 45; do /home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1 --seed "$seed" --total_steps 240000 || break; done
```

## 运行前验证

新模式 2 步训练入口与 2 局随机验证地图评估通过；96-ray h8 的 917D 观测和 K1 动作空间未变。物理环境检查每个阶段的 20 张评估地图都恰有 1/2/3/5 棵挡路树、没有简单阶段混合，地图标记可达。课程调度器测试验证短起点和简单障碍回合不触发晋级，阶段选择测试验证模型选取不会混合两个阶段的评估窗口。未进行完整 240k 训练，因此复杂森林成功率仍待实验验证。
