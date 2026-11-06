# full_ft_v2 保守微调：预检与启动阻塞

状态更新：用户已明确答复“允许仅重放既有 seed43 训练前缀；新增验证禁用40–79”。A/B/C已按下列预设判据完成，每个分支均通过4k完整状态精确恢复核验，并在5k/6k三路零成功后于6k早停。统一结果见[消融报告](2026-10-12_full_ft_v2_ablation_results.md)。以下阻塞分析保留作为决定依据。

## 阻塞的具体依据

用户同时要求复现 full_ft_v1 seed43 的相同 4k 前缀，并禁止使用环境 seeds40–79。现有经过验证的确定性复现方法会走 `TD3Trainer.__init__()`，其中 `trainers/td3_trainer.py:76` 调用 `self.env.reset(seed=args.seed)`。原快照 `args.seed=43`。

这不是仅供网络初始化的种子：`envs/forest/core.py:251–254` 将非空 reset seed 直接传给 `np.random.default_rng(seed)`；同文件 `_addObstacles()` 将这个 RNG 传给地图生成器。训练和验证的地图目录名称不同，并不能消除这次环境 seed43 的使用。

4k 学习快照明确记录 `simulation_live_resume_supported=False`。不能简单加载网络、优化器和 Replay 后另开一局环境，就声称恢复了同一训练起点。把训练 reset seed 改成80或其它允许值，也不再是已验证的同一前缀。当前没有尝试这些不可比较的替代运行，也没有将更换 API 或 RNG 状态注入当作绕过禁令的方式。

用户已允许**仅重放既有 seed43 训练前缀**的限定例外；所有新增验证严格使用80–89。后续训练延续原环境RNG，episode reset不再显式传入40–79中的seed。

## 锚定代码的真实语义

- `TD3UpperRecurrent.set_policy_anchor_from_current()` 在权重为正时深拷贝当前 Actor，置为 eval，并关闭其参数梯度。
- `train()` 从成功 Replay 的同一次 minibatch 抽样获得 `anchor_states`，计算归一化三维动作的 `F.mse_loss(actor(anchor_states), source_actor(anchor_states))`。默认 mean reduction 同时平均样本和动作维度；不是 m/s 空间损失，不是参数距离，也不是模仿 Replay 中已存的执行动作。
- v1 成功 Replay 比例为0.2；没有成功抽样状态或没有显式初始化锚定网络时，权重0.1不会自行启用有效锚定。
- 4k 当前 Actor 等于源 Actor，因此首次 Actor 更新之前的锚定损失及其梯度在数学上为零。GRU有梯度/无梯度路径可能产生约浮点精度的差别，测试中MSE为4e-17量级；仅加该约束不应被描述为必然减小第一次更新。
- `BestCheckpointCallback.on_step()` 保存合格 best 后会无条件调用 `set_policy_anchor_from_current()`，即使 `enable_conservative_phase=False`。B/C 运行时必须锁定4k源策略，防止最佳模型回调改变约束目标。
- 不应调用 `enter_conservative_phase()` 来实现 A/C，因为它还会改变 Critic LR、目标噪声和更新频率。应只设置指定优化器学习率。

原损失可以复用，但直接修改 CLI 权重不能满足固定源策略的语义。`scripts/run_full_ft_v2.py` 在4k完整核验之后启用原损失，并锁定锚定目标；生产TD3学习代码保持原实现。

## 起点核验：启动前静态检查与实际恢复

统一起点：`checkpoints/full_ft_v1_unfreeze_audit_20261012/seed43/model_004000_full_snapshot.pt`。

| 检查 | 结果 |
| --- | --- |
| 上层转移 / Critic 更新 / Actor 更新 | 4000 / 3489 / 0 |
| 主 Replay / 成功 Replay | 4000 / 3459 |
| Actor 与目标 Actor | 均逐张量等于冻结源 Actor |
| Actor Adam | 空状态 |
| 主 Replay 状态与动作 | 与原始训练轨迹前4000步逐项相等 |
| 主 Replay 奖励 | 与原轨迹奖励乘0.1后逐项相等 |
| 源 Actor SHA-256 | 与冻结清单匹配 |
| 本次实际恢复 | A/B/C各自六线程重放前4k；全部保存字段逐项相等、全轨迹差异为0 |

快照 SHA-256：`d397dce0c6a356141dc4c069401ff186860ccba080dd024b19b8f2011ee90499`。

源 Actor SHA-256：`88a55281a313dbd75ee892951241ad3bf515a6d56e10d19b00e058c67199c757`。

机器可读预检：`analysis_outputs/full_ft_v2_seed43/preflight.json`；本次三个分支的实际恢复证据在各分支`prefix_verification.json`。历史 `reproduction_summary.json` 记录了原5k复现的全轨迹及最终Actor/Critic最大差为0，该文件仅作为历史依据。

## 约束解决后适用的预设实验判据

以下在看到任何v2分支结果之前记录；隔离运行器已按此实现。

| 分支 | Actor LR | Critic LR | 固定源动作锚定权重 |
| --- | --- | --- | --- |
| A | 1e-4 | 3e-4 | 0 |
| B | 3e-4 | 3e-4 | 0.1 |
| C | 1e-4 | 3e-4 | 0.1 |

1. 每个分支先按原方法使用六CPU线程确定性重放到4k。在改变LR或启用锚定前，核验主/目标网络、优化器、Replay、成功Replay、待提交成功转移、课程采样器、RNG及观测/控制状态。任一不一致则停止，保存差异。
2. 固定917D观测、Recurrent TD3、Full物理/奖励/控制、Stage0/3三比七课程及原探索噪声日程。总步数包含原4k前缀。
3. Actor更新1/10/50/100/500次对应预期全局步4001/4019/4099/4199/4999，按实际更新计数触发。保存主/目标网络和优化器；固定192状态库记录动作漂移、Q1/Q2/minQ，近目标64状态记录朝目标径向和垂直速度偏置。保存4k基线供崩溃前对照。
4. 短期检查至少到6000步；5000及6000步均评估Full Stage0、Full Stage3、K1 Stage3的80–89十张地图。只有两次评估三路均0成功，才判为“持续性全面崩溃”并允许在6000提前停；数值非有限则作为异常停止，不能混称性能崩溃。保存5000/6000原始网络和完整诊断。
5. 未全面崩溃者到10000步。每5000步保留三路逐图success/collision/timeout和其它终止原因，以及固定库Actor漂移、Q值、近目标速度偏置。
6. 延长40k的稳定性门槛：5000和10000两次均满足Full Stage0成功率≥0.8、Full Stage3≥0.7、K1 Stage3≥0.8；所有参数及Q诊断有限。该门槛是本实验预设，不是已验证最优阈值。延长后每5k检查；若连续两次三路均0成功，可提前终止。
7. 原始终点单独保存为`model_raw_final`。`model_best`必须明确标注选择时刻及是否满足稳定窗口；若无合格微调模型，单独记录“无合格best”，保留源策略fallback并明确命名/标注，不冒充适应收益。不得用best恢复后的评估覆盖raw终点记录。
8. 统一报告比较相对源策略的Full Stage3逐地图变化，单独报告Stage0/K1保留和训练稳定性。A对C隔离低LR下锚定效果，B对C隔离有锚定时LR效果；原v1只作已验证前缀的历史参考。单训练seed、十张验证地图的结果不支持普遍最优或正式held-out收益结论。

## 测试

锚定/保守阶段、CPU RNG保护、Q排序近似平局处理、TD目标/终止mask/奖励尺度四项现有测试通过。新增测试验证完整状态差异检测、三路稳定/崩溃判据以及首次锚定更新的精度边界。测试未创建仿真环境。第一次pytest启动被环境自动加载的ROS插件因缺少yaml阻断；关闭无关插件自动加载后执行通过。
