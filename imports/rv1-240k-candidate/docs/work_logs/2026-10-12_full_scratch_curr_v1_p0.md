# full_scratch_curr_v1 — seed43 P0 工程冒烟

2026-10-12。结论：**P0 通过，正式运行停在 2,000 条上层转移；未启动 P1。** 这次验证了 Full 训练数据链路和独立的网络更新链路，尚未验证学习效果、7m 策略成功率或 Stage 3 能力。

## 配置与实现

注册 `full_scratch_curr_v1`，继承 `upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1`。Actor、双 Critic、各自 GRU、117D 当前观测 + 8×100D 历史、Replay、分层成功 Replay、动作投影、奖励参数、课程晋级门槛、探索和更新日程均复用基线。

实验差异为 Full/DSLPID 执行、关闭地图内部 simple mix、评估/保存间隔 5k，以及独立输出目录和审计/完整续训支持。没有加载 K1 Actor/Critic/优化器，没有启用 `full_ft_v1` 或其 Actor 冻结。初始 Actor/Critic 哈希、空优化器状态、完整参数保存在 `checkpoints/full_scratch_curr_v1_seed43/initialization.json`。被冻结的 K1 四 seed 选定源共 20 个文件与原 manifest 校验一致。

起点课程拆为共用 mixin；Full 类仅在 reset 时复用 rigid-body 放置、碰撞刷新函数，step 的继承链进入 `LearnedUpperDSLPIDEnv.step`，不经过 K1 积分。原始起点保留概率仍为 `(0.20, 0.20, 0.30, 0.50, 1.0)`。达到 L4 后保持原始 7m 起点，再由原障碍课程门槛晋级。地图 atlas 会覆盖构造参数，所以 Full 类构造完成后再次强制 `CURRICULUM_SIMPLE_MIX_PROBABILITY=0`。测试覆盖 Stage 0–3 的 1/2/3/5 棵路线树和原始起点。

## 实际执行

```bash
cd /home/tequila/UAV-Latent-RL
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-full-scratch \
/home/tequila/miniconda3/envs/drones/bin/python -u -m scripts.run_change_td3 \
  full_scratch_curr_v1 --seed 43 --total_steps 2000
```

本机 CUDA 不可用，本次使用 CPU。重复从头运行相同 checkpoint 目录会被拒绝，防止覆盖已有结果。一次启动前断言把预处理观测与原始观测比较，导致在第 0 步停止；已改为相同预处理语义，零步启动产物保留在 `*_setup_attempt0`，未并入正式 P0 数据。

| 项目 | 结果 |
|---|---:|
| 上层转移 / 主 Replay | 2,000 / 2,000 |
| 实际 120Hz 下层控制 tick | 15,937 |
| 完整回合 | 11 |
| 成功 / timeout / 碰撞 / 高度越界 / 姿态越界 | 2 / 9 / 0 / 0 / 0 |
| 预算末尾未完成回合 | 1（178 步） |
| 近目标 / 原始起点 reset（含未完成回合） | 10 / 2 |
| 成功 Replay | 193 条，全部来自两次真实成功回合 |
| 课程位置 | L0 / Stage 0，未强制晋级 |
| 完整回合进入 1.2m 引导区 | 9/11（81.8%） |
| 完整回合最终目标距离均值 | 2.2004m |
| 全部下层 tick 位置参考跟踪 RMSE | 0.02836m |
| 下层树木净空 min / P05 / P50 / P95 | 0.43768 / 0.68301 / 1.25831 / 1.70879m |

两次成功分别结束在累计第 1,424、1,460 步，回合长度 157、36，最终三维距离 0.1987m、0.1988m。成功回合转移数之和恰为 193。以上成功率来自随机探索的训练起点分布，不代表训练后的策略成功率；2 个原始 7m 回合均 timeout。

1,989 条转移执行 8 ticks；9 条执行 2 ticks，另有 3 和 4 ticks 各 1 条，均为真实终止或超时截断。保留原环境终止和超时判断，无填补 tick、无按 8 倍估算奖励。Replay 的 done 沿用基线 `terminated or truncated`，包括 timeout，不在此实验改变 bootstrap 语义。

## P0 契约证据

- 每次 reset 审计 UAV 位置、姿态、线速度/角速度、碰撞检测，以及 DSLPID 所有数值内部状态与 fresh controller 一致。Reference Generator 位置重设到课程起点，参考速度/加速度/命令速度清零，packet/sample 清空，计数和时间清零。
- 917D reset 观测与当前位置重新计算的预处理观测一致；历史 8 帧均来自新起点，没有跨回合历史。离线检查全部 2k 转移的历史移位、相邻状态连续性及有限值。后续在线断言也检查新历史帧内容。
- 每条上层转移的奖励与实际下层 reward 精确求和一致，Full 连续项积分使用 `1/120s`；Replay reward 使用原 `reward_scale=0.1`。
- 所有 packet sample 有效，packet version 与上层转移关联正确；四电机实际 RPM 与 DSLPID 经既有 motor codec 输出一致。
- Replay 三维动作逐条等于投影后的执行动作，XY 范数不超过 1。随机探索 XY 落在投影边界比例 22.95%，不是 Actor 饱和；未训练 Actor 固定 probe 的轴饱和和 XY 边界比例均为 0。
- 每个回合都保存上层/下层轨迹及对应 Replay 数据。`upper_policy_action` 为当时 Actor 输出，`upper_action` 为实际执行的随机探索/投影动作；P0 不把两者混为一谈。

完整第一回合：`runs/full_scratch_curr_v1_seed43/p0_audit/traces/episode_at_0.{json,npz}`，对应 `episode_at_0_replay.npz`。第一成功回合：`episode_at_1267.{json,npz}` 与 `episode_at_1267_replay.npz`。上层文件包含预处理前向输入、Actor/执行动作、packet endpoint/version/time；Replay 文件包含 917D next_state、动作、缩放奖励、terminated/truncated、实际 tick 数、packet positions/velocities；下层文件包含参考/实际位置速度、姿态、角速度、四电机 RPM、奖励与净空。

## 网络更新与稳定性

基线 `start_timesteps=5000`、`update_after=5000`，因此**正式 P0 的 Actor/Critic 更新次数都是 0**，初始与原始 2k 网络逐参数相同。这是原配置的预期行为，不能声称 2k 生产运行已完成学习更新。

在正式 2k Replay 和成功 Replay 的隔离副本上，运行 8 次 Critic 更新、4 次延迟 Actor 更新，网络参数均发生变化，Adam 状态建立，梯度/损失/Q/TD 误差有限。Critic loss 范围 0.10896–12.56248，Critic 梯度范数 0.1820–2.3661，TD1/TD2 绝对误差均值范围分别 0.1865–0.5876、0.1775–0.4763。探针没有写回正式 checkpoint，其 SHA256 前后相同。

训练日志增加 Q1/Q2 均值、标准差、范围、TD 绝对误差，保留原梯度和动作饱和统计；固定状态探针记录 Q 分位数、朝目标径向平面速度、垂直速度、相邻保存时策略 RMS 变化。正式 2k 相对初始化策略变化为 0，固定 probe 的径向/垂直命令均值约 +0.0476/+0.00845m/s。8 次更新探针只证明链路可执行，不证明长程稳定性。

## 续训与 checkpoint 保护

`training_state.pt` 保存在线网络、独立 target 网络、Actor/Critic Adam 状态及参数关联、更新计数、主/成功 Replay、课程窗口/等级/计数、课程/地图/NumPy/Python/Torch RNG、best/guard callback 状态及累计上层步数。保存发生在训练末尾 best 恢复之前；另外保存 `model_raw_final_*`。后续每 5k 也保存原始网络和完整续训快照。

**续训语义是开始新的物理回合，不是恢复暂停瞬间的 PyBullet 世界。** 预算截断回合的已有合法转移留在主 Replay；不把未完成回合计入课程统计，也不作为成功轨迹。重新初始化的是新回合物理状态、PID/参考/历史，不是学习器或课程。当前预算截断回合为 178 步。

隔离副本 CLI 实测从 2,000 到 2,016 步：网络/target 不变，前 2k Replay 逐元素不变，成功 Replay 和课程统计不变，新增 16 条 Full 转移。正式目录仍为 2,000 步。另有非空 Adam、非同步 target 的单元测试，验证保存/恢复后下一次采样更新与未中断分支逐参数一致，覆盖优化器状态不能在零更新 P0 中实证的问题。

未来若决定进入 P1，使用累计预算和完整快照（本次没有执行）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-full-scratch \
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 \
  full_scratch_curr_v1 --seed 43 --total_steps 10000 \
  --resume_training checkpoints/full_scratch_curr_v1_seed43/training_state.pt
```

不要用普通 `--load_checkpoint` 作为本实验续训入口；本配置显式拒绝它。参数协议不一致的完整续训也会被拒绝。

## 评估与验证范围

注册的每 5k 评估使用现有固定验证地图种子 0–19，分别执行当前障碍 Stage（原始起点）与固定 Stage 0 / 7m；保留引导区覆盖、终点距离、碰撞/timeout、跟踪误差、逐 tick 净空轨迹。L0–L3 训练回合的近目标难度表现由训练日志单独反映。**P0 未到 5k，因此这两组周期评估尚未实际执行；不报告它们的成功率。** 不强制按步数晋级。

相关回归测试 **28 passed**：Full reset/Stage 0–3、K1 原课程、Full FT 原冻结逻辑、917D recurrent 网络、episode coverage、轨迹导出、配置继承、覆盖保护及完整续训。命令和完整结果保存在 `runs/full_scratch_curr_v1_seed43/p0_audit/tests.log`。pytest 使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免系统 ROS 自动插件缺少 yaml 引起的导入错误。

机器可读证据：同目录 `summary.json`、`isolated_update_probe.json`、`resume_probe.json`、`protected_sources.json`、`policy_diagnostics.jsonl`、`resets.jsonl`、`training_stdout.log`。离线审计可用 `python -m scripts.audit_full_scratch_p0` 重跑，不更新正式网络。
