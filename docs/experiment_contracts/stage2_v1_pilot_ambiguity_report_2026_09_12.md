# Stage2-v1 Pilot and Teacher-Label Ambiguity Report

日期：2026-09-12
状态：pilot 完成；ambiguity gate 未通过；未训练网络。

## 执行范围

本轮只执行：

1. Stage2-v1 external-packet collector；
2. seeds 0/1/2 的 hover 与 7 m straight pilot；
3. 跨 episode 近邻歧义与同 context 反事实 PID 隐状态歧义分析。

没有执行正式数据采集、BC、Actor-only gate 或 TD3 fine-tuning。

## Pilot 数据合同

输出目录：`data/behavior_cloning/stage2_v1_external_packet_pilot`。

- 6 episodes，14,955 transitions；
- `training_allowed=false`，只允许诊断；
- `context_contract_id=stage2_v1_external_packet_30d`；
- world-frame `ReferencePacket`，15 Hz，1 s / 16 点；
- `control_time=sim_time+1/120`；
- future reference 为 `control_time+0.2 s`；
- current-reference DSLPID 标签；
- asymmetric-RPM；
- teacher 执行率 100%，Actor 执行率 0%；
- context 重建最大误差 0；
- RPM round-trip 最大误差 0.0001072 RPM；
- 电机饱和率 0。

| Task | Seed | 扰动 | Steps | Done | Position RMSE | Velocity error | Height RMSE | Max roll/pitch | Gate |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| hover | 0 | nominal | 3602 | timeout | 0.000000 m | 0.000000 m/s | 0.000000 m | 0.000000 rad | PASS |
| straight | 0 | nominal | 1383 | success | 0.014578 m | 0.053466 m/s | 0.000213 m | 0.224842 rad | PASS |
| hover | 1 | small positive recovery | 3602 | timeout | 0.004412 m | 0.015522 m/s | 0.003415 m | 0.059954 rad | PASS |
| straight | 1 | small positive recovery | 1383 | success | 0.014906 m | 0.056874 m/s | 0.005549 m | 0.216992 rad | PASS |
| hover | 2 | small negative recovery | 3602 | timeout | 0.004456 m | 0.015575 m/s | 0.003409 m | 0.057038 rad | PASS |
| straight | 2 | small negative recovery | 1383 | success | 0.017787 m | 0.062605 m/s | 0.005393 m | 0.243006 rad | PASS |

## 歧义结果

### 实际 rollout 的跨 episode 近邻

- 97.08% 样本能在另一个 episode 找到 30D 距离不超过 0.05 的邻居；
- close-pair action `L∞`：median `7.8e-10`，p95 `0.0002723`，max `0.008948`；
- 局部标签离散 RMSE / teacher action RMS：`5.10%`；
- 数值上，实际 pilot 分布中的 context→label 基本接近单值。

Torque sign conflict 的总体值为 7.73%，但 hover 的 p90 torque 阈值接近数值零，
因此该项混入了近零 torque 的符号翻转，不能单独作为主要失败原因。正式数据分析需
增加绝对物理 salience floor。

### 完全相同 30D context 的反事实隐状态

对 48 个 pilot context 保持位置、姿态、速度、角速度、current/future reference、
Packet age 和 previous action 不变，只改变 DSLPID 积分及历史状态：

- 原始已观测标签复现误差：p95 `0.00466 RPM`，max `0.00545 RPM`；
- 任一电机归一化 action range：median `0.00468`，p95 `0.13683`，max `0.18134`；
- 任一电机 RPM range：median `47.31 RPM`，p95 `1017.99 RPM`，max `1750.24 RPM`。

因此冻结 30D context 在数学上不能唯一决定 stateful DSLPID 标签。实际 teacher
rollout 上的歧义较小，但 Actor 发生 covariate shift 后可能访问到相同可观测状态、
不同有效 PID 历史的区域。

## Gate 与下一步

`ambiguity_gate_passed=false`。

暂不执行正式大规模数据采集和 BC。下一轮应保持 Packet、30D context、Actor 网络
和动作接口不变，增加 teacher determinization probe：分别量化 `last_rpy`、位置积分、
姿态积分的贡献，并验证一个只由当前可观测量构造 PID 历史状态的 canonical DSLPID
标签器能否继续通过 teacher-only hover/7 m straight gate。

只有 deterministic-teacher gate 通过，或经审查明确接受 stateful teacher 的
on-distribution 近似风险后，才恢复正式数据采集。
