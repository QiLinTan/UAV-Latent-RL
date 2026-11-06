# Stage2-v1 Teacher Determinization Probe

日期：2026-09-12
状态：determinization gate 通过；推荐 canonical world-angular-velocity DSLPID。

## 原因

Pilot 证明 stateful DSLPID 在实际 rollout 邻域中的标签离散较小，但同一冻结 30D
context 在不同 PID 积分和历史姿态下可以产生明显不同的动作。由于 Stage2-v1 不增加
PID 隐状态、不增加动作历史且不修改 Actor 网络，本 probe 寻找仍使用 current-reference
DSLPID 公式、但标签只依赖当前可观测量的确定化教师。

## 候选

| 模式 | 积分状态 | DSLPID `last_rpy` | 是否是 context 的确定函数 |
| --- | --- | --- | ---: |
| stateful | episode 累积 | 上一真实控制周期 | 否 |
| canonical zero-rate | 每标签清零 | 当前姿态 | 是 |
| canonical world-angular-velocity | 每标签清零 | 用当前四元数、world 角速度和 `dt` 反推上一姿态 | 是 |

两个 canonical 模式都在每次 `computeControl()` 前重置 DSLPID。world-rate 模式通过
PyBullet 四元数乘法向后积分一个控制周期，并选择相对当前欧拉角的最短角差；因此不依赖
未写入 30D context 的历史状态。

## 3-seed 闭环结果

每种模式均运行 nominal、small-positive-recovery、small-negative-recovery 三个 seed，
每个 seed 含 hover 和同一条 7 m straight，共 18 episodes。

| 模式 | 通过数 | Hover | Straight | Saturation | Packet/fallback | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stateful | 6/6 | PASS | PASS | 0% | clean | 对照通过，但标签非唯一 |
| canonical zero-rate | 1/6 | FAIL | FAIL | 0% | clean | 缺少姿态速率反馈，不可用 |
| canonical world-angular-velocity | 6/6 | PASS | PASS | 0% | clean | 推荐 |

推荐模式的关键数值：

| Task | Seed | Position RMSE | Velocity error | Height RMSE | Max roll/pitch | Final distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hover | 0 | 0.000000 m | 0.000000 m/s | 0.000000 m | 0.000000 rad | N/A |
| straight | 0 | 0.013559 m | 0.053060 m/s | 0.000214 m | 0.225384 rad | 0.197589 m |
| hover | 1 | 0.004326 m | 0.015225 m/s | 0.003410 m | 0.061756 rad | N/A |
| straight | 1 | 0.014157 m | 0.057107 m/s | 0.005552 m | 0.220249 rad | 0.197589 m |
| hover | 2 | 0.004398 m | 0.015352 m/s | 0.003409 m | 0.057668 rad | N/A |
| straight | 2 | 0.016508 m | 0.060777 m/s | 0.005413 m | 0.238447 rad | 0.197589 m |

## 隐状态贡献分解

在 48 个完全相同可观测输入的反事实样本上，相对原 stateful 标签的任一电机 action
`L∞` 影响如下：

| 改变量 | Median | P95 | Max |
| --- | ---: | ---: | ---: |
| 清零位置积分 | 0.002170 | 0.020147 | 0.022773 |
| 清零姿态积分 | 0.000062 | 0.000072 | 0.000077 |
| 只清除 rate history | 0.000000 | 0.028900 | 0.115167 |
| canonical zero-rate | 0.002337 | 0.046034 | 0.115362 |
| canonical world-angular-velocity | 0.002337 | 0.020212 | 0.023796 |

主要歧义来自位置积分和用于姿态速率估计的 `last_rpy`；姿态积分影响可以忽略。
world-angular-velocity 模式保留可观测的姿态速率反馈，并将相对 stateful teacher 的
p95 差异降至约 0.0202。

## 决策

`teacher_determinization_gate_passed=true`，正式 Stage2-v1 数据采集的 teacher 合同
调整为：

```text
teacher_contract_id = stage2_v1_canonical_dslpid_world_rate
target = current_reference(control_time)
integral_pos_e = 0 before every label
integral_rpy_e = 0 before every label
last_rpy = previous attitude inferred from current quaternion,
           current world angular velocity, and dt
```

Packet、30D context、MotorActionCodec、Actor/Critic 和任务 gate 不变。Torque sign
conflict 保留为局部敏感性诊断，不再单独作为“同 context 多标签”的判据；确定化由相同
显式输入严格重复输出及闭环 gate 共同证明。
