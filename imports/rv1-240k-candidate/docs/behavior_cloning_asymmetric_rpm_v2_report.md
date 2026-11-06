# asymmetric_rpm_v2 行为克隆阶段诊断报告

生成日期：2026-07-29

## 1. 阶段状态

```text
behavior_cloning_dataset_recollected = true
dataset_version = asymmetric_rpm_v2
actor_bc_training_completed = true
actor_hover_gate_passed = true
actor_disturbance_recovery_gate_passed = false
td3_training_allowed = false

motor_action_interface_gate_passed = true
recommended_interface = asymmetric_rpm
lower_motor_control_path_connected = true
actor_dynamic_recovery_observed = true
formal_imitation_fidelity_gate_passed = false
```

这里必须区分两个结论：

1. 下层控制链已经打通。普通 MLP 能独立输出四电机动作，经 T3 非对称
   RPM 接口直接驱动仿真电机；教师没有参与接管。首版 Actor 在全部正式
   工况中均完成回合，并具备实际悬停和扰动恢复行为。
2. 行为克隆的正式准入门槛尚未全部通过。Actor 在自身访问状态上的关键
   三轴力矩方向与并行教师不够一致，因此当前不允许进入 TD3。

## 2. 已确认的源码事实

- 教师为 `DSLPIDControl`，每个环境回合均调用 `reset()`。
- 教师标签路径为：

  ```text
  当前状态与 ReferencePacket
  → reset 后的 DSLPIDControl
  → 原始教师 RPM
  → asymmetric_rpm 编码
  → encoded_teacher_action
  → 直接施加给环境
  ```

- 教师数据采集未加入电机动作探索噪声。
- `encoded_teacher_action` 与 `applied_action` 逐元素完全相等。
- T3 未触及物理极限时可逆；本次数据集的 float32 RPM 往返最大误差为
  `0.000212 RPM`，物理饱和电机比例为 `0`。
- Actor 是普通前馈 MLP：

  ```text
  30-D context → 256 → 256 → 4 → tanh
  ```

- 30 维输入依次为：

  ```text
  归一化运动状态[12]
  + 当前位置/速度误差[6]
  + 前视位置/速度误差[6]
  + reference age/valid[2]
  + 上一实际动作[4]
  ```

- 当前 Actor 输入不包含教师的 `last_rpy`、`integral_pos_e`、
  `integral_rpy_e` 或 `control_counter`，也不包含多步历史。
- `DSLPIDControl.computeControl()` 虽接收 `cur_ang_vel`，但其姿态 D 项
  实际使用 `(cur_rpy - last_rpy) / dt`；`cur_ang_vel` 未参与教师输出。

## 3. 旧数据隔离和新数据版本

旧投影动作数据未删除，已隔离并标记为：

```text
data/behavior_cloning/legacy_projected_v1/
```

新主数据位于：

```text
data/behavior_cloning/asymmetric_rpm_v2/
```

训练加载器采用失败关闭策略，显式检查：

```python
assert dataset.dataset_version == "asymmetric_rpm_v2"
assert dataset.motor_action_codec == "asymmetric_rpm"
assert dataset.teacher_reset_enabled is True
```

数据摘要：

| 项目 | 数量 |
|---|---:|
| 回合总数 | 250 |
| 样本总数 | 360,500 |
| 标称状态样本 | 50,470 |
| 初始恢复工况样本 | 245,140 |
| 运行中冲击工况样本 | 64,890 |
| train | 135 回合 / 194,670 样本 |
| validation | 45 回合 / 64,890 样本 |
| unseen seed | 45 回合 / 64,890 样本 |
| unseen condition | 25 回合 / 36,050 样本 |

训练、验证和测试按完整 `episode_id`、随机种子和扰动工况划分，没有将同一
恢复轨迹的相邻控制步随机拆到不同集合。

数据 SHA-256：

```text
71a2f5b4e97becfd0f29a96cf3c7a7baf1497d0444c792cf2cb5f823f6c99822
```

## 4. 首版普通 MLP 的离线结果

训练 80 epoch，最佳检查点为第 80 epoch。未运行 TD3。

### 4.1 验证集

| 样本组 | 动作 RMSE | RPM RMSE | 总推力 MAE | roll/pitch/yaw 力矩 MAE |
|---|---:|---:|---:|---:|
| overall | 0.00224 | 21.61 | 1.61e-4 N | 2.44e-6 / 2.32e-6 / 2.75e-6 Nm |
| nominal | 0.00108 | 12.19 | 1.45e-4 N | 2.93e-6 / 3.02e-6 / 3.14e-6 Nm |
| initial recovery | 0.00740 | 72.13 | 3.84e-4 N | 9.94e-6 / 1.08e-5 / 1.28e-5 Nm |
| impulse recovery | 0.00907 | 73.31 | 3.68e-4 N | 1.58e-5 / 1.48e-5 / 1.55e-5 Nm |

教师轨迹验证集上的力矩方向一致率：

| 样本组 | roll | pitch | yaw |
|---|---:|---:|---:|
| initial recovery | 97.6% | 97.4% | 97.5% |
| impulse recovery | 89.5% | 92.3% | 92.7% |

### 4.2 未见随机种子和未见扰动组合

- 未见随机种子 overall 动作 RMSE 为 `0.00216`，RPM RMSE 为
  `22.05 RPM`。
- 未见扰动组合 overall 动作 RMSE 为 `0.00969`，RPM RMSE 为
  `73.38 RPM`。
- 未见组合中的冲击恢复力矩方向一致率下降到
  `84.9% / 82.3% / 62.3%`，其中 yaw 是最弱通道。

离线结果说明恢复状态比总体样本明显更难，且未见组合上的 yaw 泛化最弱；
但首版 Actor 在教师访问状态上的恢复动作并非完全没有学到。

## 5. 首版 Actor 独立闭环结果

闭环中 Actor 完全接管动作，教师仅在相同状态上并行计算诊断标签。

### 5.1 开发门槛

- 5 个不同随机种子的 12 秒固定悬停：`5/5` 完成。
- roll、pitch、yaw 的 `±0.05 rad` 初始偏差：
  30 个回合全部完成并恢复。
- 姿态、平面和高度越界：`0`。
- 开发门槛：通过。

### 5.2 正式闭环

正式集合共 28 回合，包括：

- 5 个 30 秒固定悬停；
- 单轴和组合姿态偏差；
- 初始线速度、角速度和高度偏差；
- 运行中线速度、角速度和组合冲击；
- 高度阶跃、低速直线、缓弯；
- 未见组合姿态、对角速度、组合冲击和反向圆弧。

| 指标 | 首版 Actor | 原始教师 |
|---|---:|---:|
| 完整回合 | 28/28 | 28/28 |
| 姿态/平面/高度越界 | 0/0/0 | 0/0/0 |
| 扰动恢复成功率 | 100% | 100% |
| 平均恢复时间 | 0.948 s | 0.932 s |
| 最长恢复时间 | 3.908 s | 3.700 s |
| 最大 roll/pitch/yaw | 0.299 / 0.289 / 0.115 rad | 0.316 / 0.287 / 0.127 rad |
| 最大角速度 | 2.978 rad/s | 3.036 rad/s |
| 最大位置误差 | 0.200 m | 0.200 m |
| 最差末段稳态位置误差 | 0.00533 m | 0.00379 m |

该实验直接支持：普通 MLP 已具备基础电机级悬停和恢复能力，下层动作链路
不是“只验证了接口而没有闭环飞行”。

### 5.3 未通过项

在 Actor 实际访问状态、恢复有效区间和关键教师力矩样本上，三轴方向一致率为：

```text
roll  = 44.5%
pitch = 47.0%
yaw   = 63.3%
```

准入要求为三个通道均不低于 90%，因此：

```text
actor_disturbance_recovery_gate_passed = false
```

首个显著动作差异多数发生在回合开始或扰动发生后，主要表现为 yaw 力矩
幅值或方向与教师不同。它没有导致首版 Actor 失稳，但说明“Actor 独立稳定”
与“逐步复现带隐藏状态的 PID 动作”并不是同一个判据。

## 6. DAgger 式诊断迭代

由于教师轨迹离线拟合较好、Actor 访问状态上动作一致性明显下降，按情况 B
执行了一轮 DAgger 式数据聚合：

- Actor 执行动作；
- DSLPID 在 Actor 实际访问状态上重新标注；
- 教师不接管；
- 训练种子为 5/6/7，验证种子为 8；
- 5 个未见扰动组合未进入聚合训练。

聚合数据：

| 项目 | 数量 |
|---|---:|
| 回合 | 180 |
| 样本 | 259,560 |
| train | 135 回合 / 194,670 样本 |
| validation | 45 回合 / 64,890 样本 |
| recovery active | 16,568 样本 |
| recovery early | 13,978 样本 |

DAgger 使 Actor 访问状态验证集的恢复动作 RMSE 从 `0.02375` 降到
约 `0.01411`，但闭环结果变差：

- 28/28 回合仍跑满且未越界；
- 稳态角速度出现约 `0.30 rad/s` 的高频振荡；
- 小姿态扰动恢复判据从 100% 降为 33.3%；
- 正式扰动恢复判据降为 50%；
- 力矩方向一致率仅提高到约 70%；
- DAgger 检查点不应替代首版 Actor。

这说明单纯聚合更多同类标签不能解决当前矛盾。

## 7. 教师隐藏状态反例

进行了一个受控实验：保持传给 `computeControl()` 的当前位置、姿态、速度、
角速度、目标位置、目标速度和时间步完全相同，只改变教师内部历史状态。

结果：

```text
相同可见输入下，单电机标签最大范围 = 0.1493
相同可见输入下，单电机 RPM 最大范围 = 2159.8 RPM
pitch 和 yaw 力矩方向发生翻转
```

因此已获得情况 C 的直接证据：

> 当前 30 维 Actor 输入不能唯一决定 DSLPID 教师动作。教师的
> `last_rpy`、位置积分和 yaw 姿态积分是标签生成函数的一部分，但未提供给
> Actor。

这也解释了为什么 DAgger 可以降低固定访问数据上的 MSE，却在策略改变后
形成新的闭环振荡。

## 8. 结论分类

### 源码已经确认的事实

- T3 动作接口可逆且已通过教师扰动恢复准入。
- 新数据使用正确 reset 的教师，旧数据被隔离。
- 普通 30→256→256→4 MLP 已完成训练。
- DSLPID 的输出依赖未包含在 Actor 输入中的 `last_rpy` 和积分状态。
- DSLPID 未使用传入的 `cur_ang_vel`，而使用姿态差分计算 D 项。

### 实验直接支持的结论

- 首版 Actor 可独立完成悬停、跟踪和所有已测扰动恢复工况。
- 首版 Actor 的在线关键力矩方向没有达到教师一致性门槛。
- 一轮 DAgger 降低了在线数据 MSE，但引入稳态高频角速度振荡。
- 同一可见输入对应不同教师隐藏状态时，教师电机标签可相差 0.1493，
  并可发生力矩方向翻转。

### 尚未验证的推测

- 仅加入显式 `last_rpy` 和必要积分状态是否足以同时通过离线与闭环门槛。
- 将教师姿态 D 项改为使用真实 `cur_ang_vel` 后，是否可消除主要标签歧义。
- 无积分、可观测的几何教师是否比当前 DSLPID 更适合生成 BC 标签。

### 后续候选，不属于本阶段已完成工作

按信息增益排序：

1. 先做显式教师状态对照：加入 `last_rpy`、`integral_pos_e` 和 yaw
   `integral_rpy_e`，继续使用同一个普通 MLP。
2. 做可观测教师对照：教师 D 项直接使用 `cur_ang_vel`，并重新通过 T3
   教师扰动门槛后再采集数据。
3. 只有前两项仍不足时，比较短历史与 GRU；不应直接默认 GRU。
4. 只有新的 Actor 同时通过悬停和扰动恢复正式门槛后，才允许小规模 TD3。

## 9. 最终工程判断

```text
下层网络动作链：已打通
基础闭环悬停：已证明
基础闭环扰动恢复：已观察到
严格教师模仿一致性：未通过
继续 DAgger：当前不建议
使用 DAgger 检查点：不建议
使用首版 MLP 检查点作为后续诊断基线：建议
进入 TD3：禁止
```

推荐保留的基线检查点：

```text
checkpoints/behavior_cloning/asymmetric_rpm_v2_plain_mlp_b4096/actor_best.pt
```

DAgger 检查点仅用于失败诊断：

```text
checkpoints/behavior_cloning/asymmetric_rpm_v2_plain_mlp_dagger1_b4096/actor_best.pt
```
