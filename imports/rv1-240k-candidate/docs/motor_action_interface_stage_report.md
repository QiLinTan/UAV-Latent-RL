# 教师生命周期与直接电机动作接口阶段诊断报告

## 阶段结论

```text
motor_action_interface_gate_passed = true
```

在本阶段规定的仿真范围内，T3 非对称 RPM 接口和 T4 单电机推力接口均通过准入门槛。
两种接口都保留了原始 DSLPIDControl 的闭环恢复行为、总推力和三轴力矩权限。

本阶段没有训练 Actor，没有执行 TD3、DAgger、森林导航、latent 上层或奖励函数实验。

## 1. 教师 reset 修复位置和测试结果

### 源码已经确认的事实

- `DSLPIDControl.reset()` 会清零：
  - `integral_pos_e`
  - `integral_rpy_e`
  - `last_rpy`
  - `last_rpy_e`
  - `control_counter`
- `TD3ReferenceTracking.reset_episode()` 现在会在教师已创建时调用
  `self._teacher_controller.reset()`。
- 训练器在环境初始 reset 和每个回合结束后的环境 reset 之后调用
  `agent.reset_episode()`。
- 独立验证脚本也在每个回合开始时调用 `agent.reset_episode()`。

### 自动化测试直接支持的结论

- 连续回合之间，教师积分状态、上一姿态和控制计数器均恢复为零。
- 相同状态、相同参考和相同初始教师状态产生一致的首个教师动作。
- 完整测试集 32 项全部通过。

## 2. 旧动作接口的数学定义

旧接口首先使用固定比例将 RPM 编码为动作：

```text
a_i = (RPM_i / HOVER_RPM - 1) / 0.05
```

随后执行结构投影：

```text
c = clip(mean(a), -0.45, 0.45)
d_i = a_i - mean(a)
若 max(|d_i|) > 0.1875，则按比例将 max(|d_i|) 缩小到 0.1875
a'_i = clip(c + d_i, -0.75, 0.75)
```

最后仍按固定比例解码：

```text
RPM'_i = HOVER_RPM × (1 + 0.05 × a'_i)
```

该接口同时混合了动作编码、总体动作范围和 collective/differential 人为结构限制。
现将它保留为 `legacy_projected`，仅用于兼容和 T2 负对照。

## 3. 新非对称 RPM 接口的数学定义

设物理下限、悬停转速和物理上限分别为
`RPM_min`、`RPM_hover`、`RPM_max`。

```text
RPM_i >= RPM_hover:
    a_i = (RPM_i - RPM_hover) / (RPM_max - RPM_hover)

RPM_i < RPM_hover:
    a_i = (RPM_i - RPM_hover) / (RPM_hover - RPM_min)
```

解码为：

```text
a_i >= 0:
    RPM_i = RPM_hover + a_i × (RPM_max - RPM_hover)

a_i < 0:
    RPM_i = RPM_hover + a_i × (RPM_hover - RPM_min)
```

该映射满足：

```text
RPM_min   ↔ -1
RPM_hover ↔  0
RPM_max   ↔ +1
```

当前仿真诊断使用 `RPM_min=0` 和环境根据推重比计算的 `RPM_max`。

## 4. 新单电机推力接口的数学定义

先将每个电机 RPM 转换为推力：

```text
F_i = KF × RPM_i²
```

设 `F_min`、`F_hover`、`F_max` 分别由三个 RPM 边界计算得到。

```text
F_i >= F_hover:
    a_i = (F_i - F_hover) / (F_max - F_hover)

F_i < F_hover:
    a_i = (F_i - F_hover) / (F_hover - F_min)
```

解码后通过下式还原 RPM：

```text
RPM_i = sqrt(F_i / KF)
```

该接口仍直接输出四个单电机控制量，不引入 PID 内环。

## 5. 编码—解码误差统计

同状态离线分析使用 T1 轨迹上的 164,388 组四电机教师动作。所有原始教师动作均未触及
当前仿真物理上下限。

| 接口 | 动作被修改比例 | 未饱和 RPM 平均绝对误差 | 未饱和 RPM 最大绝对误差 |
|---|---:|---:|---:|
| T1 原始 RPM | 0% | 0 | 0 |
| T2 旧投影 | 4.392% | 3.694 RPM | 5384.886 RPM |
| T3 非对称 RPM | 0% | 0 | 0 |
| T4 单电机推力 | 0% | 3.57×10⁻¹³ RPM | 1.82×10⁻¹² RPM |

T2 的 95% RPM 误差接近零，但少量恢复阶段的峰值动作被大幅修改。这说明只看平均 MSE
或高分位以下的误差会掩盖关键恢复动作丢失。

## 6. 总推力和三轴力矩保持情况

四电机 RPM 按 CF2X 电机顺序和仿真动力学转换为总推力、roll、pitch、yaw 力矩。

| 接口 | 总推力相对平均绝对误差 | roll 力矩增益 | pitch 力矩增益 | yaw 力矩增益 |
|---|---:|---:|---:|---:|
| T1 原始 RPM | 0 | 1.0000 | 1.0000 | 1.0000 |
| T2 旧投影 | 0.0308% | 0.3487 | 0.3528 | 0.1720 |
| T3 非对称 RPM | 0 | 1.0000 | 1.0000 | 1.0000 |
| T4 单电机推力 | 2.41×10⁻¹⁵% | 1.0000 | 1.0000 | 1.0000 |

实验直接支持以下结论：

- T2 基本保留了总推力，却系统性削弱了三轴力矩。
- T2 的恢复阶段 roll 力矩整体增益为 0.3705，即约削弱 62.95%。
- T2 的恢复阶段 roll 峰值力矩只保留 31.14%。
- T2 没有系统性改变力矩方向，主要问题是力矩幅值衰减。
- T3、T4 保留了原始教师的总推力和三轴力矩。

## 7. T1～T4 扰动恢复结果

### 实验设置

- 控制频率：120 Hz。
- 单回合时长：12 s。
- 随机种子：0、1、2。
- 工况数：38。
- 总回合数：456。
- 每个接口：114 回合。
- 使用相同环境、ReferencePacket、初始条件、扰动和种子。

工况覆盖：

- roll、pitch、yaw 各自的 ±0.05、±0.10、±0.20 rad；
- 六组多轴组合姿态扰动；
- 已知失败回归工况 `[0.12,-0.10,0.08] rad`；
- 三轴随机初始角速度；
- 四个方向的初始水平速度；
- 高度偏差 ±0.1、±0.2 m；
- 运行中线速度和角速度冲击；
- 固定悬停、直线和缓弯参考。

恢复判据为：位置误差不高于 0.05 m、最大姿态角不高于 0.05 rad、速度误差不高于
0.15 m/s、角速度不高于 0.15 rad/s，并连续保持 30 个控制步。

### 汇总

| 接口 | 完整回合率 | 失稳率 | 姿态越界率 | 平面越界率 | 是否全部恢复 |
|---|---:|---:|---:|---:|---:|
| T1 原始 RPM | 100% | 0% | 0% | 0% | 是 |
| T2 旧投影 | 86.84% | 13.16% | 10.53% | 2.63% | 否 |
| T3 非对称 RPM | 100% | 0% | 0% | 0% | 是 |
| T4 单电机推力 | 100% | 0% | 0% | 0% | 是 |

T2 的 15 个失败回合包括：

- 12 个多轴初始姿态扰动导致的姿态越界；
- 3 个运行中冲击导致的平面越界。

在 `[0.12,-0.10,0.08] rad` 回归工况中：

- T1、T3、T4 均运行满 1442 步，恢复时间为 0.758 s；
- T2 在第 183 步姿态越界，未恢复。

T3 和 T4 的终止原因、回合步数、恢复时间及闭环误差逐工况与 T1 一致。

## 8. 推荐进入下一阶段的接口

T3 和 T4 均满足教师闭环准入要求。当前实验没有证明哪一种对神经网络学习更优。

建议第一轮行为克隆优先使用 T3 非对称 RPM 接口，理由是：

- 它已通过全部物理量和闭环门槛；
- 数学及软件链路更短；
- 与现有 RPM 电机模型和教师输出直接对应；
- 便于将后续 Actor 问题与额外的平方根变换解耦。

T4 作为保留候选和后续受控对照。推力空间是否更利于 MLP 学习目前尚未验证，不能作为
本阶段的实验结论。

新接口路径默认使用已有的普通四输出 MLP，不再默认使用带 collective/differential
硬范围的残差 Actor。残差 Actor 和结构化 Actor 仍保留为旧接口对照，但本阶段没有训练
任何一种 Actor。

## 9. 仍未解决的技术风险

### 尚未验证

- 新接口尚未用于重新采集行为克隆数据，也没有训练或闭环验证 MLP Actor。
- 未验证真实电机的最小稳定转速、最大安全转速和 RPM 变化率；当前 `RPM_min=0`
  是仿真物理边界，不是实机标定结果。
- 未加入电机延迟、传感器噪声、动力学随机化、通信延迟和实机执行器误差。
- 当前教师动作没有触及仿真物理饱和，因此闭环饱和恢复行为仍未覆盖。
- T3 与 T4 对网络优化条件、动作误差到力矩误差的敏感度差异尚未通过学习实验验证。
- 旧行为克隆数据尚未删除或隔离。

### 仅作为后续候选

- 真实物理 RPM 变化率限制；
- 单电机推力接口作为 MLP 对照；
- 显式积分、长历史和 GRU；
- 小规模 TD3 稳定性微调。

这些项目均不属于本阶段已经验证的结论。

## 10. 行为克隆数据重采集准入判断

```text
motor_action_interface_gate_passed = true
recommended_interface = asymmetric_rpm
behavior_cloning_data_recollection_allowed = true
actor_training_completed = false
```

可以进入“隔离旧数据并使用正确 reset 的教师重新采集扰动恢复数据”阶段，但本报告
不代表 Actor、行为克隆、强化学习或完整分层架构已经通过。

## 结果文件

- 完整 T1～T4 结果：
  `runs/motor_action_interface_diagnostics/t1_t4_full_seeds_0_1_2.json`
- T3 教师悬停集成验证：
  `runs/motor_action_interface_diagnostics/teacher_hover_t3.json`
- T4 教师悬停集成验证：
  `runs/motor_action_interface_diagnostics/teacher_hover_t4.json`
