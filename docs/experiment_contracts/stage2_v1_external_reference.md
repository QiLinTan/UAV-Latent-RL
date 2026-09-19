# Stage 2 v1 External Reference Controller Contract

日期：2026-09-12
状态：接口和验证框架冻结；尚未采集 Stage2-v1 数据，也未训练或验收 Actor。

## 固定链路

```text
TrapezoidalWorldTrajectory / ConstantWorldTrajectory
  -> ContinuousWorldPacketSource (15 Hz, 1 s, 16 points)
  -> models.reference_packet.ReferencePacket (world)
  -> AsyncReferenceBuffer
  -> TD3ReferenceTracking(reference_source="external_packet", 120 Hz)
  -> MotorActionCodec(asymmetric_rpm)
  -> CustomForestAviary / PyBullet
```

Stage 1 和 Stage 2 共用 `models/continuous_world_reference.py` 的轨迹与 Packet
source。两条正式链路只允许下层控制器不同。Stage2-v1 不使用 legacy generator、
Upper planner、latent、森林障碍、`TD3HierarchicalAsync` 或
`models.hierarchical_async.AsyncReferenceCache`。

控制周期 `k` 的仿真时间为 `sim_time = k / 120`，实际执行目标的采样时间统一为
`control_time = sim_time + 1/120`。current reference 在 `control_time` 采样；future
reference 在 `control_time + 0.2 s` 采样，两者不得互换。Packet 每 8 个低层周期
发布一次。

Packet 缺失或在 `control_time` 过期时不调用 Actor：输出上一次已执行归一化电机
动作的 `0.85` 倍。由于 asymmetric-RPM 的零点是 hover RPM，这一策略把动作平滑
衰减到悬停等效值，并记录 fallback。

## 30D Context

合同标识：`stage2_v1_external_packet_30d`。所有向量分量均按 world 轴排列。

| 维度 | 字段 | 物理定义 | 输入归一化 |
| ---: | --- | --- | --- |
| 0 | `position_x` | 当前 world `x`，m | `(x - 0) / 0.5` |
| 1 | `position_y` | 当前 world `y`，m | `(y - 0) / 0.5` |
| 2 | `position_z` | 当前 world `z`，m | `(z - 1) / 0.5` |
| 3 | `roll` | 当前 roll，rad | `roll / pi` |
| 4 | `pitch` | 当前 pitch，rad | `pitch / pi` |
| 5 | `yaw` | 当前 yaw，rad | `yaw / pi` |
| 6 | `velocity_x` | 当前 world `vx`，m/s | `vx / 2` |
| 7 | `velocity_y` | 当前 world `vy`，m/s | `vy / 2` |
| 8 | `velocity_z` | 当前 world `vz`，m/s | `vz / 2` |
| 9 | `angular_velocity_x` | 当前 `wx`，rad/s | `wx / 4` |
| 10 | `angular_velocity_y` | 当前 `wy`，rad/s | `wy / 4` |
| 11 | `angular_velocity_z` | 当前 `wz`，rad/s | `wz / 4` |
| 12 | `current_position_error_x` | `p_ref,x(control_time) - p_x`，m | `clip(value / 0.5, -4, 4)` |
| 13 | `current_position_error_y` | `p_ref,y(control_time) - p_y`，m | 同上 |
| 14 | `current_position_error_z` | `p_ref,z(control_time) - p_z`，m | 同上 |
| 15 | `current_velocity_error_x` | `v_ref,x(control_time) - v_x`，m/s | `clip(value / 2, -4, 4)` |
| 16 | `current_velocity_error_y` | `v_ref,y(control_time) - v_y`，m/s | 同上 |
| 17 | `current_velocity_error_z` | `v_ref,z(control_time) - v_z`，m/s | 同上 |
| 18 | `future_position_error_x` | `p_ref,x(control_time + 0.2) - p_x`，m | `clip(value / 0.5, -4, 4)` |
| 19 | `future_position_error_y` | `p_ref,y(control_time + 0.2) - p_y`，m | 同上 |
| 20 | `future_position_error_z` | `p_ref,z(control_time + 0.2) - p_z`，m | 同上 |
| 21 | `future_velocity_error_x` | `v_ref,x(control_time + 0.2) - v_x`，m/s | `clip(value / 2, -4, 4)` |
| 22 | `future_velocity_error_y` | `v_ref,y(control_time + 0.2) - v_y`，m/s | 同上 |
| 23 | `future_velocity_error_z` | `v_ref,z(control_time + 0.2) - v_z`，m/s | 同上 |
| 24 | `reference_age_ratio` | `(control_time - packet.t_gen) / valid_duration` | `clip(value, 0, 1)` |
| 25 | `reference_valid` | Packet 在 `control_time` 是否有效 | false=0，true=1 |
| 26 | `previous_motor_action_0` | 电机 0 上一实际执行 asymmetric-RPM action | 已归一化到 `[-1, 1]` |
| 27 | `previous_motor_action_1` | 电机 1 上一实际执行 asymmetric-RPM action | 同上 |
| 28 | `previous_motor_action_2` | 电机 2 上一实际执行 asymmetric-RPM action | 同上 |
| 29 | `previous_motor_action_3` | 电机 3 上一实际执行 asymmetric-RPM action | 同上 |

没有多步动作历史。Actor 输入为 30D、输出为 4D；Actor/Critic 网络结构没有在本轮
改变。正式 checkpoint 必须声明相同的 `context_contract_id`、
`motor_action_codec="asymmetric_rpm"` 和 `multi_step_action_history=0`。

## 第一版验收范围

`scripts/validate_stage2_reference_tracking.py` 只执行已有 Actor checkpoint，不创建
教师、不调用教师动作、不更新网络。任务为固定起点 hover 与和 Stage 1 完全相同的
7 m world-frame 直线。每个 episode 必须满足 `teacher_intervention_count=0` 和
`actor_control_ratio=1.0`，并输出跟踪、姿态、高度、动作平滑度、饱和率与 Packet
健康度指标。

当前仓库没有符合本合同标识的 checkpoint，因此 Stage2-v1 gate 仍是未运行/未通过，
不能由接口单测或历史 BC 结果替代。
