# Stage 1 Continuous World Reference v1

日期：2026-09-11

## 目的

验证冻结架构中的第一条正式控制链路：

```text
固定 7 m 全局连续轨迹
  -> 15 Hz 滑动未来窗口
  -> models.reference_packet.ReferencePacket
  -> AsyncReferenceBuffer
  -> 120 Hz 仿真时钟插值
  -> DSLPIDControl
  -> MotorActionCodec(asymmetric_rpm)
  -> CustomForestAviary / PyBullet
```

本实验不导入 RL agent、planner、latent 或
`models.hierarchical_async.AsyncReferenceCache`，也不修改历史
`TD3ReferenceTracking._refresh_reference()` 的局部 minimum-jerk 行为。

## 固定配置

| 参数 | 值 |
| --- | ---: |
| 起点 | `[-3.5, 0.0, 1.0] m` |
| 终点 | `[3.5, 0.0, 1.0] m` |
| 路程 | `7.0 m` |
| 最大速度 | `0.6 m/s` |
| 最大加速度 | `1.5 m/s^2` |
| 上层窗口频率 | `15 Hz` |
| 下层控制频率 | `120 Hz` |
| 未来窗口 | `1.0 s`, 16 点 |
| settle time | `2.0 s` |
| 目标容差 | `0.20 m` |
| 姿态限制 | `0.65 rad` |
| 障碍物 | 0 |

轨迹采用加速—匀速—减速梯形速度曲线，全局只生成一次。每个 Packet 从同一
全局轨迹的当前仿真时间截取未来 1 s，不从无人机当前位置重新规划，也不重置
参考速度或轨迹进度。

物理匀速下界为 `7 / 0.6 = 11.6667 s`；包含加减速后的实际轨迹时长为
`12.0667 s`。timeout 由 `trajectory_duration + settle_time` 得到，为
`14.0667 s`。执行器启动时对上述可达关系做硬断言。

## Gate

| 指标 | 门槛 |
| --- | ---: |
| success rate | 100% |
| 平均最终距离 | <= 0.20 m |
| 平均 position RMSE | <= 0.10 m |
| 平均 velocity tracking error | <= 0.15 m/s |
| 最大 roll/pitch | <= 0.65 rad |
| 姿态越界周期 | 0 |
| 平均 height RMSE | <= 0.05 m |
| 电机饱和比例 | <= 5% |
| missing/stale/fallback | 0 |
| Packet 版本 | 严格连续 |
| timeout 可达断言 | true |

## 命令

```bash
/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.validate_stage1_reference_pid \
  --seeds 0 1 2 \
  --output analysis_outputs/stage1_reference_pid/straight_line_results.json
```

## 结果

| Seed | Success | Final distance | Position RMSE | Velocity error | Max roll | Max pitch | Saturation | Fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.1993 m | 0.0146 m | 0.0535 m/s | 0.0000 rad | 0.2248 rad | 0% | 0 |
| 1 | 1 | 0.1993 m | 0.0146 m | 0.0535 m/s | 0.0000 rad | 0.2248 rad | 0% | 0 |
| 2 | 1 | 0.1993 m | 0.0146 m | 0.0535 m/s | 0.0000 rad | 0.2248 rad | 0% | 0 |

汇总：

- success rate：100%；
- height absolute error：`0.000065 m`；
- height RMSE：`0.000213 m`；
- attitude violation count：0；
- Packet age mean / p95 / max：`0.0375 / 0.0667 / 0.0667 s`；
- missing / stale / fallback：`0 / 0 / 0`；
- 每回合接受 173 个包，版本 `1..173` 严格连续；
- 所有 gate checks 通过。

三个 seed 结果一致符合预期：空环境、固定初始状态、固定轨迹和确定性 PyBullet
配置没有引入随机扰动。多 seed 在此用于验证 reset/版本状态没有跨回合泄漏，
不用于声称随机鲁棒性。

## 结论

`stage1_straight_line_gate_passed = true`。

Stage 1 的 7 m 连续 world-frame 直线链路已经成立。该结论只覆盖标称空环境
直线；不覆盖异步传输延迟、丢包、扰动恢复、RL 下层或避障。
