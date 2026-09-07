# Latent Reference Sequence Planner 阶段报告

日期：2026-09-04

## 实现结论

在冻结 `latent_only_affordance_dim16` Encoder、旧 Actor、DSLPID、环境、奖励和
异步缓存的前提下，新增了以下链路：

```text
263D observation
  -> frozen Encoder -> 16D latent
  -> [latent, relative goal, 8 range rays] (27D)
  -> ReferenceSequencePlanner (27 -> 256 -> 256 -> 30)
  -> 5 x [relative position, world velocity]
  -> fixed [0.1, 0.2, 0.3, 0.4, 0.5] s timestamps
  -> t=0 continuous anchor + five future points
  -> ReferencePacket -> AsyncBuffer -> DSLPID (120 Hz)
```

时间戳不作为网络回归目标。这样避免非单调或越界时间输出，同时保留需求中的
五个未来预测点。`ReferencePacket` 内部额外插入 `t=0` 锚点，保证新旧包切换时
位置和速度连续。

## 教师与数据

冻结 direct-motor Actor 在当前代码状态下，无障碍探测 10/10 以 `xy_bound`
结束，不能充当成功轨迹教师。因此保留其作为实验 baseline，监督数据改由障碍
膨胀 A* 产生路径、DSLPID 执行路径。Encoder 仍在每个教师状态上冻结推理，得到
Planner 的 16D latent 输入。

正式 v2 数据集：

- 30 个成功 episode；
- 4,252 个 15 Hz 样本；
- 无障碍、单障碍、24 树各 10 个成功 episode；
- 只保存成功轨迹；
- train/validation/test 按 episode 分离；
- 位置标签是相对当前位置的未来位移，速度标签使用世界坐标系。

## 模仿训练

仅 Planner MLP 参与优化。100 epoch 最佳结果：

| 指标 | 结果 |
| --- | ---: |
| best epoch | 100 |
| validation loss | 0.4291 |
| test position RMSE | 0.0146 m |
| test velocity RMSE | 0.0813 m/s |

训练期间 Encoder、旧 Actor、PID 和异步模块均未进入 optimizer。

## 闭环逻辑异步评估

配置：高层 15 Hz、低层 120 Hz、上层计算延迟均匀分布于 0--50 ms，每项 5 个
未见 seed。两种控制器均通过同一个 `ReferencePacket -> AsyncBuffer -> DSLPID`
路径执行。

| 场景 | 控制器 | 成功率 | 碰撞率 | 最终距离 | 平均最小净空 | 路径长度 | 跟踪 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无障碍 | Actor adapter | 100% | 0% | 0.197 m | N/A | 6.803 m | 0.010 m |
| 无障碍 | Reference Planner | 100% | 0% | 0.198 m | N/A | 7.598 m | 0.021 m |
| 单障碍 | Actor adapter | 0% | 100% | 3.721 m | 0.001 m | 3.279 m | 0.014 m |
| 单障碍 | Reference Planner | 100% | 0% | 0.198 m | 0.638 m | 7.479 m | 0.019 m |
| 24 树 | Actor adapter | 0% | 100% | 4.439 m | 0.001 m | 2.561 m | 0.016 m |
| 24 树 | Reference Planner | 0% | 100% | 4.960 m | 0.001 m | 2.774 m | 0.025 m |

所有评估项的低层中断数、stale ratio 和 reference switching error 都为 0；平均
packet age 约 58 ms。

审计澄清（2026-09）：上述调度在同一 Python 执行流中使用仿真时钟模拟计算
延迟，并未实现独立 Worker，也未测量真实 8.33 ms deadline。`low_level_interruptions`
为逻辑计数，不能证明实际 120 Hz 实时性。启动期无包周期应另见原始 JSON 中
`missing_packet_cycles`，不能用 stale ratio=0 代替无包统计。

## 判断

第一阶段证明了 16D latent 与局部 goal/range 信息能够通过独立 Planner 生成可由
PID 稳定执行的未来参考序列，并在单障碍任务中表现出明确绕障能力。24 树尚未
通过，因此当前结果不能声明已解决复杂森林导航。

单帧局部观测与全局 A* 标签之间的信息差异、闭环分布偏移、参考过滤带来的
轨迹变化均为待检验假设。较小的平均 PID 跟踪误差只能说明多数采样点跟踪较好，
不能单独排除碰撞附近的执行问题。此前将信息不足直接定为主因的表述证据不足。
该实验未做地图级隔离、latent 消融或运行时 deadline 测量，不能证明 latent
改善森林泛化或系统满足实时性。下一轮应先完成原生 Actor 接口审计和跨地图诊断。
