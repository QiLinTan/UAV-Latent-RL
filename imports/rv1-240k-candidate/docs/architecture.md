# UAV-Async-DRL 架构冻结

日期：2026-09-11
状态：架构契约；本文件不改变任何训练实现或既有实验结论。

## 1. 系统目标与当前边界

目标系统是一个分层、异步的无人机导航系统：上层以低频产生**未来位置/速度
参考序列**，下层以高频执行该序列。上、下层唯一的控制边界是
`ReferencePacket`；下层不读取上层网络、latent、地图规划器或其内部状态。

```text
上层（15 Hz，允许慢、允许延迟）
  fixed trajectory / learned planner / Upper DRL
                 |
                 | ReferencePacket（最新包覆盖旧包）
                 v
异步传输、版本检查、有效期、插值
                 |
                 v
下层（120 Hz，绝不等待上层）
  DSLPID 或 RL reference-tracking controller
                 |
                 v
MotorActionCodec -> normalized asymmetric RPM -> PyBullet plant
```

运行时高层可位于独立进程。低层每个控制周期只进行非阻塞读取：有有效包则插值
执行；无包或过期则进入定义明确的悬停/安全降级策略。

## 2. 三阶段实验路线

### Stage 1：接口与经典控制闭环

```text
固定、连续的世界系参考轨迹
  -> ReferencePacket -> buffer/interpolation -> DSLPID -> motor codec -> plant
```

目的不是训练，也不是避障，而是证明包的坐标、时钟、插值、切换、过期和 PID
闭环正确。验收指标为目标成功率、终点距离、跟踪 RMSE、姿态/高度越界率、电机
饱和率、包年龄和 fallback 周期数。

固定轨迹必须是连续的：不得在每次高层刷新时从零速度重新启动 minimum-jerk
曲线。对于 7 m 路线，试验时长必须与速度/加速度上限相容；短于物理可达时间的
超时不能被解释为控制失败。

### Stage 2：学习型下层控制

```text
Stage 1 已冻结的固定轨迹和 Packet 合同
  -> RL reference-tracking controller -> motor codec -> plant
```

先后顺序为固定悬停、直线、转弯、再通信故障。Stage 2 的 RL 下层消费与 DSLPID
完全相同的 `ReferencePacket` 样本；不得将旧的直接四电机导航 Actor 当作已验证的
reference-tracking lower controller。只有 RL 下层达到与 Stage 1 相同的稳定性门槛，
才可进入 Stage 3。

### Stage 3：学习型上层 + 已验证下层

```text
Upper DRL / ReferenceSequencePlanner
  -> future ReferencePacket sequence -> runtime async interface
  -> frozen, validated lower controller
```

Stage 3 首先只替换包的生产者；下层、动作编码、环境和 Packet 合同保持不变。
`TD3HierarchicalAsync` 的内部 step-cache + 直接四电机 Actor 属于历史端到端支线，
不作为这一阶段的正式集成路径。

## 3. 模块职责

| 模块 | 职责 | 阶段定位 |
| --- | --- | --- |
| `envs/forest/*`, `envs/ForestAviary.py` | PyBullet 植物、终止条件、奖励、观测 | 所有阶段 |
| `models/motor_action_codec.py` | RPM 与归一化动作的唯一物理转换 | 所有阶段 |
| `models/reference_packet.py` | 带坐标系的参考包、插值、逻辑通道和缓存 | 冻结后的规范基线 |
| `async_control/*` | latest-only IPC、runtime worker、运行时计时 | Stage 1 runtime / Stage 3 |
| `DSLPIDControl` + `TD3ReferenceTracking.teacher_action_from_env()` | 教师/经典下层 | Stage 1，Stage 2 教师 |
| `TD3ReferenceTracking` | 消费参考上下文的 RL 下层 | Stage 2 |
| `LearnedUpperDSLPIDEnv` | 低频速度命令到持久 `p_ref/v_ref` 的适配器 | 可复用的连续参考生成逻辑 |
| `high_level_planner/*` | 未来参考序列预测及其数据集 | Stage 3 上层候选 |
| `RuntimeHighLevelWorker` | 独立高层执行和 IPC 传输 | Stage 3 运行时承载 |
| `TD3HierarchicalAsync`, `models/hierarchical_async.py` | 内部归一化 sequence cache 与直接电机 Actor | 历史实验，隔离而非扩展 |

## 4. 唯一 `ReferencePacket` 规范

从本冻结版本起，正式高低层接口以 `models/reference_packet.py` 的语义为基线，
并在后续小步迁移中补充 `clock_domain`。它的控制载荷是未来位置和速度，**不是**
电机动作，也不是 latent。

```text
ReferencePacket
  version: int                         # 单调递增；仅接受严格更新的版本
  t_gen: float                         # 生产完成时间
  t_start: float                       # 首个参考点可执行的时间
  t_receive: float                     # 低层/通道实际接收时间
  valid_duration: float                # 从 t_start 起的有效时长
  clock_domain: "simulation" | "monotonic"  # 迁移时新增，不得混用
  relative_timestamps: float64[N]      # [0, ...]，严格递增，单位秒
  positions: float32[N, 3]             # frame_id 中的位置
  velocities: float32[N, 3]            # frame_id 中的速度
  frame_id: "world" | "local_navigation"
  origin_position: float32[3]          # local_navigation 到 world 的平移原点
  origin_attitude: float32[3]          # 保留的姿态元数据；v1 仅轴对齐局部系
  valid: bool                          # 生产者显式撤销时为 false
  code: optional float32[]             # 仅追踪元数据；下层不得依赖其控制
```

约束如下：

1. `positions`、`velocities` 与 `relative_timestamps` 的长度均为 `N >= 2`；
   所有数值有限。
2. 包在 `t_start <= now <= t_start + valid_duration` 且 `valid=true` 时有效。
3. 低层总是在 `world` 系消费插值后的 `current_position/current_velocity`；局部系
   必须先通过包自身的原点变换，禁止由调用者猜测原点。
4. 缓存拒绝版本倒退、重复包和接收时间倒退的包；只保留最新可用包。
5. 上层只能通过未来的位置/速度影响低层。latent、规划器调试信息和 policy
   action 可写入遥测，但不得成为下层控制输入的隐式旁路。

## 5. 异步时间模型

`relative_timestamps` 永远以秒表示相对执行时间；包内三个绝对时间必须处于同一
`clock_domain`。

| 场景 | `clock_domain` | 有效性判断 | 说明 |
| --- | --- | --- | --- |
| 逻辑异步/可重复单元测试 | `simulation` | `runtime_step * CTRL_TIMESTEP` | 可确定性地注入延迟、丢包与乱序 |
| PyBullet 非实时仿真 | `simulation` | 同上 | 不把 wall-clock 误当作仿真推进 |
| 多进程 runtime 延迟与 deadline 计时 | `monotonic` | `time.perf_counter()` | `sim_time` 仅作遥测，不参与包有效性 |

高层在 `t_gen` 生成决策，在 `t_start` 后可执行。低层读到包时写入 `t_receive`，
随后非阻塞地按 `now - t_start` 插值。正常包切换必须以旧包在新包 `t_start` 的
采样值（或已保存的 fallback 状态）作为新序列起点，以保持位置和速度连续。

首次包尚未抵达、包过期、或 worker 异常都不是等待条件：低层以最后有效参考的
悬停状态为 fallback，并记录 `missing/stale/fallback` 遥测。

## 6. 坐标系约定

| 名称 | 约定 |
| --- | --- |
| `world` | PyBullet 世界系，右手系，单位 m；`z` 向上，`x/y` 为水平平面 |
| `local_navigation` | 与 world 轴平行的局部导航系；位置以 `origin_position` 为零点 |
| 速度 | 永远是 world 轴方向的向量；局部包的速度不叠加平移 |
| 机体系 | 仅用于姿态、传感器或电机混控；不得写入 v1 `ReferencePacket` 的位置/速度字段 |
| 姿态/角速度 | 弧度和 rad/s；世界速度与机体系速度必须显式转换，禁止依赖 yaw 猜测 |

Stage 1 固定轨迹应直接产出 `world` 包，减少变量。只有需要保存“相对起点”的
数据集或 planner 标签时使用 `local_navigation`，并必须填充原点。

## 7. 现有实现差异

| 实现 | 时间字段和有效期 | 坐标语义 | 载荷与缓存 | 结论 |
| --- | --- | --- | --- | --- |
| `models/reference_packet.py` | `t_gen/t_start/t_receive/valid_duration` | `world` 或轴对齐 `local_navigation`，有 origin | 位置+速度、`code`、`AsyncReferenceBuffer`、故障通道 | 语义最完整；作为规范基线 |
| `async_control/reference_packet.py` | `t_generate/t_ready/valid_until` | 仅 world，未显式声明 frame | 可选且固定 16D `latent`、`AsyncBuffer` | runtime 专用重复定义；需改为使用规范包 |
| `models/hierarchical_async.py` 的 `AsyncReferenceCache` | 整数 `created_step/point_interval/valid_steps` | 归一化相对 sequence；没有 frame/origin | latent + sequence；内部直接供电机 Actor | 不是 `ReferencePacket`，不能跨进程或作为正式接口 |

特别说明：用户请求中所称的 `hierarchical_async.py` 实际并未声明一个
`ReferencePacket` 类型；它声明的是 `ReferenceCacheSnapshot/AsyncReferenceCache`。
它缺少物理单位速度、绝对时间、坐标系、接收时间和跨进程传输语义，因此不能被
静默视为第三种 Packet 实现。

## 8. 小步统一迁移方案

本方案只定义迁移顺序，不在本次冻结中改训练代码。

1. **先锁定契约测试。** 为规范包添加/保留坐标变换、插值、版本、同一 clock
   domain、过期、连续切换和 fallback 测试；Stage 1 以此作为唯一验收入口。
2. **规范包做唯一数据类型。** 在 `models/reference_packet.py` 以向后兼容方式
   增加 `clock_domain`；保留其字段名，避免打断 `TD3ReferenceTracking` 与
   `LearnedUpperDSLPIDEnv`。
3. **迁移 runtime 层，不碰训练器。** `async_control/reference_packet.py` 改为
   重导出规范包或提供单向、显式字段转换适配器：
   `t_generate -> t_gen`、`t_ready -> t_start`、接收端填 `t_receive`、
   `valid_until - t_ready -> valid_duration`，并显式写 `frame_id="world"` 与
   `clock_domain="monotonic"`。`latent` 仅转入遥测，不进入下层控制。
4. **隔离旧 step-cache。** 不修改 `TD3HierarchicalAsync` 训练语义；仅将其标为
   legacy，不允许新的 Stage 1/2/3 代码消费 `AsyncReferenceCache`。未来若保留该
   实验，需在 cache 边界显式解码为规范包，不能把归一化点数组冒充物理轨迹。
5. **最后替换生产者。** 固定轨迹、Planner 和 Upper DRL 都输出同一规范包；仅在
   Stage 2 lower 通过后，才把 runtime worker 的旧 motor-actor adapter 换成 Planner
   生产者。

迁移完成的判据是：项目中只有一个可跨模块导入的 `ReferencePacket` 数据类型，
所有控制消费者只接收其 world 系插值样本；旧类型只存在于有明确弃用说明的兼容
适配层。
