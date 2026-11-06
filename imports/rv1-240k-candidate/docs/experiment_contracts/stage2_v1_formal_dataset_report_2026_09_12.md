# Stage2-v1 Formal Canonical Dataset

日期：2026-09-12
状态：正式数据采集与歧义 gate 通过；允许进入 BC warm-start，尚未训练网络。

## 冻结合同

```text
dataset_version = stage2_v1_external_packet_canonical_v2
context_contract_id = stage2_v1_external_packet_30d
teacher_contract_id = stage2_v1_canonical_dslpid_world_rate
reference = models.reference_packet.ReferencePacket, world frame
packet/control frequency = 15 Hz / 120 Hz
teacher target = current_reference(sim_time + 1/120)
action = asymmetric_rpm
```

Teacher 每次标签计算前清零位置和姿态积分，并由当前 quaternion、world angular
velocity 与 `dt` 重建 DSLPID 的 `last_rpy`。数据不包含 Actor 动作，也未执行任何网络
更新。

## 数据组成

- 51 个确定性 seed：seed 0 nominal，另外 25 组正负成对初始扰动。
- 每个 seed 包含 hover 和固定 7 m straight，共 102 episodes。
- 扰动范围不超过 teacher determinization probe 已验证的 envelope。
- 共 254,286 transitions；34 个样本字段的首维均严格一致。
- `train/validation/test` 按 seed 隔离，无 episode 或 sample 重叠。

| Split | Seeds | Episodes | Samples |
| --- | ---: | ---: | ---: |
| train | 39 | 78 | 194,454 |
| validation | 6 | 12 | 29,916 |
| test | 6 | 12 | 29,916 |

## Teacher 闭环质量

| Task | Passed | Position RMSE mean/max | Velocity error mean/max | Height RMSE mean/max | Max roll/pitch | Final distance max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hover | 51/51 | 0.00292 / 0.00386 m | 0.01038 / 0.01497 m/s | 0.00214 / 0.00327 m | 0.06122 rad | N/A |
| straight | 51/51 | 0.01435 / 0.01749 m | 0.05580 / 0.06607 m/s | 0.00346 / 0.00531 m | 0.26070 rad | 0.19759 m |

全数据 motor saturation ratio 为 0；missing/stale packet、fallback 和 packet version
continuity failure 均为 0。最大 context reconstruction error 为 0，最大 RPM codec
round-trip error 为 0.000108 RPM，最大 canonical hidden-state 合同误差为
`1.44e-8`。

## 标签歧义结果

跨 episode 最近邻中，99.18% 的样本存在半径 0.05 内的邻居：

- close-pair action `L∞` p95：0.000405；
- local dispersion / teacher RMS：13.12%；
- overall、hover、straight 的局部平滑度 gate 均通过。

对 256 个跨全数据区间的样本，从保存的 quaternion、状态和 current reference 独立
重算 canonical teacher：

- 最大 RPM `L∞` 误差：0.003766 RPM；
- 最大 normalized-action `L∞` 误差：`4.53e-7`；
- 最大 hidden-input 重建误差：`4.44e-16`；
- 相同输入重复调用最大 RPM 误差：0。

因此 `collection_gate_passed=true`、`ambiguity_gate_passed=true`，数据目录内的
`quality_gate.json` 明确给出 `training_allowed=true`。

## v1 失败工件

第一次正式采集的 v1 中，审计字段 `state_quaternion` 被每步重复追加，导致其首维与
其他字段不一致。控制 rollout 和标签未受影响，但独立重算发生索引错位。v1 保留为失败
证据，禁止训练；collector 已增加全字段 sample-axis 硬断言，正式发布版本为 v2，未对
v1 做原地修补。

## 下一步

只允许使用 v2 的 train split 执行 BC warm-start；validation 用于 checkpoint 选择，
test 在模型冻结前不得参与训练或调参。Actor-only hover 和 7 m straight gate 仍未执行。
