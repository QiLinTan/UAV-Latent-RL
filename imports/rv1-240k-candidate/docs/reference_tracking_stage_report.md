# 下层参考跟踪阶段实施与验证记录

## 当前目的

本阶段冻结学习型上层，不使用 latent，只验证以下链路：

`规则可行参考 → ReferencePacket → 异步缓存/插值 → 下层控制器 → 四电机动作`

只有下层门槛通过后，才恢复学习型上层、复杂感知和联合强化学习。

## 已实现内容

- 具有位置、速度、相对执行时间、生成/启动/接收时间、有效期、版本、
  坐标系和参考原点的 `ReferencePacket`。
- 原子式 `AsyncReferenceBuffer`，拒绝旧版本和乱序包。
- 基于执行时间的插值、前视读取、年龄和过期判断。
- 保守的规则直线/悬停参考生成器，以及速度、加速度和垂向速度检查。
- 固定悬停锚点；参考刷新不再把漂移后的当前位置误当作新悬停目标。
- 不含 latent 的下层 TD3；经验回放保存动作时刻的精确参考上下文。
- 独立的电机约束层和降级悬停控制器。
- PID/几何教师、教师样本回放、扰动恢复样本和行为克隆预训练。
- 残差四电机 Actor 与结构化虚拟通道混控 Actor 对照。
- 最近四步动作历史输入。
- 可复用的教师/Actor门槛验证脚本。

## 验证门槛

- 平均飞行长度不低于完整回合的 90%；
- 参考跟踪 RMSE 不高于 0.5 m；
- 姿态、高度、碰撞或平面越界失稳率不高于 2%；
- 电机饱和比例不高于 5%。

## 已获得结果

| 控制器 | 任务 | 平均步数 | 跟踪RMSE | 失稳率 | 结论 |
|---|---|---:|---:|---:|---|
| PID/几何教师 | 直线 | 1442 | 0.014 m | 0% | 通过 |
| 残差Actor（2500步BC） | 直线 | 511 | 0.088 m | 100% | 不通过 |
| 结构化混控Actor（2500步BC） | 直线 | 233 | 0.081 m | 100% | 不通过 |
| PID/几何教师 | 固定锚点悬停 | 1442 | 0.000 m | 0% | 通过 |
| 残差Actor（1800步BC、4步历史） | 固定锚点悬停 | 522 | 0.062 m | 100% | 不通过 |

结果文件位于 `runs/reference_tracking_validation/`。

当前事实是：参考包、轨迹和RPM接口可由教师稳定执行，但神经网络下层尚未达到
闭环稳定门槛。Actor没有电机饱和，主要失败形式为小动作误差长期累积后的姿态越界。

## 当前决策

暂不运行 50 万步联合训练。继续训练当前 Actor 会重新混合上层、下层和 TD3 问题，
无法形成可信的技术验证结论。

下一阶段应先比较：

1. 带隐藏状态的 GRU 下层控制器；
2. 更长的短时状态/动作历史；
3. 角速度加总推力接口与四电机直控接口；
4. 悬停行为克隆后仅针对稳定性的小规模强化学习微调。

在任一神经网络下层通过固定锚点悬停门槛后，再进入直线、圆弧和异步故障测试。

## 验证命令

```bash
/home/tequial/miniconda3/envs/drones/bin/python \
  -m scripts.validate_reference_tracking \
  --controller teacher \
  --episodes 5
```

```bash
/home/tequial/miniconda3/envs/drones/bin/python \
  -m scripts.validate_reference_tracking \
  --controller actor \
  --reference-mode hover \
  --checkpoint checkpoints/reference_tracking_hover_history_bc/model_1800 \
  --episodes 5
```
