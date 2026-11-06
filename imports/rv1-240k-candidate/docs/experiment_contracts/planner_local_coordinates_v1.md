# Planner 局部坐标合同 v1

- Encoder 保持原生 263D 输入及 `preprocess_state`，不得旋转或重新归一化其输入。
- Planner 的原点和 yaw 均取状态快照生成时刻；x 向前、y 向左、z 向上。
- `goal_local = Rz(-yaw) @ (goal_world - position_world)`，单位米；不可用被截断的归一化 goal 尾部反推距离。
- 标签 `delta_p_local = Rz(-yaw) @ (future_position_world - snapshot_position_world)`，单位米。
- 标签 `v_local = Rz(-yaw) @ future_velocity_world`，单位米/秒；不是减去 UAV 当前速度的相对速度。
- 参考点固定在 0.1/0.2/0.3/0.4/0.5 秒。转换回世界系必须使用同一快照的 origin/yaw，不能使用延迟完成时的新姿态。
- 本合同转换不自动完成延迟补偿或连续切换；后续 PacketGenerator 仍需单独验证两者。
- 当前雷达是世界系八条水平射线，索引 j 对应 j*pi/4，值为距离/1.75m。
- Planner 局部八方向采用相邻世界射线最小值重采样，整45度旋转时为精确置换；任意角度为保守的离散近似，不是额外传感器，也无法发现原射线间隙的障碍。
- 保留原始 world ranges 和 yaw。将历史雷达用于当前航向系时，各历史记录都按当前快照 yaw 重新分桶；历史采样原点不同，不得冒充当前地图。
- 89D v2 预留接口为 latent16 + local_goal3 + ranges_history64 + local_velocity3 + previous_reference_velocity3；A–D 阶段不训练该 Planner。
- 归一化统计只能来自 train maps。旧27D世界系模型作为冻结旧版本，不能直接读取本合同的新输入。
- Map ID 对 start、goal、障碍 xy/radius/height 做6位小数规范化，障碍排序后计算 SHA-256；seed、episode、body ID 不参与。
- yaw 属于 episode 姿态配置；相同几何不同 yaw 不允许跨 split。测试 split 不供诊断脚本读取。
