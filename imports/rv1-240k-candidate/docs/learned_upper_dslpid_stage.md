# 学习型上层 + 冻结 DSLPID 下层阶段说明

更新日期：2026-08-08

## 目的

新增一条较低风险的分层训练主线，同时完整保留原有
`TD3HierarchicalAsync` 直连四电机版本：

```text
29 维上层观测
  → Plain TD3（15 Hz）
  → 三维世界系速度命令
  → 持久 p_ref/v_ref 与加速度限制
  → ReferencePacket
  → DSLPIDControl（120 Hz）
  → asymmetric_rpm
  → 四电机
```

该模式用于先验证“学习型导航上层是否有效”，不用于证明学习型电机下层、
联合端到端训练、极端通信鲁棒性或 sim-to-real。

原直连电机模式没有被替换：

```bash
python -m scripts.run_change_td3 hierarchical_async
```

## 接口定义

### 上层观测

上层只接收：

- 12 维实时运动学；
- 3 维归一化目标相对位置；
- 8 维障碍距离。
- 3 维归一化参考位置误差；
- 3 维归一化参考速度。

基础环境的 240 维电机动作历史不暴露给上层。适配器内部的持久参考状态会
影响下一状态，因此将其显式暴露给策略，上层观测为 29 维。

### 上层动作

TD3 输出三维归一化动作：

```text
[世界系 vx, 世界系 vy, 世界系 vz]
```

标称上限为水平 `0.8 m/s`、垂向 `0.3 m/s`。适配器以 120 Hz 对命令做
加速度限幅，持续更新持久的 `p_ref/v_ref`，并从上一包已执行状态生成下一包。
该模式不再调用 minimum-jerk 参考生成器。

### 时间尺度与 replay 语义

一次上层 `env.step()` 默认执行 8 个底层控制周期：

- 底层：120 Hz；
- 上层：15 Hz；
- replay buffer 保存上层动作；
- 环境奖励在 8 个底层周期内求和；
- 若中途成功、碰撞或越界，立即结束该上层 step。

这避免了把 DSLPID 产生的电机动作错误地作为 TD3 上层动作写入 replay。

## 快速运行

无障碍固定起终点版本：

```bash
cd /home/tequial/projects/UAV-Latent-RL
/home/tequial/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 upper_dslpid
```

默认输出：

- TensorBoard：`runs/upper_dslpid_velocity_v2/`
- 检查点：`checkpoints/upper_dslpid_velocity_v2/`

小规模验证：

```bash
/home/tequial/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 upper_dslpid \
  --total_steps 5000 \
  -- --eval_interval 1000 --eval_episodes 3
```

## 当前森林配置

`upper_dslpid` 现在恢复为森林环境：起点 `(-3.5, 0.0, 1.0)`，终点
`(3.5, 0.0, 1.0)`，场景生成 24 棵树。显式关闭路线阻挡树，并固定在课程
阶段 0，使起终点直线周围保留宽净空走廊；树木只作为航道两侧的森林背景，
本阶段不测试避障。

至少同时检查：

- `eval/final_goal_distance` 相对随机初始化明显下降；
- `eval/success_rate` 不再长期为零；
- `eval/collision_rate` 为零；
- `eval/max_roll_pitch` 保持在稳定范围；
- `eval/tracking_position_rmse` 有限；
- 终止原因不是高度或姿态越界。

不要同时引入航道内障碍、latent、学习型电机下层或通信故障，否则无法判断
性能变化来自哪一层。

## 已完成的工程验证

- 新适配器真实 PyBullet 单步和跨包 `p_ref/v_ref` 连续性测试通过；
- 原 `hierarchical_async` 与 `reference_tracking` 回归测试通过；
- 最大前进命令 5/5 成功完成 7 m 路线，稳态实际速度约 `0.7998 m/s`；
- 零速度悬停稳定，四方向命令与位移符号一致；
- 5,000 步 TD3 评估最终距离由 `5.664 m` 改善至 `3.173 m`，但成功率仍为
  0，最终主要失败原因为横向越界；
- 5,000 步结果位于
  `runs/upper_dslpid_velocity_v2/20260808_seed42_5k_retry1/`。

接口可达性已经通过，但 5,000 步策略尚未学会稳定沿目标方向导航，不能进入
障碍或 latent 阶段。

## 单树障碍微调（50,000 步）

无障碍 `model_best` 已归档为
`checkpoints/upper_dslpid_velocity_v2/baseline_clear_route/model_best`。独立模式
`upper_dslpid_one_tree` 在路线中点加入一棵阻挡树，从该基线继续微调，并将
结果写入 `runs/upper_dslpid_one_tree_v1` 和
`checkpoints/upper_dslpid_one_tree_v1`。

50,000 步最终评估为 5/5 成功、0 碰撞，平均 145.2 个高层动作到达目标，
最终距离约 0.199 m。训练中的 50 次周期评估有 40 次为 5/5 成功；中途存在
actor 策略漂移造成的横向越界或终点附近超时，因此正式使用时应选择评估保存的
checkpoint，而不是假定训练步数越大越好。

## 固定五树验证（50,000 步）

`upper_dslpid_five_trees` 的第二版使用 5 棵固定、不规则分布的树，不叠加随机
背景树。测试中存在一条计入无人机半径后仍保持至少 0.35 m 净空的连续路径。
该版本同时关闭远距离悬停仍会累积的正 `distance_reward`，并加入每低层步
0.01 的时间成本。

结果写入 `runs/upper_dslpid_five_trees_fixed_v2` 和
`checkpoints/upper_dslpid_five_trees_fixed_v2`。50,000 步最终模型成功、无碰撞，
154 个高层动作到达；自动选择的最佳模型位于 37,000 步，其最近三轮评估均
成功且无碰撞，窗口平均回报 274.76。最佳模型单回合最小净空约 0.314 m，
最终模型约 0.291 m，均低于期望的 0.35 m 安全距离，因此固定布局任务已经
跑通，但安全裕量仍需继续改善。

## 固定五树 + 24 棵随机背景树（50,000 步）

模式 `upper_dslpid_five_trees_forest24` 从固定五树最佳模型继续训练。每个布局
包含 5 棵固定主障碍和 24 棵随机背景树；生成器拒绝任何侵入已验证安全通路的
背景树，并在无法生成完整 29 棵树时显式报错。

50 次周期评估有 45 次达到 10/10 成功。34,000 步的 `model_best` 窗口平均
回报为 307.86；50,000 步最终模型也达到 10/10 成功、0 碰撞，平均 145.9
个高层动作。额外逐 seed 检查中，最终模型最小净空为 0.266 m、平均最小净空
为 0.294 m，仍低于 0.35 m 目标，因此不需要回退背景树数量，但后续选模和奖励
应显式纳入最小净空。

## 多帧横向语义 latent（50,000 步）

改造前的完整森林最佳模型已归档到
`checkpoints/upper_dslpid_five_trees_forest24_v1/baseline_pre_semantic_latent/model_best`。
第一阶段语义版本保持 ReferencePacket、速度接口和 DSLPID 不变，将上层观测从
29 维扩展为 77 维：29 维当前状态加 4 帧 `[8 向测距、3 维速度命令、跟踪误差]`
历史。16 维任务 latent 同时接受未来测距、危险概率和任务推进监督，并只通过
最大幅度 0.25 的残差修正冻结的基线 Actor。

结果位于 `runs/upper_dslpid_semantic_latent_v1` 和
`checkpoints/upper_dslpid_semantic_latent_v1`。50 次周期评估全部为 10/10 成功、
0 碰撞；47,000 步最佳窗口平均回报 317.58、平均长度 144.3、跟踪误差
0.0221 m。语义总损失从 1.064 降至 0.056，测距预测损失从 0.227 降至
0.013，风险损失从 0.696 降至 0.043。该结果支持“多帧语义 latent 有用”，
但严格归因仍需要无 latent 残差、无历史等消融对照。
