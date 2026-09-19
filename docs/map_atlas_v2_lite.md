# v2-lite 地图加载器与地图集

## 目标

该阶段只替换环境几何和任务布局，不改变以下既有控制契约：

```text
上层 3D 速度动作
  -> 持久 p_ref / v_ref
  -> ReferencePacket
  -> 冻结 DSLPID（120 Hz）
  -> asymmetric_rpm
```

`MapSpec` 是地图的统一描述：来源、许可、训练划分、起终点、边界、测距范围和
碰撞对象。启动时会在考虑无人机半径后验证起终点及 primitive 地图的二维连通性。
不连通的地图在创建 PyBullet 环境前直接报错。

## 已交付的初始地图条目

| Map ID | Split | 用途 |
| --- | --- | --- |
| `procedural_forest_train_v1` | train | 大量不同随机种子的森林训练族 |
| `procedural_forest_multitree_train_v2` | train | 1→2→3→5 主障碍课程；最终阶段保留 20% 简单难度 |
| `procedural_forest_validation_v1` | validation | 固定五主障碍 + 背景森林验证族 |
| `procedural_forest_mirror_probe_v1` | test | 五主障碍左右镜像的封闭泛化探针 |
| `subt_tunnel_proxy_v1` | validation | 隧道、交替挡板与岩柱的加载器验证 |
| `flightbench_maze_proxy_v1` | test | 迷宫布局的端到端回归测试 |

后两项是本地结构代理，不能写为 DARPA SubT 或 FlightBench 的正式结果。正式地图
接入后应创建新的 `map_id`，并将代理结果与外部 benchmark 结果分开报告。

## 运行检查

先运行地图和控制链回归测试：

```bash
cd /home/tequila/UAV-Latent-RL
/home/tequila/miniconda3/envs/drones/bin/python -m unittest tests.test_map_atlas
```

在隧道代理地图上做 2,000 个上层 step 的接口冒烟训练：

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 upper_dslpid_recurrent \
  --total_steps 2000 \
  -- --map_id subt_tunnel_proxy_v1 --eval_interval 1000 --eval_episodes 3 \
  --log_dir runs/upper_dslpid_recurrent_subt_proxy_smoke \
  --ckpt_dir checkpoints/upper_dslpid_recurrent_subt_proxy_smoke \
  --trajectory_output_dir analysis_outputs/upper_dslpid_recurrent_subt_proxy_trajectory
```

这只是接入检查，不以成功率作为学习能力结论。正式训练应从零开始，并固定地图
划分：程序化地图仅用于 train；未见的固定布局用于 validation；外部成熟地图用于
封闭 test。

训练和验证地图可在同一次运行中显式分开。推荐使用独立的
`upper_dslpid_recurrent_stable_v3` 模式：它仍然输出 `ReferencePacket` 给冻结
DSLPID，但会在接近目标时抑制高速和背离目标的速度，并取消越过目标后的正向路线
进度奖励。它还会保留成功 episode 的经验、在稳定成功后降低更新幅度并锚定 actor；若
连续验证显示策略崩溃，会恢复 `model_best` 后停止。它不能与旧奖励训练出的 checkpoint
混用，因此需要从零开始训练：

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 upper_dslpid_recurrent_stable_v3 \
  --total_steps 120000 \
  -- --map_id procedural_forest_train_v1 \
  --eval_map_id procedural_forest_validation_v1 \
  --eval_interval 2000 --eval_episodes 5 \
  --log_dir runs/upper_dslpid_recurrent_stable_v3_forest \
  --ckpt_dir checkpoints/upper_dslpid_recurrent_stable_v3_forest \
  --trajectory_output_dir analysis_outputs/upper_dslpid_recurrent_stable_v3_forest_trajectory
```

每次评估会覆盖写出 seed 0 的最近一次轨迹：
`seed_0_trace.npz`、`seed_0_xy.png`、`seed_0_xyz.png` 和 `summary.json`。训练
日志中的 `reward/near_goal_*`、`reward/route_overshoot_penalty` 与
`reward/proximity_penalty` 可用于核验新项是否实际生效。

`flightbench_maze_proxy_v1` 已被标记为 test。若显式把 test 地图作为
`--eval_map_id`，启动器会警告它不应用于模型选择。

`procedural_forest_mirror_probe_v1` 同样是 test。训练结束后加载
`model_best` 对其评估；不要在训练过程中把它作为 `--eval_map_id`，以免测试集泄漏。

## 外部资产放置位置

```text
data/map_atlas/raw/        # 下载的上游 SDF、PLY、纹理和路径；不提交 Git
data/map_atlas/converted/  # 转换后的 OBJ/STL 碰撞资产；不提交 Git
data/map_atlas/catalog_v1.json
```

SubT 的 SDF/Gazebo 世界需要提取或转换碰撞网格。FlightBench 的 `env.ply` 是点云，
必须经过清理、体素化或网格重建后才可用于可靠碰撞；不能直接把点云当实体墙体。
每次加入外部地图都必须新建一条 catalog 记录，保留下载来源、commit、许可、尺度
和坐标系变换。

## 96 射线多障碍 v4

`upper_dslpid_recurrent_multitree_v4` 使用 32 个方位角和
`-30°/0°/+30°` 三层仰角，共 96 条世界坐标系射线。上层动作仍是三维速度，且继续
通过 `ReferencePacket` 交给冻结的 DSLPID。该版本必须从零训练，不能加载 8 射线
版本的 checkpoint。

课程按 episode 数在第 100/250/450 局进入 2/3/5 主障碍阶段；5 障碍阶段有 20%
概率从 1/2/3 障碍中采样。成功经验按 1/2/3/5 难度分池并均衡抽样。周期验证为 10
局，正式报告建议加载选中 checkpoint 后独立验证 30 局。
