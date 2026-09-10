# 专利主实施例冻结清单

## 冻结对象

本清单冻结 `upper_dslpid_five_trees_forest24_v1` 作为“连续时序参考的无人机多速率控制”专利的主实施例代码与仿真证据。其代码基线为 Git 提交 `b90ba78507b3a15e8a6ba08490b16128510339b5`；本清单及其复现实用程序由后续冻结提交纳入版本控制。

## 已冻结的技术链路

```text
29 维上层观测
  -> Plain TD3，15 Hz
  -> 三维世界坐标系速度命令
  -> 持久计划参考位置和计划参考速度
  -> 15 节点、1 s 前视 ReferencePacket
  -> 冻结 DSLPIDControl，120 Hz
  -> 非对称 RPM 编码和解码接口
  -> CF2X 四旋翼 PyBullet 物理仿真，240 Hz
```

主代码文件如下：

- `envs/learned_upper_dslpid.py`：15/120 Hz 适配器、跨包参考连续性、上层转移聚合；
- `models/reference_packet.py`：时序位置速度参考及时间字段；
- `models/motor_action_codec.py`：以悬停转速为零点的非对称 RPM 接口；
- `envs/forest/`：森林任务、成功和终止判定；
- `scripts/run_change_td3.py`：本模式训练配置；
- `scripts/replay_upper_dslpid_forest24.py`：不训练、不写检查点的 GUI 复现程序。

## 冻结的模型与原始实验数据

```text
checkpoints/upper_dslpid_five_trees_forest24_v1/model_best_*
checkpoints/upper_dslpid_five_trees_forest24_v1/model_best_metrics.json
checkpoints/upper_dslpid_five_trees_forest24_v1/model_50000_*
checkpoints/upper_dslpid_five_trees_forest24_v1/model_final_*
runs/upper_dslpid_five_trees_forest24_v1/events.out.tfevents.*
```

在当前任务中，以 `model_best` 加载后，按本文件列出的配置独立复跑 seed 0--9，均由环境返回 `done_reason=success`。成功判定是目标距离严格小于 `0.20 m`；该边界条件必须随实验表一起提交。

GUI 复现命令：

```bash
PYTHONPATH=. /home/tequial/miniconda3/envs/drones/bin/python \
  -m scripts.replay_upper_dslpid_forest24 --seed 0
```

## 专利表述的证据边界

可以作为主实施例事实提交的内容：

- 学习型上层按 15 Hz 输出三维任务级速度命令；
- 下层以 120 Hz 从带时间标签的位置速度参考中恢复当前参考，并使用冻结 DSLPID 闭环；
- 物理仿真频率为 240 Hz；
- 新参考包由先前计划参考状态初始化，位置和计划速度跨包连续；
- 一个上层动作实际执行至多 8 个下层周期，奖励和终止信息按该区间聚合；
- 上层观测包含参考位置误差和参考速度；
- DSLPID 输出经过非对称 RPM 接口适配动作格式。

不得据该主实施例单独声称的内容：

- 学习型上层在通信延迟、丢包、乱序和过期条件下的端到端联合导航已验证；
- 返航或安全接管已经实现并验证；
- 学习型电机下层与当前学习型导航上层已经组成统一闭环；
- 复杂随机森林中具有部署级避障鲁棒性，或最小安全净空已达到 0.35 m；
- sim-to-real、实机飞行或工程安全认证已完成。

## 场景解释

该场景包含 5 棵固定路线相关树与 24 棵背景树。背景树生成器不允许侵入已验证通路。因此，结果支持“固定障碍与受保护背景树场景中的端到端到达”，但不应扩大表述为无约束随机森林避障的全面验证。

## 建议提交结构

1. 本清单、当前 Git 提交号与版本标签；
2. 主代码及其依赖版本说明；
3. `model_best`、指标 JSON 与原始 TensorBoard event 文件；
4. GUI 复现脚本、10 个固定 seed 的逐回合结果、代表性轨迹图；
5. 单元测试：`test_learned_upper_dslpid.py`、`test_reference_tracking.py`、`test_motor_action_codec.py`；
6. 异步缓存、latent、学习型下层的独立材料，必须标注为补充或可选实施例。
