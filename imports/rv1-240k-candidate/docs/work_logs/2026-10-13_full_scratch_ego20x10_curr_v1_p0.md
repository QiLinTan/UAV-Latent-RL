# full_scratch_ego20x10_curr_v1：实现与seed43 P0结果

2026-10-13。新配置已注册，A/B/C冒烟完成，seed43停在累计6000上层转移。**工程契约通过，但在线更新后出现动作饱和与越界，暂不放行10k/20k或长训。** 保留原奖励、GRU和优化器设置；先定位价值估计/策略更新问题。

设计见[课程与冒烟协议](2026-10-13_ego20x10_training_smoke_protocol.md)。本轮没有启动其他训练seed，也没有在封存test40地图上做策略评估。

## 已实施

- 冻结train128 / validation20 / test40地图，地图候选seed分别1000–1127 / 2000–2019 / 3000–3039，本轮拒绝0张。开发seed43不在池中。
- 20×10m矩形、起点[-8,0,1]、目标[8,0,1]、40s、原1.75m感知。各地图完整40树池按生成顺序激活10/20/30/40棵嵌套子集；停用树没有碰撞体，也不产生ray回波。每阶段缓存膨胀BFS路径、长度和几何hash。
- 起点距离课程按剩余路径长度0.35–1.2 / 1.2–4 / 4–8 / 8–当前地图完整路径长度，之后原始起点；原始起点保留概率20% / 20% / 30% / 50% / 100%。门槛沿用窗口100、最少100、SR≥70%、CR≤20%。旧7m bands保持原默认值。
- 距离课程完成后才解锁密度课程，原始起点固定，10→20→30→40树；门槛窗口/最少100，SR80%/75%/70%，CR15%/20%/20%。最后一个L3成功回合不能计入L4密度门槛。关闭隐藏容易密度混合。
- 成功Replay按实际10/20/30/40树分层，每层5000条，整段成功回合入池；失败和预算截断回合不入池。地图采样独立于网络/探索RNG。
- 配置继承Full scratch的917D、独立Actor/Q1/Q2 GRU、15Hz/120Hz、XY投影、0.8/0.3m/s、原奖励、gamma=.99、主Replay60k、成功Replay20k/20%、batch256、5k warmup/update_after、初始LR3e-4、policy_freq2。没有预训练加载或Actor冻结。
- 每5k固定20张validation图，原始16m起点评估当前密度及Stage0；相同密度复用结果。记录SR/CR/timeout、引导区、终点距离、跟踪误差和回合最小净空分位数。periodic_raw / final_raw / selected各有独立目录、Actor hash和来源步数。
- 完整快照保存网络、targets、Adam、Replay分层、课程窗口、全局RNG、地图采样器RNG/访问计数和manifest指纹。恢复按新物理回合开始，丢弃中断回合的成功候选和课程结果。协议/地图改变、旧实验快照及覆盖已有实验均受检查。

地图manifest：`data/map_atlas/generated/ego20x10_curr_v1/manifest.json`，SHA256协议指纹：

```text
f8a871f25663cc0d6141c7cc9e42cefc1d7579b72a50a778833261a659dbdec2
```

代码：`envs/map_atlas/forest_pool.py`、`envs/forest/ego_curriculum_env.py`、`trainers/callbacks/ego_pool_eval.py`；入口`scripts/run_change_td3.py`。显式独立评估入口`scripts/evaluate_ego_forest_pool.py`，仅显式`--split test`才飞封存地图，不接入训练/选模。

## A：契约和回归

最终相关测试**34 passed**。日志：`analysis_outputs/full_scratch_ego20x10_curr_v1_p0/final_regression_tests.log`。

覆盖3张不同地图×5个起点级×4个密度级，共60组Full reset；每组执行真实下层tick后再reset。检查实际激活树数、ray、碰撞、解析净空、矩形边界、PID、运动状态、reference/packet和917D历史。强制组合及小窗口只用于隔离测试。

真实近目标Full测试控制器成功回合完整进入10树成功Replay；后续真实height-bound及40s timeout均未增加成功池。测试控制器未进入正式run。非空Adam、不同步targets及课程/地图采样器恢复后，在相同Replay/RNG下下一次更新与未中断分支逐网络hash一致。旧地图、旧起点/障碍课程、旧Full scratch与Full FT回归通过。

## B/C：正式冒烟结果

| 检查 | 2k | 累计6k |
|---|---:|---:|
| 主Replay | 2000 | 6000 |
| Critic更新 | 0 | 1001 |
| Actor更新 | 0 | 500 |
| 成功Replay，10树层 | 269 | 297 |
| 20/30/40树层 | 0/0/0 | 0/0/0 |
| 起点/密度 | L0/Stage0 | L0/Stage0 |

2k Actor/Critic与初始化hash完全一致、Adam为空。恢复前后2k Replay的全部state/action/next/reward/not_done逐数组一致，采样器恢复证据匹配快照。到6k Actor/Critic及各target均改变，Adam非空，无冻结或保守阶段切换。

累计采样23次，访问20张不同train地图；6000条transition都有地图身份、几何hash、实际树数、起点source与终止原因，未混入validation/test。逐tick奖励求和、packet/RPM、历史移位及投影动作Replay检查通过；另复核6000条Replay动作×[0.8,0.8,0.3]与实际command_velocity一致。

21个完整回合：success3、timeout6、height_bound8、xy_bound4、collision0。16个band回合中success3，另5个原始起点回合；band窗口SR=3/16，未到最少100回合，所以不晋级。密度门槛累计0回合。2k与6k预算分别截断529步和72步的未完成回合，不把它们记作success或课程结果。

**5k在线更新后完成的11个回合全部失败：height_bound7、xy_bound4；其中8个band回合也是失败。** 因此不能把此前随机探索的3次成功当成已学会近目标导航。

| 固定validation20，Stage0原始16m | SR | CR | timeout | height_bound |
|---|---:|---:|---:|---:|
| 2k raw | 0% | 5% | 95% | 0% |
| 5k周期raw（Critic只更新1次，Actor尚未更新） | 0% | 5% | 95% | 0% |
| 6k raw | 0% | 5% | 0% | 95% |
| 6k结束恢复的selected，来源5k | 0% | 5% | 95% | 0% |

`model_final`是恢复后的5k peak，Actor仍与初始化相同；**它不是6k原始策略，也不是已成功策略。** 后续分析与续训使用`model_raw_step6000`和`training_state_step6000.pt`。2k/5k/6k快照均保留，raw/selected评估没有互相覆盖。

## 稳定性异常及先行机制审计

已记录每次更新Q1/Q2均值、标准差、范围/分位数，TD绝对误差均值/分位数，梯度、Actor更新/冻结、学习率及成功Replay采样数。固定共同状态集记录相邻策略RMS；当前访问状态另记录近/远组动作边界、垂直与径向命令、真实速度。近/远分组已修正为反解KIN归一化后的真实米制距离；原始goal特征被缩放/截断，直接用其范数会分错组。该修正仅影响监测，不改变Actor输入或训练奖励；已重算2k/5k/6k探针，保留原记录。

主要异常：共同状态上2k→5k策略RMS变化0，5k→6k变化0.734；6k垂直命令均值约+0.29995m/s。按真实距离采样的256个近状态和256个远状态，垂直动作均100%达到饱和阈值。训练Q/TD/梯度均有限；Critic裁剪前梯度最大105.61，按原阈值1裁剪。这些不能证明策略稳定。

先行审计结论：

- Replay真实执行动作、终止/超时mask、成功整回合入池及10树分层均通过；每次更新实际抽取51条成功样本，累计297条成功经验没有因续训清空。
- reset状态、917D历史、8个真实tick/终止部分tick、连续奖励积分和RPM执行一致；没有把40棵树误标成10棵。
- 原LR/TD3延迟更新、target滞后、Adam绑定及恢复一致性通过；未启用冻结或意外保守阶段。
- 固定XY扫描z∈{-1,-0.5,0,0.5,1}，近状态Q1均值从-0.96065上升至-0.93079，远状态从-0.50754上升至-0.48165；近/远组约82.8%/57.8%的Q1局部z梯度为正。Actor所优化的Q1确有向上偏好。
- 3张隔离Full地图、相同reset及初始Actor XY命令，仅干预垂直动作，+1的实际折扣回报均低于0，尽管起始Q1将+1排在0之上。这是有限时域的排序不一致证据，**不是完整Q真值或最终根因证明**。固定XY本身也会越界，所以零z干预不能被称为修复后的成功策略。

建议下一项是使用原Replay/快照重构相同minibatch的TD target，并对关键近目标状态做有界动态反事实比较，区分价值估计和Actor更新偏置。此次未改奖励、GRU或训练预算。

## 产物与复现

汇总及硬断言：`analysis_outputs/full_scratch_ego20x10_curr_v1_p0/report.json`；机制审计：`stability_audit.json`；Full干预：`vertical_interventions.json`。完整首回合、实际成功/失败及预算边界轨迹在`runs/full_scratch_ego20x10_curr_v1_seed43/p0_audit/traces/`，含上下层及Replay侧车NPZ。修正的米制探针：`policy_diagnostics_world_corrected.json`。

本轮实际执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/uav-mpl \
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 \
  full_scratch_ego20x10_curr_v1 --seed 43 --total_steps 2000

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/uav-mpl \
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 \
  full_scratch_ego20x10_curr_v1 --seed 43 --total_steps 6000 \
  --extra --resume_training checkpoints/full_scratch_ego20x10_curr_v1_seed43/training_state_step2000.pt
```

这些目录已存在，第一条fresh命令再次运行会拒绝覆盖；`--total_steps`是累计预算。当前保留的继续训练来源是`training_state_step6000.pt`，但先完成上述稳定性审计再增加预算。

最终检查14个既有K1/旧Full scratch/Full FT产物SHA256均未改变。运行使用CPU；未自动启动P1、240k、多seed或封存test策略评估。
