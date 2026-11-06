# Full EGO20x10：Critic验收、任务候选与短机制对照

日期：2026-10-13。本轮按用户确认的方向执行；全部实验隔离于正式模式，未启动10k/20k/240k续训，未执行test40。最终报告：[REPORT.zh.md](../../analysis_outputs/full_ego20x10_next_step_v1/REPORT.zh.md)。

## 已完成

1. 旧7m冻结Actor实验使用已有4k/10k/20k快照完成确定性及目标噪声验收。20k Q1主排序52.1%、主要局部梯度42.0%，未通过。并发恢复追加的相同记录在新目录中去重，原始记录保留、冲突0。报告：[完整验收](../../analysis_outputs/full_critic_frozen_v1_seed43/completed_review/REPORT.zh.md)。
2. 新16m工程候选分别验证目标差尺度[20,10,2]、位置x/10/y/5/(z−1)/1、引导速度投影、gamma0.997。917D与历史布局、Full物理轨迹、奖励逐tick聚合均一致；只在目标不可执行时改变对应跟踪成本。输入目标X饱和20.32%→0%；候选学习收益尚未验证。
3. 原5k完整学习状态上做Actor单步保护短对照，每分支200次Critic更新、100次Actor候选。相同真实Replay、采样及目标噪声，无保护四网络匹配A5200。保护后的L0成功率：初始11/20→第1次12/20→第10次4/20→第50/100次0/20，单步约束不足以保留能力。
4. 冻结Actor进行TD-only与TD+局部Full回报差监督短对照，每分支200次Critic更新。12状态/36动作对，拟合与检查按train地图分开。可区分对仅4/5；拟合Q1方向75%→100%，检查20%→40%，检查RMSE未改善。最终新梯度分支可区分样本仍不足，未通过。

局部监督使用原A至6k的离线代表状态和6s有限回报，不是假设在在线5k时已获得这些未来标签，也不宣称等于无限时域TD3目标。Actor保护原`train`字段中的更新后饱和日志描述未经回溯的候选；采用后的实际策略状态以`checkpoint_probes.json`和Full评估为准。

## 决策

不采用单步保护或当前小型局部校准作为已解决方案，不恢复大规模训练。下一轮先扩大可区分局部方向样本、检验完整剩余时域，并为Actor增加相对固定保留策略的累计偏离约束；16m任务候选需要一致编码的独立学习对照。当前没有选择或注册正式组合配置。

## 实现与验证

- [候选模块](../../scripts/ego20x10_contract_variants.py)
- [单因素工程探针](../../scripts/probe_ego20x10_contract_variants.py)
- [Actor保护短对照](../../scripts/run_ego20x10_actor_guard_pilot.py)
- [Critic局部校准短对照](../../scripts/run_ego20x10_local_calibration_pilot.py)
- [Critic验收去重与汇总](../../scripts/review_full_critic_acceptance.py)
- [综合报告工具](../../scripts/summarize_ego20x10_next_step.py)

17项相关测试通过；346份原始输入、生产源码和checkpoint哈希不变。新在线训练转移0，新增离线Critic更新共800次。四个正式结果目录为`full_critic_frozen_v1_seed43/completed_review`、`full_ego20x10_contract_corrections_v1`、`full_ego20x10_actor_guard_pilot_v1`、`full_ego20x10_local_calibration_pilot_v1`（均位于`analysis_outputs`）。`*_precheck`过程目录保留但不参与结论。

所有新实验入口要求新的输出目录，避免覆盖已有产物；不需要再次运行已完成的实验。
