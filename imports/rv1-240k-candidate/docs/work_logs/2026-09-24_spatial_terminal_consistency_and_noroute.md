# 三维接近—末端入球回报审计与水平推进消融

## 已完成结果

20 个 seed 全部复现原正式评估：17 次 timeout、3 次 height bound。共采样 **95** 个
真实策略状态，执行 **760** 条后续轨迹，使用 `gamma=0.99`、`reward_scale=0.1`。

| 项目 | 原奖励（水平权重 12） | no_route（水平权重 0） |
| --- | ---: | ---: |
| 同时存在成功与失败候选的状态 | 32 | 32 |
| 最佳成功优于最佳失败 | 32/32 | 32/32 |
| 所有成功/失败配对中成功回报更高 | 272/272 | 272/272 |
| 最小成功回报优势 | 8.504 | 8.166 |

原始起点的 20 个 seed 均由诊断航路候选实现成功。这是已知地图的脚本控制结果，
不是学习策略的成功率。剩余 63 个状态没有采到成功候选，其中 **35 个状态**即使忽略
障碍和加速度，按最大垂直速度纠正高度也已来不及，因此不能把这些失败都归因于奖励。

seed 0 原始起点的例子（训练尺度的折扣回报）：

| 后续行为 | 结果 | 原奖励 | no_route |
| --- | --- | ---: | ---: |
| 诊断航路入球 | success | 11.155 | 6.849 |
| 当前 Actor | timeout | -0.889 | -4.241 |
| 制动停留 | timeout | -3.088 | -3.088 |
| 直接朝目标飞 | collision | -19.238 | -20.750 |

在 32 个可比较状态中，最佳成功候选有 3 个前 1 秒回报低于 Actor，13 个前 2 秒回报
低于 Actor。这提示短期信号与长期结果可能存在差异，不能据此证明 Critic 已经学错，
也不能把所有差异归因于末端速度惩罚。

完整 Actor 轨迹共有 3425 个非终止步，其中 24 步出现“三维距离增加，但推进奖励为正”；
这 24 步计入其他惩罚后的总奖励都不是正数。因而有局部推进目标不一致的现象，
但目前没有充分证据证明它是失败的主因。未发现采样轨迹上的“失败整体比成功更划算”。

**结论：no_route 是有边界的训练消融，不是已经证明能改善成功率的修复。**
它保留了本次采样中的成功/失败排序，并移除水平辅助目标；是否改善三维接近的学习，
需要下一轮训练检验。删除水平项同时降低了推进奖励总量，不能把训练变化完全解释成
方向变化，且不能跨两种奖励直接比较 episode return 的高低。

## 检查范围

本次不启动训练。使用现有 Terminal-v2 `model_final`，通过生产 CLI 构建原验证森林，
加载原 Actor，固定 seeds 0–19。每个 seed 先执行完整策略回合，并逐一对照上一轮正式
评估的终止原因、步数、最终三维距离；容许距离浮点误差为 1e-4 m。

分叉点包括：原始起点、3 秒、6 秒、最后终止前的最近目标状态、首次 XY 进入 1.2 m
的状态（如存在），重合点合并。每个候选从 reset 重放原策略动作前缀，再校验位置、
速度和完整观测历史。保留原来的剩余时间，不把末端超时状态改成全新的 12 秒任务。

比较 8 类后续行为：原 Actor、直接快速入球、按原引导速度曲线入球、先纠正高度再入球、
仅沿 XY 接近而保持分叉高度、向引导区外锚点移动、制动停留、沿诊断规划航路入球。
诊断规划器只用来生成可执行候选，使用已知树木几何的膨胀二维网格路径，不能代表策略
的能力，也不进入训练。每条轨迹仍由生产 K1、扫掠碰撞检测和原终止规则决定结果。
若规划失败，则回退到直接入球，并在结果中标记 `planner_path_available=false`。

每条物理轨迹同时计算两个奖励：

- baseline：水平路线推进权重 12，三维距离推进权重 8。
- no_route：水平路线推进权重 0，三维距离推进权重 8。

其他奖励项相同。逐步验证两种奖励差恰好是当步水平推进奖励；终止步奖励差为零。
折扣总回报采用训练实际 `gamma` 和 `reward_scale`，不把未经缩放的整局奖励之和当作 Q。
另外分别记录前 1 秒和前 2 秒回报，避免混淆短期代价与最终成功的回报排序。

仅有成功和失败候选同时存在的状态才参与成功/失败排序。没有采到成功轨迹不代表
证明了任务不可达。另报告乐观垂直时间下界 `(abs(z_error)-0.2)/0.3` 已超过剩余时间
的状态；该下界还忽略了加速度、水平移动和障碍。

冻结策略的分叉回报不能证明 Critic 的排序、不能证明删除水平项会提升训练成功率。
这些结果用于排除明显奖励冲突，并为单变量训练消融提供依据。

## 正式产物与复现

正式产物目录：`analysis_outputs/spatial_terminal_consistency_formal_20260924/`。
早期 smoke / forest 试跑用于调试；其中诊断课程设置与正式验证地图不一致，不纳入结论。

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.audit_spatial_terminal_consistency \
  --output-dir analysis_outputs/spatial_terminal_consistency_formal_repeat \
  --reference-eval runs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_guidance_coverage_eval/evaluation_metrics.jsonl

/home/tequila/miniconda3/envs/drones/bin/python -m scripts.summarize_spatial_terminal_audit \
  analysis_outputs/spatial_terminal_consistency_formal_repeat
```

`summary.json` 保存状态定义、checkpoint hash、排序和各类候选结果；
`rollouts.jsonl` 保存逐轨迹回报与奖励分项；`traces.npz` 保存逐步物理状态和奖励；
`comparisons.json` 与图片由第二条命令生成。

## 新配置与训练命令

新模式：`upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1`。

相对 Terminal-v2，唯一学习参数变化为 `route_progress_reward_weight: 12 -> 0`。
保留 `goal_progress_reward_weight=8`、原三维终端速度引导、碰撞/成功/超时奖励、
TD3、h8、K1、rayfix、圆盘动作表示、训练与验证地图、课程、探索和 replay 配置。
日志和 checkpoint 写入独立目录。额外开启的引导覆盖统计只用于评估观测，不参与训练。

从头训练，与原 Terminal-v2 的 seed 42、120k 步预算对照；本次没有执行这条训练命令：

```bash
cd /home/tequila/UAV-Latent-RL
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 \
  upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1 \
  --seed 42 --total_steps 120000
```

输出：

- `runs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1/`
- `checkpoints/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1/`
- `analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1_terminal_coverage/`

每次周期评估写入引导区进入回合比例、区内状态比例、成功转移数量、最小三维距离，
同时写入 `evaluation_metrics.jsonl`、TensorBoard 的 `eval/terminal_coverage/*`、
以及 `terminal_coverage.jsonl`。这些是评估轨迹统计，不是训练主 replay 的逐步覆盖统计。

训练后做同一组 20-seed 独立评估：

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 \
  upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1 -- \
  --evaluate_only true \
  --load_checkpoint checkpoints/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1/model_final \
  --eval_interval 1 --eval_episodes 20 --eval_gui false --eval_stepsleep false \
  --log_dir runs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1_formal_eval \
  --trajectory_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1_formal_eval_trajectory \
  --collision_diagnostic_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1_formal_eval_collision \
  --timeout_diagnostic_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1_formal_eval_timeout \
  --eval_terminal_coverage_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1_formal_eval_coverage
```

主要比较：入区回合比例、成功率、接近目标时的高度误差、碰撞率、最终三维距离。
如果改善，再用多个训练 seed 验证；单个训练 seed 的 20 局评估不等于多个独立训练复现。

## 验证

- 23 个相关测试通过，另有 9 个子测试。
- 配置对比测试确认：除独立输出路径/评估日志开关外，只有水平推进权重发生变化。
- 奖励测试确认：非终止步只去除水平推进贡献，所有终止奖励完全一致。
- 新模式载入旧 checkpoint 完成 2 个 seed 的仅评估启动检查，覆盖日志正常输出。
  这是配置接线验证，不是新策略训练或新策略效果评估。
- 图表已生成并检查可读性；未启动用户负责的训练。
