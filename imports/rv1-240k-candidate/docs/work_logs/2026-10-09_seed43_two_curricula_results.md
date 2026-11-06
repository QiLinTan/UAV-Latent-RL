# seed 43 两种训练机制结果：Stage 0/1 微调与从零障碍课程

## 判断

两个实验都比原冻结 seed 43 策略前进了一步，但解决的问题不同。`stage01_ft_v1` 的选中模型把 Stage 1 的 20 张固定验证地图从 16/20 提到 20/20，原始 Stage 0 保持 20/20；在未参与选模的 seeds 20–39 上，Stage 0/1 也都是 20/20。不过它没有学过 Stage 2/3，held-out Stage 3 只有 6/20，大量失败发生在已经进入目标附近之后。`startobscurr_v1` 从零经历起点课程和 Stage 0→3 障碍课程，选中模型在 held-out 随机五树 K1 Stage 3 达到 **18/20**，Full Stage 3 达到 **14/20**；它解决了大部分复杂障碍导航，但 Full 链路仍额外暴露局部振荡碰撞。

这是**单个训练 seed 43** 的结论。held-out seeds 20–39 没有参与周期评估或 checkpoint 选择，但使用同一个随机地图生成器；这些结果不能证明跨地图分布泛化，也不能证明四个独立训练 seed 都会成功。两种训练机制的初始权重、训练步数和课程分布均不同，不能把两者的差异归因于单一超参数。

## 评估口径与所选网络

原冻结网络是 `checkpoints/startcurr_v1_frozen_20261009/seed43/model_final`。微调网络来自它，训练 40k 步，选择 28k 的稳定 `model_best`；从零课程训练 240k 步，选择 Stage 3 的 240k 稳定 `model_best`。两个 `model_final` 的 Actor 参数与各自 `model_best` 逐张量完全相同；文件 SHA-256 不同来自序列化差异。两份最终网络及优化器已复制到 `checkpoints/seed43_course_selected_20261009/`，`manifest.json` 记录源、选中步和 10 个文件哈希。

正式复核统一使用原始 `[-3.5, 0, 1]` 起点、三维 0.2 m 成功球、K1 或 Full、随机验证地图 `procedural_forest_multitree_validation_random_v1`，环境 seeds 20–39 各 20 局，关闭简单阶段混合。物理 reset 逐一验证 Stage 0/1/2/3 都分别为 1/2/3/5 棵挡路树，初始目标距离 7 m。三个模型在适用对照中使用同一批地图种子。每个评估目录的 `protocol.json` 记录实际 checkpoint SHA-256、动力学、阶段和环境种子；`evaluation_metrics.jsonl` 逐局记录种子与终止原因。

## held-out 20 张地图结果

| 动力学与障碍阶段 | 原冻结 seed 43 | Stage 0/1 微调 | 从零完整障碍课程 |
| --- | ---: | ---: | ---: |
| K1 Stage 0 | 20/20 | 20/20 | 20/20 |
| K1 Stage 1 | 17/20 | **20/20** | **20/20** |
| K1 Stage 2 | 未测 | 11/20 | **20/20** |
| K1 Stage 3 | 3/20 | 6/20 | **18/20** |
| Full Stage 0 | 未测 | 未测 | **20/20** |
| Full Stage 3 | 2/20 | 未测 | **14/20** |

Stage 0/1 微调在 Stage 1 的三次原冻结失败地图（21、35、38）全部通过，20 局没有碰撞或超时。Stage 2 的 9 次失败为 4 碰撞、5 超时；Stage 3 的 14 次失败为 4 碰撞、10 超时。Stage 3 的 20 局均曾进入 1.2 m 引导区，10 个超时按现有启发式诊断为 T1 5 次、T3 5 次；这说明其高阶段主要新增了近目标收敛和高度耦合问题，不能把 6/20 简单称为“完全不会避障”。这些超时标签不是因果证明。

完整障碍课程在 K1 Stage 3 的两次失败是环境 seeds 26、33 的碰撞，窗口分类均为 C（局部指令振荡）；原冻结 K1 Stage 3 的 16 次碰撞则分类为 A 11 次、C 5 次，另有 1 次超时。新策略把“继续直冲障碍”的大类失败显著减少。K1 Stage 3 从 3/20 提至 18/20，同一批地图上的入 1.2 m 引导区比例从 4/20 提至 18/20。

同一新策略从 K1 换成 Full 后，Stage 0 仍 20/20，Stage 3 从 18/20 降到 14/20。Full Stage 3 的 6 次失败全是碰撞，诊断均为 C；其中 seeds 26、33 与 K1 重合，21、24、30、36 只在 Full 失败。Full Stage 3 平均位置/速度跟踪 RMSE 约 0.034 m / 0.148 m/s，Full Stage 0 为 0.022 m / 0.090 m/s。这个对照表明 Full 的控制与状态演化会在复杂障碍处进一步降低安全裕度，但不能单凭 RMSE 判定控制跟踪误差是这四次新增碰撞的唯一原因。原冻结模型在相同 held-out Full Stage 3 只有 2/20，说明新模型 Full 的 14/20 仍是很大的真实改进。

## 训练过程稳定性

微调训练在 step 0 的固定 seeds 0–19 上，Stage 0 为 20/20、Stage 1 为 16/20；28k 选中模型两阶段均 20/20。训练并非单调改善：12k 时两个阶段均 0/20，全部 timeout；16k 恢复，40k 原始末期网络 Stage 1 又跌到 17/20（3 次碰撞），恢复 28k checkpoint 后才是最终 20/20。Stage 0 在其余周期评估均为 20/20，选模的 Stage 0 保留门槛避免了把 12k 崩溃网络作为最佳模型。训练回合实际 247 局，其中一树 121、二树 126；在探索噪声下，二树回合成功率约 79.4%、碰撞约 5.6%，不能与确定性正式评估直接比较。

从零课程在固定 seeds 0–19 上，约 96k 进入 Stage 1、116k 进入 Stage 2、132k 进入 Stage 3。Stage 3 在 132k 曾达到 20/20，但 164k 降到 11/20，后又回升，236k 和 240k 都是 20/20；选中模型来自 240k。240k 的高分不代表训练机制完全稳定。Stage 3、原始起点、五树主难度的训练回合有 576 局；其中 180k–240k 的 314 局在探索噪声下成功率约 93.6%、碰撞约 3.2%，近 24k 的 119 局成功率约 87.4%。这些训练数据说明策略确实反复面对五树，而非只通过简单障碍混合拿到验证成绩。

## 下一步

两份所选模型已经冻结。优先把完整障碍课程在独立训练 seeds 42、44、45 上重复，并沿用本次未参与选模的 Stage 0–3 K1 / Full Stage 3 评估；原实验设计的工程门槛是至少 3/4 个训练 seed 在 held-out K1 Stage 3 达到 50% 成功率。一个 seed 43 达到 90% 只能算首个正例。若其他 seed 也达标，再单独针对 Full Stage 3 的 C 类振荡和 K1→Full 差距做控制或训练微调，不要同时重新改奖励、观测和课程。Stage 0/1 微调可以作为“从好策略继续训练且保留 Stage 0”的证据，但它自身在 Stage 3 只有 6/20，不宜直接宣称完成复杂森林导航。

## 产物

- 两份所选模型及哈希：`checkpoints/seed43_course_selected_20261009/manifest.json`
- 所有 held-out 评估与逐局终止原因：`analysis_outputs/seed43_course_holdout_20261009/`；统一机器可读对照为其中的 `comparison.json`
- 可复用评估入口：`scripts/evaluate_seed43_course_holdout.py`（现有同名输出不会覆盖）
- 周期评估：`runs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1_seed43/` 与 `runs/upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1_seed43/`
