# K1 Stage 0/1 冻结策略微调最小实验

本实验从已冻结的四个 `startcurr_v1` 成功 checkpoint 各自初始化，在原始 7 m 起点上训练。只检验“Stage 1 两棵随机挡路树能否改善，同时保住 Stage 0 一棵树的成功率”。Stage 2/3 暂不参与训练或选择，Full 也不参与。此前从零开始的 `startobscurr_v1` 保留为独立实验，不作为本轮第一步。

## 基准与唯一训练分布变化

源模型在 `checkpoints/startcurr_v1_frozen_20261009/seed{42,43,44,45}/model_final`，清单 `manifest.json` 保存各文件 SHA-256；启动器每次会核验所选 seed 的全部五个文件。此前 20 张环境地图上的零样本 K1 结果：Stage 0 四模型均 20/20；Stage 1 依次为 17/20、16/20、16/20、20/20。此前正式评估在训练地图族上强制关闭简单阶段混合；新随机验证地图与其起终点、障碍生成配置相同，唯一差异是地图默认的简单阶段混合概率为 0。训练开始前会在**新验证地图**分别重测两个纯阶段，形成严格同协议的 step 0 基线。

训练地图锁在 Stage 1；每个 episode 以 0.5 概率选 Stage 0（1 棵挡路树），其余选 Stage 1（2 棵挡路树）。这是训练分布混合，**并非在同一条轨迹中切换阶段**。训练仍从原始 7 m 起点开始。40 个不同 reset seed 的物理检查得到一树 20 次、二树 20 次，观测全部 917D。地图的 `curriculum_simple_mix_probability` 会覆盖构造参数，所以训练环境构造后显式设为 0.5；评估环境单独强制 Stage 0 或 Stage 1，并使用验证地图默认的 0 混合。

奖励仍为 `terminal_guidance_v5`，水平 route progress 权重仍为 0；rayfix、圆盘动作、917D GRU 观测、TD3、H8、K1、success radius 和碰撞惩罚均沿用原模式。Checkpoint `load()` 恢复 Actor、Critic 及两个优化器，target 网络由加载后的网络复制；Replay 与训练步数重新开始。因此这是从冻结策略微调，不是恢复原训练时的完整 Replay。`start_timesteps=0` 让已训练 Actor 从首步执行并加入原有探索噪声；仍到第 5,000 步才开始梯度更新，以先收集当前 Stage 0/1 数据。原有 TD3 更新算法和 reward 不变。

## 评估与选择

每 4,000 训练步，用环境 seeds 0–19 各执行 20 局纯 Stage 0 和 20 局纯 Stage 1，起始 step 0 也评估一次。两个日志分别在 `runs/<mode>_seedN/stage0_retention/evaluation_metrics.jsonl` 与 `runs/<mode>_seedN/evaluation_metrics.jsonl`，后者为 Stage 1。记录成功、碰撞、引导区、回报及轨迹；`eval/curriculum_stage` 在两个日志中分别为 0 和 1。成功定义仍是进入三维 0.2 m 球。

Checkpoint 选择以 Stage 1 为主，但只有同一步 Stage 0 达到 20/20 才允许将网络列为 peak 或稳定 best；稳定 best 仍要求连续两次 Stage 1 成功率至少 80%、碰撞率至多 20%。如果整个实验没有满足 Stage 0 保留条件的候选，`model_final` 回退为原冻结模型，`model_final_selection.json` 记录源路径。注意 `model_peak_validation` 只来自单次评估，属于候选；最终报告应同时检查两个阶段的 20-seed 明细，不应只看 `model_final` 文件名。

初步有效性标准：在同一批地图上，Stage 1 相对各自 step 0 基线提高，且 Stage 0 仍 20/20、碰撞不增加；四个训练 seed 的趋势比单个 seed 更有说服力。seed 45 的 Stage 1 基线已是 20/20，它主要作为保持性能与训练稳定性的对照。若 Stage 1 没改善，先分析训练数据中的阶段比例、两阶段独立成功率和碰撞/价值学习，再决定是否改变混合比例或引入下一阶段。只有本实验有效，才在保留该模型与相同隔离评估办法的基础上逐级推进 Stage 2、Stage 3。

## 运行

建议先跑一个信息量较大的 seed 43（原 Stage 1 为 16/20）：

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1 --seed 43 --total_steps 40000
```

如要四个训练 seed 顺序运行，在仓库根目录执行完整循环（seed 43 若已跑完可从列表中删除）：

```bash
for seed in 42 43 44 45; do
  /home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1 --seed "$seed" --total_steps 40000 || break
done
```

正式训练命令会按 seed 写入独立的 `runs/`、`checkpoints/` 和 `analysis_outputs/` 目录，不修改冻结源文件。每个 seed 结束后，可将两个 `evaluation_metrics.jsonl` 和 `model_final_selection.json` 连同 20-seed episode 明细进行前后对照。

## 已完成的校验

- 新模式参数与原 `noroute` 模式的 reward、K1、网络和 TD3 设置逐项核对一致。
- 物理 reset：原始起点、917D、Stage 0/1 均被采样。
- 使用 seed 43 冻结模型完成 1 步 `/tmp` 冒烟运行，走通加载、step 0 双阶段评估、周期双阶段评估、checkpoint 选择/恢复和最终评估；这**不是** 40k 微调结果，也没有梯度更新。
- 相关 6 个测试文件合计 42 项通过；另以 0 步完整运行验证 Stage 0/1 日志标注和冻结源模型回退。
