# `full_ft_v1`：K1 Actor → Full 在线微调实验契约

## 已实现的固定条件

- 源模型：`checkpoints/startobscurr_v1_selected_20261010/seed{42,43,44,45}/model_final_actor`。启动器逐 seed 核对冻结清单 SHA-256；目标目录独立为 `checkpoints/full_ft_v1_seed{seed}`，源文件不写入。
- 仅加载源 Actor，复制到目标 Actor。Twin Critic、目标 Critic、两个优化器和主/成功 Replay 均新建。成功 Replay 仅从本次 Full 成功回合填入。
- 复用 `TD3Trainer` 与 `TD3UpperRecurrent.train()`。初始 `start_timesteps=0`，由预训练 Actor 加投影后的探索噪声采样；从 512 个 Full 转移开始更新 Critic，前 4000 个 Full 上层转移不更新 Actor 或目标 Actor。之后使用原 TD3 延迟 Actor 更新。目标 Critic 在冻结期仍按 TD3 周期软更新。
- 原始 917D 观测、三维圆盘一致动作、15Hz 上层/120Hz 下层、0.8m/s 水平参考上限、原奖励、12 秒 episode、0.2m 成功半径。没有近目标随机 reset、规划器示范或 K1 Replay。
- 原始 7m 起点；每十局打乱后精确抽取三局 Stage 0、七局 Stage 3。地图配置本身的 20% 简单难度混合在环境构造后显式关闭，并在每局 reset 校验 Stage 0/3 分别使用 1/5 个路线障碍。
- 每 5000 个上层转移在验证地图环境 seeds 80–89 分别测 Full Stage 3、Full Stage 0、K1 Stage 3。Full Stage 3 为主指标；模型选择另要求 Full Stage 0 成功率至少 80%。保留 `model_initial`、`model_raw_final`、`model_peak_validation`、符合窗口要求时的 `model_best`，以及恢复所选策略后的 `model_final`。Stage 0 保留阈值只是选择规则，不代表已验证最优阈值。
- 训练每局保存 15Hz Actor 原始动作、执行动作、ReferencePacket、相对目标的侧向转向符号、净空及 120Hz 实际状态/姿态/RPM；验证保存 Full Stage 3/0 轨迹及碰撞窗口。训练轨迹在 `analysis_outputs/full_ft_v1_seed{seed}/training_traces`；验证轨迹在对应 `runs/full_ft_v1_seed{seed}` 子目录中按 step 分组。

## 开跑前证据

四个冻结源 Actor 已在同一 Stage 3 环境 seed 80 分别做 K1/Full 零样本推理接口审计；两模式初始 917D 观测、首个 Actor 输出和首个实际速度指令逐项差值均为 0，Full 均采集到 120Hz 轨迹。审计数据：`analysis_outputs/full_ft_v1_source_audit_20261010/summary.json`。这是**单张地图的接口一致性审计**，不是成功率估计。另有 `tests/test_full_ft_v1.py` 验证 Actor-only 加载、Critic 冻结预热、3/7 阶段表和三路评估汇总。

临时目录中的两步集成冒烟在验证 seeds 80–89 测得源 seed43 的 Full Stage 3 = 8/10 成功、2/10 碰撞，Full Stage 0 = 10/10 成功，K1 Stage 3 = 9/10 成功。这些是训练前验证地图上的数字，不属于保留的 40–79 正式评估，也不能推断微调收益。

## 首轮训练与监控

依次运行，先看 seed43 的 5k/10k/15k 验证趋势，再决定是否让 seed45 继续。以下每条命令可单独粘贴；默认预算 40k 上层转移。

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 full_ft_v1 --seed 43 --total_steps 40000
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.run_change_td3 full_ft_v1 --seed 45 --total_steps 40000
```

若首轮稳定且仍在改善，可把**新的独立运行**预算改为 60000；本预设尚不支持从中断点恢复，所以不要把重新运行误当作接着训练。每 5k 步观察三路成功/碰撞、`actor_frozen`、`actor_updated`、Critic loss/Q gap、Full Replay 和成功 Replay 大小。若 Full Stage 3 连续下降或 Stage 0 明显遗忘，先检查这些日志、碰撞前的轨迹和 Actor 更新，再判断是否值得加长预算。当前没有自动宣称 Full 微调有效的依据。

## 正式成对评估

环境 seeds 20–39 只用于已完成的故障诊断；seeds 40–79 保留为最终 held-out，**不要用于超参数选择**。训练前后采用同一验证地图与相同 seeds，分别比较 Full Stage 3、Full Stage 0、K1 Stage 3 的每张地图结果。训练结束后运行：

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.evaluate_full_ft_v1_heldout --seed 43 --phase both --finetuned-checkpoint checkpoints/full_ft_v1_seed43/model_final
/home/tequila/miniconda3/envs/drones/bin/python -m scripts.evaluate_full_ft_v1_heldout --seed 45 --phase both --finetuned-checkpoint checkpoints/full_ft_v1_seed45/model_final
```

输出在 `analysis_outputs/full_ft_v1_heldout_20261010/seed{seed}`。脚本锁定 40–79、源 Actor 哈希与实验协议，给出逐图 success/collision/timeout 和源→微调翻转；Full Stage 3 保留逐局 15/120Hz 轨迹。再结合 `model_raw_final` 与 `model_final` 比较，可以判断后期回退和 checkpoint 选择的影响。确认 seed43/45 的独立收益与 Stage 0 保留之后，再以相同流程扩展 seed42/44。

此预设中的 4000 步冻结、512 步 Critic 起训、探索噪声 0.15→0.05、3/7 阶段比和 40k 步预算都是**待验证的实验设定**，不是项目中已经证明最优的参数。
