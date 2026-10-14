# Full 改进前的三项定向实验：P0、B、A3

本轮仅重放四份冻结 checkpoint 并运行真值辅助诊断控制器；没有训练、修改奖励或改变 917D Actor。P0 选取 seed 42/map 34、seed 45/map 20 和低净空 seed 42/map 32 三张代表图；B 检查 seed 45 的 Full Stage 0 地图 25、30；A3 检查同命令 Full 仍失败的 42/24、43/21 和命令带耗尽的 43/24。正式评估仍采用原 12 s、0.2 m 成功判定。P0/B 的原始记录和计算汇总见 [analysis_summary.json](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/analysis_summary.json)，A3 正例见 [summary.json](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/a3/summary.json)。

## P0：首次持续命令分叉及其输入

重跑三对原 K1/Full 闭环，逐步保存 Actor **决策前**的完整 917D 输入。定义首次持续分叉为 K1 与 Full 三维命令差 `>0.2 m/s` 连续三个 15 Hz 决策步的起点；也保留第一次单步越阈的时刻。当前观测布局为：12D 动力学、3D 目标、96D 当前世界坐标系雷达（3 个仰角 × 32 个方位）、6D 参考状态，以及 8×100D 的雷达/命令/误差历史。冻结 Actor 直接从保存的输入重算动作，与评估原动作逐点一致。

| 案例 | 首次单步 / 持续分叉 | 持续分叉前 K1–Full 位置 / 速度差 | 当前雷达差（96 射线） | 平面命令差异 |
| --- | --- | --- | --- | --- |
| 42/34 | 第 8 / 22 步，持续分叉约 1.467 s | 0.129 m / 0.104 m/s | 23 条差值 >0.05；最大 0.617（水平层 78.75°） | 同为 0.8 m/s，方向差 52.4° |
| 45/20 | 第 5 / 5 步，约 0.333 s | 0.069 m / 0.208 m/s | 10 条差值 >0.05；最大 0.440（水平层 0°） | 同为 0.8 m/s，方向差 11.9° |
| 42/32 | 第 5 / 5 步，约 0.333 s | 0.067 m / 0.182 m/s | 6 条差值 >0.05；最大 0.452（水平层 348.75°） | 约 0.8 m/s，方向差 31.8° |

因此这三次初始分叉**主要是速度方向变化，不是单纯减速**。42/34 的目标横向指令 K1/Full 分别为 +0.780/+0.335 m/s；45/20 为 +0.567/+0.672 m/s，仍在目标路径同一侧；42/32 为 −0.019/+0.405 m/s，首次分叉时 K1 近乎切向零值，不能把该瞬间直接判成可靠的“换边”，但随后两条路线的横向指令明显不同。雷达差异发生时已经存在 0.067–0.129 m 的位置差和 0.104–0.208 m/s 的速度差；本轮不能把这些雷达变化称为独立于 Full 执行误差的外生原因。

为进一步定位 Actor 对输入的敏感性，在**不运行环境**的情况下，把 K1 输入中的某一组数值替换到 Full 的同一决策输入，重新计算冻结 Actor。这个混合输入可能不对应真实物理状态，只用于网络敏感性诊断。表中是混合动作与原 Full 命令的三维速度差，括号内是与原 K1 命令还相差多少。

| 案例 | 换当前雷达 | 换历史雷达 | 当前+历史雷达一起换 | 只换当前动力学 |
| --- | ---: | ---: | ---: | ---: |
| 42/34 | 0.398（余 0.328） | 0.449（余 0.276） | **0.707（余 0.001）** | 0.066（余 0.648） |
| 45/20 | 0.111（余 0.110） | 0.072（余 0.169） | **0.158（余 0.070）** | 0.018（余 0.203） |
| 42/32 | 0.046（余 0.481） | 0.245（余 0.199） | **0.343（余 0.099）** | 0.060（余 0.392） |

数字单位均为 m/s。42/34 的命令分叉几乎完全可以由**当前与历史雷达输入差异**在冻结网络上的作用重现；42/32 的历史雷达比当前雷达更敏感；45/20 雷达也有较大作用，但不能单独解释全部差异。这不证明雷达传感器错误，也不证明应该删掉历史：不同执行运动会改变真实可见几何，且混合输入不保持特征之间的物理一致性。

## P0：相同 Full 状态的短窗口干预

在上述持续分叉的决策步重新运行原 Full 闭环，只把接下来 **8 个 15 Hz 上层三维速度命令**替换为相同时间索引的 K1 成功命令，随后恢复 Full Actor。干预和基线在替换前的全部 917D 输入与 120 Hz 位置逐点相同，替换命令与 K1 来源 tape 逐点相同。因此干预起点的 Full 物理状态、参考状态和观测历史一致；这里不把不同 K1/Full 状态直接当控制器对照。

| 案例 | 原 Full → 干预后 | 干预后前 8 步最小净空（原 Full → 干预） | 解读 |
| --- | --- | --- | --- |
| 42/34 | 碰撞 → 碰撞 | 0.389 → 0.376 m | 首个 8 步替换不足以修复；不能排除更后面的闭环差异 |
| 45/20 | **碰撞 → 成功** | 0.698 → 0.674 m | 早期方向差能改变整条后续闭环轨迹；短期净空并未提高 |
| 42/32 | 碰撞 → 碰撞 | 0.589 → 0.552 m | 替换初段反而降低局部净空；本图 K1 原路线仅约 0.013 m 净空 |

45/20 的成功是同状态命令干预的因果结果，但**不是“立即增大净空”**：前 8 步及后续 15 步的最小净空都略低于原 Full。它更像一次早期路线/状态反馈分叉的长期效应。42/34 的固定**全长** K1 tape 在 A2 可成功，而这里只替换 8 步仍碰撞；说明不能用一次短干预否定后续决策的作用。三例结果不可外推为其余六张固定 tape 可成功地图的比例。

## B：seed 45 的 Full Stage 0 近目标 timeout

同一冻结策略、地图和控制参数分别允许名义 12、14、16 s；三条重放在原 12 s 终止前的 120 Hz 位置、速度逐点相同。环境在 `elapsed > EPISODE_LEN_SEC` 时截断，因此名义 12 s 的实际最后采样为 12.017 s。目标成功仍按三维 `d<0.2 m`。

| 地图 | 首入 1.2 m | 原终点三维距离 | 原终点径向 / 切向实际速度 | 扩时首次入 0.2 m |
| --- | ---: | ---: | ---: | ---: |
| seed 45/map 25 | 8.425 s | 0.2041 m | +0.286 / 0.177 m/s | **12.033 s**，名义扩时后成功 |
| seed 45/map 30 | 8.358 s | 0.2175 m | +0.294 / 0.165 m/s | **12.083 s**，名义扩时后成功 |

两例均已进入引导区，并在原截断时继续朝目标接近；14 s 和 16 s 设置得到相同的入球时刻，不存在扩时后绕圈很久才偶然成功的迹象。相对实际 12.017 s 终止采样，只多 **0.017/0.067 s** 即成功。这个失败模式在 Stage 0 就存在，因而独立于复杂 Stage 3 绕障；对这两例，时限裕度是直接因素。不能据此扩大正式评估时限，也不能推论所有近目标 timeout 都只是时限问题。原正式 Full Stage 0 仍为 78/80。

## A3：三张地图的 Full 动态可达性正例

使用环境真值树木几何规划二维航路点，并用简单比例/速度反馈输出上层三维速度命令；全部命令仍由**原 Full Reference Generator → DSLPID → 120 Hz PyBullet 动力学**执行，保留原 Stage 3 随机地图、12 s、0.2 m 成功球。每次 reset 的起点、目标、树木中心与半径都与 A1 原失败轨迹逐项一致。预设的首个控制器参数（规划膨胀半径 = 碰撞半径 + 0.22 m，航路点前瞻 0.20 m）已经在三张地图成功，不需要扩大搜索。安全净空取完整 120 Hz 实际轨迹的最小值。

| 地图 | 原固定 K1 tape 在 Full | 真值辅助 Full 结果 | 成功时间 | 120 Hz 最小树净空 |
| --- | --- | --- | ---: | ---: |
| 42/24 | 碰撞 | 成功 | 10.8 s | 0.107 m |
| 43/21 | 碰撞 | 成功 | 11.2 s | 0.116 m |
| 43/24 | tape 结束，尚未入球 | 成功 | 10.8 s | 0.107 m |

这为三张图各给出一条**动力学可执行**的安全轨迹，排除了“在当前 Full/DSLPID、12 s 规则下根本无法到达”的解释；不代表无地图 Actor 已会找到这条路线。42/24 与 43/24 是相同环境地图 seed，不同训练 Actor 的旧失败案例；诊断控制器不依赖 Actor，因此两次正例轨迹一致。

## 面向改进实验的决策

现在可以停止扩大原因清单，进入**单变量 Full 适应实验**。P0 表明小幅 Full 状态偏移通过雷达及其历史影响 Actor 方向，45/20 的同状态短干预能改变结局；A3 表明难图有 Full 可执行安全路线。首版可从冻结 K1 Actor 初始化，在 Full Stage 0/3 混合任务上重新采集 Full 经验、重建 Critic 和 Replay，并保留原 917D、奖励、TD3、成功判定与 Reference Generator。先锁定新 held-out 地图作源模型基线，再比较 Full Stage 3 成功/碰撞、Full Stage 0 保留率和 K1 Stage 3 保留率。不能把 B 的扩时结果算进正式成功率，也不能把 A3 真值轨迹当作 Actor 示范成绩。若首版仍反复走厘米级净空路线，再独立测试提高训练安全裕度或雷达历史鲁棒性的改动，避免与 Full 适应同时改动而失去归因。

复核入口：`scripts/run_full_preimprovement_diagnostics.py`（P0/B 重放）、`scripts/summarize_full_preimprovement_diagnostics.py`（输入和前缀断言）、`scripts/audit_full_dynamic_reachability.py`（A3 Full 正例）。所有新增开关只允许 `--evaluate_only true`；原训练过程不使用它们。

图： [P0 42/34](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/p0_seed42_map34.png)、[P0 45/20](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/p0_seed45_map20.png)、[P0 42/32](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/p0_seed42_map32.png)；[B map 25](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/b_seed45_map25.png)、[B map 30](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/b_seed45_map30.png)；[A3 42/24](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/a3_seed42_map24.png)、[A3 43/21](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/a3_seed43_map21.png)、[A3 43/24](/home/tequila/UAV-Latent-RL/analysis_outputs/full_preimprovement_diagnostics_20261010/figures/a3_seed43_map24.png)。
