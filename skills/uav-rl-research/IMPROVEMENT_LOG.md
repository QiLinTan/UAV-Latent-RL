# UAV RL Research Skill 改进记录

## 2026-05-29

- 将 `SKILL.md` 从英文改为中文，保留原有项目级 RL 研究工作流。
- 新增 TensorBoard `.tfevents` 分析规范，明确事件文件定位、scalar tags 分组、曲线判断和多 run 对比标准。
- 新增 skill 自身改进记录要求：后续每次改进都要更新本文档，并在回复中给出可复用指令。

## 2026-06-03

- 对新环境下的 `latent_only_action_aux_dim16`、`latentonly_route_tree`、旧版 semantic latent 和 nolatent TensorBoard 事件进行了对比分析。
- 结论更新：action-conditioned auxiliary 比静态 semantic 版本健康，峰值奖励和最优距离能接近新环境 latentonly，但前期/中期推进能力、尾段稳定性和终点收敛仍不如新环境 latentonly baseline。
- 识别出上一版 action-aux 的主要问题：辅助目标过于局部，`next_lateral`、`next_min_range`、一步 `delta_goal` 很快被拟合，后期 loss 接近 0，无法持续塑造对 actor 有用的长期决策语义。
- 明确后续 latent 改进方向：不再继续堆短期预测头，转向弱权重、晚启动的 future-affordance 表征，让 latent 更关注未来推进、未来安全距离、危险概率和接近终点概率。
- 代码结构调整：保留原版 `TD3LatentOnly` 作为可复现实验 baseline，不再把实验辅助头写入 `algos/td3/td3_latent_only.py`。
- 新增 `algos/td3/latent_aux_heads.py`，用于放置通用辅助头网络，避免主算法文件膨胀。
- 新增 `algos/td3/td3_latent_affordance.py`，实现 `TD3LatentAffordance`，包含 one-step + target-head bootstrap 的 future-affordance 辅助目标。
- 更新 `scripts/change_td3.py`：增加 `--latent_only_variant base|affordance`，默认 `base`，确保原版 latentonly 不会被实验算法悄悄替换。
- 更新 `scripts/run_change_td3.py`：`latent_only` 现在跑原版新环境 baseline；`latent_only_affordance` 跑新版 affordance 实验。
- 更新 logger/monitor：记录 affordance loss、future clearance、danger probability、near-goal probability、bootstrap scale 等指标。
- 验证项：`python3 -m py_compile` 通过；随机 replay buffer 下 `TD3LatentOnly` 和 `TD3LatentAffordance` 均可训练一步、保存和加载；`git diff --check` 通过。
- 推荐运行命令：
  - 原版新环境 latentonly baseline：`python3 -m scripts.run_change_td3 latent_only`
  - 新版 affordance 实验：`python3 -m scripts.run_change_td3 latent_only_affordance`

## 2026-06-06

- 新增集中式森林观测布局定义，统一 `KIN + 动作历史 + 目标 + 雷达` 的拼接、动态切片和维度校验，避免继续硬编码 `252/255/263`。
- 修复 future-affordance 雷达索引错位：旧实现从 `state[15:23]` 读取了动作历史；新实现始终从观测尾部读取 3 维目标和 8 维雷达，并支持后续动作历史长度消融。
- 增加观测诊断日志：动作历史维度、各分区幅值、目标范数、当前/下一状态雷达最小值和最大值、雷达越界比例。
- 增加 affordance 标签诊断：各 bootstrap 标签标准差、即时危险率和即时成功率，用于识别标签越界、塌缩或正例稀缺。
- 保持 TD3、reward、网络规模和训练参数不变；本轮只修数据布局与可观测性。
- 验证项：观测布局与 affordance 尾部雷达回归测试共 5 项通过；随机 replay buffer 可训练一步并输出新日志；真实环境观测仍为 263 维且拆分为 `12/240/3/8`；`py_compile` 和 `git diff --check` 通过。

## 2026-06-08

- 将 AvoidBench 接口从不存在的 `flightgym.AvoidVisionEnv_v1` 改为实际的 `avoidbridge.AvoidbenchBridge`，补充 Python adapter 和 probe 脚本。
- 新增 `getUnityDepthImages()`，在 `perform_sgm: false` 时返回 RGB `uint8` 和 Unity depth `float32`，同时保留原 stereo SGM 接口。
- pybind11 图像返回改为 NumPy 自有内存副本，避免引用已释放的 `cv::Mat`。
- 定位并修复 Python 退出时的 `double free`：根因是 `quadrotor_common` 与 avoidlib 的 C++/Eigen ABI 对齐不一致。
- 使用 Python state proxy，并在 `updateUnity()` 中临时构造原生状态；同时将 avoidlib Eigen 对齐固定为 16 字节。
- 单目 Unity-depth 模式不再创建无用的 CUDA SGM 对象。
- 验证项：Unity 连接、障碍物生成、场景变化、RGB/depth、碰撞检测和吞吐率正常；probe 返回 `EXIT_CODE=0`，项目测试 `13 passed`。
