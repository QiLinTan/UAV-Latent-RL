---
name: uav-rl-research
description: >
  当用户在 UAV-Latent-RL 仓库中处理 TD3、latent/world-model agent、PyTorch
  强化学习训练调试、森林无人机 reward/curriculum/environment、消融实验规划、
  runs/checkpoints/TensorBoard 结果分析时使用。触发关键词包括 TD3、actor/critic
  loss、latent encoder、trust gate、detach/gradient flow、replay buffer、reward
  shaping、curriculum、UAV forest navigation、experiment table、训练不稳定、
  TensorBoard 事件文件、latent vs no-latent 对比。
---

# UAV Latent RL Research Skill

这是 UAV-Latent-RL 的项目级研究工作流。使用本 skill 时，Codex 应该像谨慎的
强化学习研究工程师：先读相关代码，再判断数据流和梯度流，保护实验变量，先做
低成本验证，再建议长时间训练。

## 核心原则

不要把这个仓库当成普通应用代码。这里一行改动就可能改变一个实验结论。始终区分：

- 算法行为：actor、critic、target network、loss、detach、trust gate。
- 环境行为：observation、action scaling、reward terms、termination。
- 实验协议：seed、total steps、curriculum、logging、checkpoint。
- 分析行为：比较的是哪个指标，还剩哪些混杂变量。

## 上下文路由

只读取当前任务需要的上下文，不要一开始吞全仓库。

| 用户任务 | 优先读取 |
| --- | --- |
| TD3 baseline、actor/critic、action selection | `algos/td3/td3_plain.py`、`algos/td3/networks.py`、`data/replay_buffer.py` |
| latent、encoder、trust、world model、detach | `algos/td3/td3_v1trust.py`、`algos/td3/td3_latent.py`、`models/encoder.py`、`models/heads.py`、`algos/td3/grad_utils.py` |
| training loop、callbacks、logging、checkpoint | `trainers/td3_trainer.py`、`trainers/callbacks/`、相关 `scripts/` 启动脚本 |
| UAV forest environment、reward、curriculum、obstacle | `envs/forest/core.py`、`envs/forest/rewards.py`、`envs/forest/curriculum.py`、`envs/preprocess.py` |
| 实验规划或消融 | `experiments/experiment_table.md`、已有 `experiments/*/README.md`、相关 `runs/` 名称 |
| run/result/TensorBoard 分析 | `runs/`、`analysis_outputs/`、`events.out.tfevents.*`、`experiments/*/metrics_summary.txt` |
| 依赖或复现问题 | `requirements.txt`、`requirements_lock.txt`、`environment_full.yml` |

编辑前，用 `rg` 找到被修改类、flag、metric、函数的所有调用点。

## TD3 算法不变量

修改 TD3 前后都要检查：

- `target_q` 必须在 `torch.no_grad()` 下计算。
- target policy smoothing 的 noise 先 clip，再把 target action clip 到
  `[-max_action, max_action]`。
- critic loss 同时使用两个 Q 网络；actor loss 使用 `critic.Q1`。
- actor 和 target network 更新必须尊重 `policy_freq`。
- soft target update 使用 `tau * param + (1 - tau) * target_param`。
- replay sample 顺序是 `(state, action, next_state, reward, not_done)`，并且张量在训练前移动到 agent device。
- gradient clipping 如果保留，优先记录 pre-clip grad norm。
- save/load 必须覆盖影响实验的所有可训练模块和 optimizer。

## Latent / World Model 不变量

涉及 encoder、recon、dynamics、trust gate 时额外检查：

- encoder 输出 shape 与 `latent_dim` 一致。
- 使用 latent 时，actor/critic 输入维度是 `state_dim + latent_dim`。
- `state[:, :12]` 的假设仍然适用于 reconstruction/dynamics head。
- detach、soft-detach、full gradient 的选择必须和实验名一致。
- trust/gating tensor 默认 detach，除非实验明确要让梯度穿过 gate。
- target network 不应意外通过 online encoder 路径反传。
- 日志要包含足够解释 latent 失败的诊断项，而不是只看 return。

## 环境与 Reward 不变量

修改森林无人机环境时检查：

- action shape 与 `env.step(action.reshape(1, -1))` 兼容。
- 改 `max_action` 或 exploration noise 前，确认 action 是 normalized、scaled、RPM-like，还是环境原生动作。
- 除非用户明确要改随机性，否则保留 seed 处理方式。
- reward 改动要暴露到 `reward_terms`，让失败模式可见。
- curriculum 改动要说明控制的是 tree count、obstacle placement、route blocking、start/goal distance 还是 safe distance。
- collision、truncation、success 定义必须能追溯到 env info 或日志。

## TensorBoard 事件文件分析规范

当用户要求分析 `.tfevents`、TensorBoard 曲线、`runs/` 或训练结果时，按这个流程做。

1. 先定位事件文件，不要只凭目录名判断：

```bash
find runs analysis_outputs experiments -name 'events.out.tfevents.*' -type f
```

2. 优先使用 TensorBoard 的 `EventAccumulator` 读取 tags 和 scalar，不手工解析二进制文件。若环境缺依赖，先说明缺什么，再给可运行命令。

3. 先列出可用 scalar tags，再按指标族分组：

- 回报与任务表现：episode return、eval return、success、collision、episode length、goal distance。
- TD3 核心：critic loss、actor loss、actor updated、target Q、Q1/Q2 差异。
- 梯度与稳定性：actor/critic/encoder grad norm、clip 后异常、NaN/Inf。
- 探索与动作：exploration noise、action mean/std/min/max、action saturation。
- reward terms：progress、distance、goal bonus、height、proximity、attitude、collision penalty。
- latent/world model：reconstruction error、dynamics error、trust mean/min/max、latent abs/std/effective scale。
- curriculum/env：curriculum stage、tree count、clearance、route offset。

4. 看曲线时不要只看最后一个点。至少比较：

- 初期是否能离开随机策略水平。
- 中期是否出现平台期、震荡、发散或突然坍塌。
- eval 与 train 是否背离。
- reward 上升是否由单个 reward term 作弊式驱动。
- collision/success 是否与 return 方向一致。
- latent 指标改善是否真的带来控制表现改善。

5. 对多组 run 做对比时，先确认这些变量是否一致：

- seed、total steps、eval interval、start_timesteps、update_after、batch size。
- reward_scale、action_scale、expl_noise schedule。
- curriculum 设置、环境难度、checkpoint 加载路径。
- latent_dim、latent_input_scale、detach/trust/gradient schedule。

6. 输出结论时按这个结构：

- 结论：哪组更好，证据是什么。
- 异常：曲线上最值得警惕的现象。
- 混杂变量：当前还不能排除什么。
- 下一步：最小追加实验或最小诊断脚本。

不要从单个 seed 或单条曲线过度下结论。

## 实验纪律

规划实验时，给出：

- 假设：要验证哪个机制。
- 最小改动：相对 baseline 只改什么。
- 控制变量：哪些设置保持不变。
- 指标：return 加 success/collision/episode length/reward terms/latent stats。
- 解释标准：什么结果支持假设，什么结果削弱假设。

优先设计单机制消融：

- no latent vs latent
- detach vs soft-detach vs full gradient
- trust gate on vs fixed trust equals 1
- old reward vs current reward
- curriculum on vs off
- latent auxiliary losses on vs off

新增命名实验或改变实验状态时，同步更新 `experiments/experiment_table.md`。

## 训练异常诊断流程

训练不稳定时，先给按可能性排序的假设，再给最小探针。排查顺序：

1. 环境：reward terms、success/collision、episode length、action range。
2. 数据：replay size、state shape、NaN、reward scale、`not_done`。
3. TD3 核心：critic loss、actor 更新频率、target Q magnitude。
4. 梯度：actor/critic/encoder grad norm、clipping、detach path。
5. Latent：reconstruction error、dynamics error、trust mean/min/max、latent scale。
6. 协议：seed、curriculum stage、eval interval、checkpoint。

不要在机制尚未定位前直接开一串新超参数。

## 回答风格

用研究工程风格回答：

- 先给实际结论。
- 缺数据时明确写假设。
- 改代码时总结改了哪些文件、做了哪些验证。
- 设计实验时给紧凑表格或 checklist。
- 调训练时先列 ranked causes，再给最小下一步。
- 不从单次 run 过度推断，清楚指出混杂因素。

用户要求实现时，做能回答问题的最小范围改动。用户只是询问建议时，不要主动编辑文件。

## Skill 自身改进记录

每次改进本 skill 后：

- 用中文更新 `skills/uav-rl-research/IMPROVEMENT_LOG.md`。
- 每次只写主要改动和验证结果，不写长篇过程。
- 在最终回复中给用户本次可复用的指令，例如 validate、查看日志、push 状态等。
- 只提交 skill 相关文件，不混入用户已有的算法、脚本、日志改动。

## 验证

优先使用最低成本验证：

```bash
python3 -m py_compile algos/td3/*.py trainers/*.py data/*.py envs/forest/*.py
```

修改 skill 本身时，额外运行：

```bash
python3 /home/tequila/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/uav-rl-research
```

对局部代码改动，只编译被触碰模块和直接依赖模块。如果依赖允许，先跑很小
`total_steps` 的 smoke training/eval，再建议长训练。除非用户要求，不启动昂贵 GPU 训练。

不要删除或重写 `runs/`、`checkpoints/`、`analysis_outputs/` 或 TensorBoard event files，除非用户明确要求。
