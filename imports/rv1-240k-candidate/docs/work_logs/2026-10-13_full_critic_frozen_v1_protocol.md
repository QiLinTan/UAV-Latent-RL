# Full Critic延长训练：源Actor全程冻结，累计20k

实现入口：`scripts/run_full_critic_frozen.py`。主预算是**累计20000个上层转移：恢复原4000条前缀，再增加16000条Full经验**。累计20k训练现已完成，本轮使用既有快照完成验收，没有重新启动20k训练。最终结果见[完整验收报告](../../analysis_outputs/full_critic_frozen_v1_seed43/completed_review/REPORT.zh.md)。下文保留原预设协议及初始烟测记录。

## 固定条件与恢复

统一起点为`checkpoints/full_ft_v1_unfreeze_audit_20261012/seed43/model_004000_full_snapshot.pt`。使用已验证的六CPU线程方法重放原前4k；继续之前，在线/目标网络、Adam、主/成功Replay、待提交成功转移、NumPy/Python/Torch RNG、课程、观测、控制/参考状态都必须逐项精确匹配该快照，同时原4000步观测/动作/奖励/位置差异必须为0。任何不一致立即停止，保留差异文件，不换种子或重置环境来替代。

恢复期间沿用之前已明确授权的seed43既有训练前缀例外；新增验证固定环境seeds80–89。4k处保留同一正在执行的episode、参考与DSLPID状态，保持Replay及成功回合待提交队列，不另开一局。

| 模块 | 实际处理与核验 |
| --- | --- |
| Actor MLP / GRU | 全程`requires_grad=False`，`actor_frozen=True`，禁止Actor optimizer.step；定期逐张量核对源参数，Adam始终为空 |
| Actor Target | 与源Actor逐张量相同，参数不可训练，不进行Actor软更新 |
| Critic Q1/Q2 / GRU | 正常梯度、裁剪、Adam更新，LR固定原`3e-4` |
| Target Critic | 原tau=0.005、policy_freq=2，按每两次Critic训练软更新，冻结Actor不会阻断该更新 |
| Replay | 原Full经验，继续追加，不丢弃原4k内容 |
| Success Replay | 原Full成功整回合提交、容量、分层与0.2抽样比例；保留原待提交队列 |
| Stage采样 | 原Stage0/3按episode每十局3/7打乱采样；不按转移数重新配比 |
| 探索 | 源Actor加原噪声0.15→0.05，衰减仍按原40000步；20k时噪声为0.10 |
| TD3目标平滑 | 原归一化sigma=0.2、clip=0.5、动作边界和圆盘投影 |
| 观测/奖励/控制 | 原917D，3D动作，15Hz/120Hz，原Full动力学、Reference Generator、DSLPID、12s/0.2m定义 |
| 折扣与缩放 | gamma=0.99，每15Hz上层转移折扣一次；Full奖励为实际下层tick之和，Replay只乘一次0.1 |

冻结的是Actor参数；观测历史及GRU正常前向推理继续进行。两个Critic GRU均保持可训练。不会调用保守阶段改变Critic LR；每次更新检查LR仍为3e-4。

主实验在4k、5k、10k、15k、20k保存包含Replay的完整学习快照。每5k沿用Full Stage0/3与K1 Stage3导航评估，但这些数字只核查固定策略/控制链路，不用来选择Critic。**取消按导航成功率选择并恢复早期Critic**；`model_raw_final`和`model_final`保存同一个20k在线Critic，真实Target Critic等完整状态保存在`model_020000_full_snapshot.pt`。

正常20k结束应有19489次Critic更新、0次Actor更新；4k后新增16000次Critic更新、8000次Target Critic软更新。若Critic出现非有限loss或冻结/LR约束失效，停止并记录failure，不替换运行。

## 两层验收

审计入口：`scripts/audit_full_critic_frozen.py`，比较4k、10k、20k三个Critic。主验收不执行任何Actor更新，只测动作空间中的Critic指导能力。

固定库取环境seeds80–89、Stage0/3共20条源策略轨迹的起点、第60步、首次进入1.2m球、终止前状态，去重后**78个状态**：Stage0=40、Stage3=38，含20个起点、20个中途、18个近目标、20个终止前状态。保留源策略发生碰撞的任务，不能仅校准成功终止奖励。

每个状态最多16个候选：源动作、停止、三轴±0.3局部扰动、沿Q1梯度正/负方向的0.05/0.15/0.30投影扰动、Q1最高与min-Q最高网格动作。重复执行动作去重，不以别名增加配对数。

**只改变第一步上层动作，后续始终为同一个冻结源Actor。** 从源轨迹重放恢复该状态，校验观测、位置、速度、地图及参考字段；按原剩余12s时限直到真实终止，计算

`G(o,a) = Σ_k 0.99^k × 0.1 × 原Full上层奖励_k`。

不能重置时限，也不能把多步候选前缀回报与单步Q混比。回报缓存绑定状态、动作、执行协议、源策略/地图/代码指纹，固定动作的G可供不同Critic复用；Critic checkpoint哈希变化或生产代码/地图输入变化会拒绝混用结果。

### 第一层：固定状态/动作库上的Full价值校准

分开报告源策略动作与所有候选动作的Q1、Q2、min-Q对G的bias、MAE、RMSE；另按Stage和起点/中途/近目标/终止前分层。预设第一层通过条件为源动作与所有候选两组的Q1/Q2均MAE≤2、|bias|≤1，单位为**已缩放回报**。这只是本实验预设，未宣称最优阈值，也不证明整个Full状态空间均准确。

训练快照另记录固定4k Replay的512条TD探针、成功/失败终止样本拟合、Replay增长及成功分层覆盖。TD拟合只作机制/经验覆盖诊断，不能代替真实G校准。

### 第二层：排序及真实局部动作梯度

Actor原目标使用Q1，因此Q1作为主要验收，Q2/min-Q并列报告。

- 排序：同状态所有唯一动作两两配对；`|ΔG|≤0.05`视为不可区分，Q差≤1e-6不算有效方向。至少200个可区分对、30个有可区分对的状态、每个Stage至少10个状态；Q1符号一致率≥80%，地图聚类bootstrap的95%区间下界>50%。
- 梯度：主要步长0.15，0.05/0.30作尺度对照。检查投影后的`a+`与`a−`是否实际不同，`G(a+)−G(a−)`方向是否正确，及`G(a+)−G(a_source)`是否真正改善。局部扰动的回报变化较小，使用预设缩放G差0.005作为局部近似平局阈值；全候选排序仍用0.05。
- 梯度证据至少30个可区分正/负对、每Stage至少5个。还需至少10个状态存在可测局部改进空间：某候选距源动作≤0.35、G提升>0.005；缺乏改进空间时不强行把“无法提高已好动作”判作Critic失败。
- 梯度通过还要求正方向真实回报优于负方向的比例≥70%、地图聚类95%区间下界>50%、平均源→正梯度G变化≥0、Q1正向高于负向比例≥80%，并在至少50%的可测局部改进空间状态上找到G提升>0.005的正方向动作。

可区分对不足时第二层标为`insufficient_evidence`，不自动算通过。动作对同属状态/地图，不能声称大量配对等于大量独立样本；bootstrap按十个环境seed聚类。

**两层独立判断。** 固定价值误差较小仍可能动作排序/梯度错误；常数价值偏移也可能排序正确。第二层通过只说明该状态库上的局部动作指导可用，不保证Actor参数更新或持续闭环更新稳定，整个主实验及审计始终不更新Actor。

### 目标噪声对照

每个状态对源动作、Q1最高动作、正梯度0.15动作，分别增加四个共同随机噪声副本：第一步候选不加噪声，后续源Actor加TD3原N(0,0.2)噪声并截断±0.5、裁剪边界和投影。对同一状态/副本，各候选及各Critic使用相同噪声序列。

主确定性协议与噪声控制分开报告。原Critic的Bellman目标使用平滑后的源策略，所以必须检查这一价值定义差别；四副本仅检查混淆，不作高精度期望估计或替代主判据。

## 能回答的经验/机制问题

- 4k→10k→20k校准/排序/梯度改善：支持在固定策略、原超参数下延长Full训练有效。
- 终止拟合、价值传播及源/候选校准改善程度不同：帮助定位仅终止奖励拟合、远处价值传播、局部动作排序等机制差异。
- 只有第一层通过：Critic尚未证明可指导Actor改进。
- 第二层通过：提供局部动作指导证据，不能把它当作实际Actor训练收益。

主实验**同时增加经验与Critic更新次数**。它不能独立归因为“经验不足”；严格分离数据量与优化时间，还需固定4k Replay额外训练的独立对照，本次主实验未加入该对照，也没有改学习率来替代。

## 运行指令

在项目根目录执行。一次完成累计20k训练及两层验收：

```bash
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 MPLCONFIGDIR=/tmp/mpl-full-critic-frozen \
/home/tequila/miniconda3/envs/drones/bin/python -u -m scripts.run_full_critic_frozen \
  --total_steps 20000 --audit
```

无人值守启动并保留控制台日志：

```bash
nohup env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 MPLCONFIGDIR=/tmp/mpl-full-critic-frozen \
/home/tequila/miniconda3/envs/drones/bin/python -u -m scripts.run_full_critic_frozen \
  --total_steps 20000 --audit \
  > analysis_outputs/full_critic_frozen_v1_seed43_training.log 2>&1 &
```

若需要先训练、之后单独审计，可在训练命令省略`--audit`，完成后执行下列命令。审计中断后也用它继续缓存任务，不必重新训练：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-full-critic-frozen \
/home/tequila/miniconda3/envs/drones/bin/python -u -m scripts.audit_full_critic_frozen \
  --phase all
```

默认产物：

- `checkpoints/full_critic_frozen_v1_seed43/model_{004000,005000,010000,015000,020000}_full_snapshot.pt`。
- `analysis_outputs/full_critic_frozen_v1_seed43/prefix_verification.json`、`frozen_checks.json`、`critic_probes.json`、`result.json`。
- `analysis_outputs/full_critic_frozen_v1_seed43/audit/report.md`、`summary.json`、`manifest.json`、逐checkpoint逐候选回报及真实分支轨迹。
- `runs/full_critic_frozen_v1_seed43`及独立训练/验证轨迹。

主训练入口拒绝非空产物路径，不将重复执行误当作实时续训。不要将`--total_steps`写成24000；20k已经包含4k起点。

## 验证记录

### 最终执行与验收

累计20k：Critic19489次更新、Actor0次更新；4k前缀精确恢复。每个4k/10k/20k快照完成1248条确定性记录和936条唯一共同噪声记录。20k源Q1 MAE为5.371（确定性）/4.634（噪声），主排序一致率52.1%、主要局部梯度一致率42.0%；第一层与第二层均未通过。噪声控制未提供足够的可靠方向证据，不能据此解冻Actor。

续跑审计发生完全相同记录的并发追加；最终报告按`step/case/candidate/replica`去重并验证重复内容一致，冲突0。原记录和checkpoint均保留，完整来源指纹见`completed_review/provenance.json`。后续读取最终验收应使用`completed_review/audit/summary.json`，避免将原JSONL重复行当作新增证据。

新16m任务的独立工程与训练机制短对照见[执行工作日志](2026-10-13_full_ego20x10_next_step_v1.md)，与本7m实验分开解释。

10项相关测试通过，覆盖Actor/Target硬冻结、Critic两支GRU实际更新、原周期Target Critic软更新、LR改变拒绝、原探索/奖励/控制配置、训练初始/终止禁用Critic选择恢复、校准与梯度结论分离、近似平局及重复动作去重、验证种子范围。

隔离烟测位于`/tmp/full_critic_frozen_smoke_20261013`：完整重放4k、逐字段及4000步轨迹精确匹配原快照，再执行2次Critic更新至累计4002。Actor/Actor Target逐张量保持源策略，Critic及Target Critic改变，LR=3e-4，Actor更新0、Critic更新3491；没有结束时恢复早期Critic。原2850个受保护产物文件哈希不变。

另在同一隔离目录完成4k单状态（Stage0、地图80、起点）的确定性16候选和三个动作×四副本噪声审计烟测。该输出标为`audit_smoke`，不能通过正式验收；正式78状态、4k/10k/20k结果须由以上指令生成。正式20k尚未启动，不能据烟测宣称价值学习已改善。
