# Full Scratch EGO20x10 — D3 Critic Lead 200

实验：`full_scratch_ego20x10_d3_critic_lead_200`。seed43，共享Replay受控离线5k→6k已完成；A复用D1精确训练复现。B新增1000次Critic、400次Actor更新，前200次Critic更新期间固定Actor及target Actor，Critic target保留原延迟Polyak节拍。

完整报告：[REPORT.zh.md](../../analysis_outputs/full_scratch_ego20x10_d3_critic_lead_200/REPORT.zh.md)。所有raw checkpoint、机器可读指标、价值扫描/真实Full轨迹、同图验证、配置指纹及科学图均在同一独立目录。

| 核心判断 | 结果 |
| --- | --- |
| Critic先行是否改善排序 | 共同冻结π5k、6s、12真实状态的Q1排序一致率54.9%→29.4%；分组有升有降，没有总体改善。 |
| 相同Actor次数是否更稳定 | B首次188状态全部正饱和在第57次，A在第63次；B第63次及6k仍100%正饱和。 |
| L0能力是否保留 | 初始A5k 11/20成功；B5325/B5600/B6000均0/20。B5200的11/20来自Actor恒等冻结，不是学习成功。 |
| XY是否也受影响 | B第63次固定状态径向命令约−0.318m/s；该checkpoint原始16m评估19个XY越界。 |
| C与在线是否启动 | 没有。B没有收益可分离，未满足事前能力保留条件；没有正式10k/20k续训，未使用test40。 |

1000次实际更新复用A的205主＋51成功采样索引和target noise，B用自身网络计算TD target。全部Critic loss重构误差0；5200/6000实捕获输入及噪声与A逐元素相同而TD target不同，最终Torch smoothing RNG与A6k一致。原任务契约、GRU、网络结构、学习率等正式配置均未修改。

新增240条Full反事实分支，复用60条已校验A6k D2分支；每候选fresh reset＋原动作前缀，12状态重构和同动作重复一步精确一致。评估200个Full validation回合，0.2/0.3/0.5m与跟踪统计均按120Hz下层记录。3项诊断测试通过，204份原输入/生产文件SHA256不变。

该结果说明当前200次先行不足以建立可靠排序或避免Actor追随不可靠Q梯度；不能证明唯一根因是Critic学习慢或Actor步幅大，也不能把2s/6s有限回报当无限Q真值。下一步先审查Replay覆盖、成功/terminal样本贡献、输入尺度及M0契约，不机械扩大先行更新次数。

入口：`scripts/run_ego20x10_d3.py`、`scripts/evaluate_ego20x10_d3.py`、`scripts/summarize_ego20x10_d3.py`。首次运行需新目录，禁止覆盖本轮或正式训练产物。
