# Full Scratch EGO20x10：P0价值审计索引

2026-10-13。诊断完成，没有执行D3或10k/20k续训；原训练checkpoint、奖励、GRU、课程和配置保持原样。

完整报告、raw中间网络、逐批TD审计、配对Full回报、validation20的L0/16m评估与证据等级均保存在独立目录：

[完整诊断报告](../../analysis_outputs/full_scratch_ego20x10_curr_v1_value_audit_seed43/REPORT.zh.md)

[结构化结果](../../analysis_outputs/full_scratch_ego20x10_curr_v1_value_audit_seed43/report.json)

固定状态集的Q1垂直梯度在5043步持续转正，5125步全部正向饱和；L0成功率由2k/5k的55%降至6k的0%。原始16m高度失败表现为上层持续爬升、下层正常跟踪。价值校准不足及Actor启动过早是较强假设，不是经消融证实的唯一根因。D3只提出共享Replay下的200次Critic先行方案，未执行。
