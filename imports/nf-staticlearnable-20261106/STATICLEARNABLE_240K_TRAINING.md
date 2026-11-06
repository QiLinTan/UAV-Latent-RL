# StaticLearnable seed43 正式训练状态

**未启动，0/240,000 transitions。**

2026-11-06 本轮复查：D:/UAVTraining 下未找到名称含 learnable 的正式训练目录；本地工程状态仍为 BLOCKED_RUNTIME，没有已有完整包可续训。WSL 进程和队列无法读取，因此不宣称已排除 Linux 中的其他任务。

当前 WSL 调用返回 Wsl/E_ACCESSDENIED。本会话执行策略禁止 sandbox 提升；此前获得的文件访问授权未解除 WSL 执行限制。磁盘不是阻塞：D 空闲 147,625,762,816 bytes（约 137.5 GiB）。计划预检至少 40 GiB，保存前至少 25 GiB；按既有 Replay 大小及临时写入估算，资源预算足够。

工程通过后才按 seed43、原 NF 环境与前馈 TD3 从头运行240k，不加载最终 NF 权重。保留七节点模型以及 40k/240k 完整恢复包，不永久保存全部中间 Replay。原两组不重训。

新增日志：gate_trajectory.jsonl 记录四个网络的权重；updates.jsonl 记录原 Loss/Q/TD Error，分支梯度在前两次及每100次更新采样（非采样项留空）；course_and_policy.jsonl 每100步记录课程窗口和无探索 Actor 输出。Near/Far RMS 与参数变化沿用 checkpoint 观测采样。恢复运行后才能确认这些接口正常，不将其列为已观测指标。
