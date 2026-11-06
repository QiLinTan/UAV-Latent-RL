# StaticLearnable 工程验收状态

**NOT_RUN：原 WSL 执行返回 Wsl/E_ACCESSDENIED；尚无数值 PASS 证据。**

本轮已核查并补全：

- 全局 Gate 的 forward 无输入参数，仅对两个可学习 Logit 执行一次 Softmax。Actor/Critic 分开，初始各为 0/0；State18 仍通过原 Actor/Critic 主体输入。
- 共有 CNN/LayerNorm/策略主体直接复用原 NF；先按原顺序构造全部网络，再替换 Gate，以保留共有参数的初始化随机数顺序。这是实现意图，逐元素数值比较仍待执行。
- 工程脚本检查前向维度、独立 Logit、无状态依赖、原优化器覆盖新参数、两路有效梯度；分别断言 Critic 更新改变 Critic 权重、Actor 更新改变 Actor 权重、仅 Critic 更新不改变 Actor 权重。
- 对实际继承的 TD3 更新检查 Target Polyak 关系，随后检查网络、优化器与推理保存恢复一致。
- 语法编译通过；数值测试和真实环境 5008→5012 步恢复冒烟未运行，故不启动正式训练。

实现：learnable_agent.py；测试：engineering_gate.py；配置：config.json；隔离训练入口：train.py。不得把本文件的检查清单当作已通过的结果。
