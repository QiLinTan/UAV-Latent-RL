# 近期工作 GitHub 同步

同步日期：2026-10-14。分支：`feature/latent-reference-planner-v2`。

本次整理 10 月 12–13 日 Full / EGO20x10 实验及其尚未提交的课程训练依赖，保留此前 10 月 8–10 日的工作日志、诊断脚本和相关测试。没有重新启动训练或运行封存 test40。

## 报告入口

- [下一步机制实验：Actor 单步保护与 Critic 局部校准](../../analysis_outputs/full_ego20x10_next_step_v1/REPORT.zh.md)
- [冻结 Actor 的 20k Critic 最终验收](../../analysis_outputs/full_critic_frozen_v1_seed43/completed_review/REPORT.zh.md)
- [Critic 确定性复核](../../analysis_outputs/full_critic_frozen_v1_seed43/deterministic_review/REPORT.zh.md)
- [Critic 完成状态复核](../../analysis_outputs/full_critic_frozen_v1_seed43/complete_review/REPORT.zh.md)
- [20×10m 任务契约审计](../../analysis_outputs/full_ego20x10_task_contract_audit_v1/REPORT.zh.md)
- [D3 Critic 提前更新 200 步对照](../../analysis_outputs/full_scratch_ego20x10_d3_critic_lead_200/REPORT.zh.md)
- [EGO20x10 价值与动作退化审计](../../analysis_outputs/full_scratch_ego20x10_curr_v1_value_audit_seed43/REPORT.zh.md)

当前结论仍为：单步 Actor 保护不足以防止累计漂移；小规模 Critic 局部校准没有通过跨地图方向检查；工程候选的学习收益尚未验证。详细证据和适用边界以各报告为准。

## 归档范围

提交环境、训练器、评估回调、实验与汇总脚本、相关测试、地图目录及冻结地图池 manifest。`analysis_outputs/` 继续默认忽略，本次显式选入 177 份正式报告、图表、协议、保护校验及结构化结果，共约 48.6 MiB；另补齐工作日志直接引用的较早报告附件。

模型权重、Replay、NPZ 原始轨迹、TensorBoard 和过程日志、预检查及被拒绝的过程目录保留在本地。本次不是完整训练状态备份；重新执行依赖这些输入的实验仍需本地原始产物。特别是 D3 的 `paired_values.json`（约 30.4 MiB）及 `validation_traces/` 未纳入 Git，报告中的对应链接指向本地产物。原报告中的 `文件.py:行号` 链接保留原样，GitHub 不将其解释为代码行锚点。

## 同步前验证

- `git diff --check` 通过。
- 待提交 Python 源码 AST 与 JSON 解析检查通过。
- 18 个相关测试文件，74 项测试通过（38.29 秒）。
- pytest 关闭系统第三方插件自动加载，避免无关 ROS 插件缺少 `yaml` 导致启动失败；测试使用现有 `drones` Python 环境。

测试范围：地图 atlas、EGO 地图/课程、任务契约、D3、价值审计、冻结 Critic、Full FT v1/v2 与解冻审计、Full scratch、响应预算、起点/障碍课程、阶段微调及 episode coverage。

测试环境：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/uav-sync-mpl`。依赖弃用、Gymnasium float32 转换和 CUDA 驱动提示未导致测试失败。
