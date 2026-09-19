# runs TensorBoard 数据清单与第一阶段清理建议

日期：2026-06-08

本报告最初只做盘点和建议。2026-06-08 已按第一阶段建议执行：

- 归档位置：`runs_archive/phase1_2026-06-08/`
- 最初共归档 95 个文件、3,848,536,404 字节。
- 随后按表中“删除”建议，永久删除 10 个 `TD3_Hover_seed42_*` 目录：
  70 个文件、57,044,379 字节（约 55 MB）。
- 当前归档保留 25 个数据文件、3,791,492,025 字节。
- 活跃 `runs/` 从约 7.4 GB 降至约 3.9 GB。
- 除明确标记“已删除”的 10 个目录外，其余第一阶段数据仍在本机归档。

## 1. 总览

以下是归档前的盘点基线；当前活跃 `runs/` 已降至约 **3.9 GB**。

- 归档前 `runs/` 占用约 **7.4 GB**。
- 归档前顶层同时存在命名实验目录和 11 个裸 `events.out.tfevents.*` 文件。
- 多个目录包含两次或三次独立训练。TensorBoard 会把同一目录中的事件文件合并成一个 run，容易让 step 重叠、曲线混在一起。
- 旧 logger 几乎每个训练 step 都写大量 scalar，单个 500k run 可达到 1.0-1.1 GB、约 1800-1900 万条 summary。
- 新 logger 使用 `train_interval=500`、`debug_interval=1000` 后，完整 500k run 通常只有 5-11 MB。
- Projector 数据通常每 10k step 保存一次，共 50 组 `tensors.tsv + metadata.tsv`。这部分留到第二阶段做目录内部优化。

建议含义：

- **保留**：独立实验版本或当前仍有比较价值。
- **保留并清理**：目录有价值，但内部混入重复、旧版或压缩副本。
- **先归档**：版本可能有历史价值，但不应继续放在活跃 TensorBoard 根目录。
- **建议删除**：空日志、明确中断的调试 run，或已被完整同版本替代。

## 2. 旧版 TD3_Hover 调试目录

这些目录来自旧版 `original_td3.py` / `change_td3_backup.py` 命名规则，记录
`Loss/*`、`Grad/*`、`Latent/*`、`Reward/*`、`Stats/*`。没有任何一次完成 500k，
且完整冻结基线已经存在。

| 顶层目录 | 大小 | 最大 step | 内容与版本 | 建议 |
|---|---:|---:|---|---|
| `TD3_Hover_seed42_1775646739` | 4.9 MB | 11,948 | 旧 latent TD3 调试，含一套 `best_*` | **已删除** |
| `TD3_Hover_seed42_1775646912` | 4.3 MB | 10,213 | 同代短 run，含 `best_*` | **已删除** |
| `TD3_Hover_seed42_1775647077` | 4.4 MB | 10,545 | 同代短 run，含 `best_*` | **已删除** |
| `TD3_Hover_seed42_1775647112` | 8 KB | 无 | 仅 88 字节初始化事件 | **已删除** |
| `TD3_Hover_seed42_1775647164` | 8 KB | 无 | 仅 88 字节初始化事件 | **已删除** |
| `TD3_Hover_seed42_1775647234` | 8 KB | 无 | 仅 88 字节初始化事件 | **已删除** |
| `TD3_Hover_seed42_1775647298` | 20 KB | 9,987 | 仅 episode reward/length，训练未正式进入稳定记录 | **已删除** |
| `TD3_Hover_seed42_1775647581` | 8.9 MB | 20,248 | 同代较长调试，含 `best_*` | **已删除** |
| `TD3_Hover_seed42_1775647770` | 4.3 MB | 10,285 | 同代短 run，含 `best_*` | **已删除** |
| `TD3_Hover_seed42_1775700185` | 28 MB | 72,943 | 同代最长调试，但仍远未完成 | **已删除** |

## 3. runs 根目录的裸事件文件

这些文件都会被 TensorBoard 视为 `runs` 根 run，彼此混线。建议只保留真正的
500k 冻结基线，并将其移入有名字的归档目录。

| 顶层文件 | 大小 | 最大 step | 内容与版本 | 建议 |
|---|---:|---:|---|---|
| `events.out.tfevents.1776255450.tequila-Vostro-3690.613517.0` | 85 MB | 220,936 | 4 月旧 Forest latent baseline 开发 run | 删除或归档 |
| `events.out.tfevents.1776388832.tequila-Vostro-3690.10724.0` | 72 MB | 189,893 | 同代 curriculum/forest 开发 run | 删除或归档 |
| `events.out.tfevents.1776408523.tequila-Vostro-3690.2.0` | 4 KB | 无 | 88 字节空事件 | 删除 |
| `events.out.tfevents.1776408988.tequila-Vostro-3690.205241.0` | 89 MB | 202,558 | 旧 Forest latent 恢复/调试 run | 删除或归档 |
| `events.out.tfevents.1776413762.tequila-Vostro-3690.2.0` | 4 KB | 无 | 88 字节空事件 | 删除 |
| `events.out.tfevents.1776670497.tequila-Vostro-3690.9138.0` | 199 MB | 500,000 | `baseline_forest_v1` 冻结代完整基线 | **保留并移入命名归档目录** |
| `events.out.tfevents.1776848283.tequila-Vostro-3690.1165090.0` | 26 MB | 39,816 | 早期 V1Trust，已有 trust/representation tags，短 run | 删除 |
| `events.out.tfevents.1777447814.tequila-Vostro-3690.6871.0` | 186 MB | 231,505 | 早期 V1Trust 开发 run | 先归档，确认无用后删除 |
| `events.out.tfevents.1778590187.tequila-Vostro-3690.743762.0` | 352 KB | 10,422 | 5 月 latent baseline 冒烟测试 | 删除 |
| `events.out.tfevents.1778590456.tequila-Vostro-3690.758089.0` | 12 MB | 51,953 | 5 月 plain/no-latent 短 run | 删除 |
| `events.out.tfevents.1778721998.tequila-Vostro-3690.2135263.0` | 192 MB | 481,778 | 5 月 latent baseline，接近完成但被完整 `baseline` 替代 | 先归档或删除 |

## 4. 命名实验目录

| 顶层目录 | 大小 | 完成度/版本 | 建议 |
|---|---:|---|---|
| `forest_curriculum_run1` | 280 KB | 旧 Forest curriculum，最大 10,278 step | 删除 |
| `forest_recover_v1` | 83 MB | 旧 Forest 恢复训练，最大 210,000 step | 先归档，通常可删除 |
| `exp_no_latent` | 117 MB | 4 月冻结环境的 no-latent 消融，完整 500k | **保留**，它对应旧基线消融 |
| `baseline` | 410 MB | 两个完整 500k + 一个 8k；最新 5 月 run 含 eval tags | **保留并清理**：留 `1779104178`，旧完整 run 先归档，8k run 删除 |
| `v1trust` | 933 MB | 三个完整 500k；其中 zip 是独立的 4 月旧 run，不是当前 `.0` 的副本 | **先拆分归档**；至少保留最新 run，旧两次在性能比较后再删 |
| `v1trust2` | 410 MB | 单个完整 500k，独立命名的 V1Trust 调参版本 | **保留** |
| `v1trust_detach_trust1` | 439 MB | 完整 500k；encoder 脱离 RL 且 trust 固定为 1 的独立消融 | **保留** |
| `nolatent` | 180 MB | 一个 268,430 step 中断 run + 一个完整 500k | **保留并清理**：仅留完整 `1779093017` |
| `latent_only` | 350 MB | 初代 latent-only，完整 500k，旧高频 logger | **保留归档**，用于 latent-only 演化对照 |
| `latent_only_soft` | 2.3 GB | 两个同配置族完整 500k；每个约 1.05 GB；另有第二次 run 的精确 zip 副本 | **保留并重点清理**：留较新的 run，旧 run 归档/删，zip 副本删除 |
| `latent_only_soft_beta_schedule` | 1.4 GB | 完整 500k；增加 critic-encoder 梯度 schedule | **保留并清理**：`.zip` 与 `.0` SHA256 完全相同，删 zip 副本 |
| `latent_only_soft_progress` | 23 MB | 完整 500k；增加 `progress_head/progress_loss`，已使用稀疏 logger | **保留** |
| `nolatent_route_tree` | 5.3 MB | route-blocking-tree 新环境的 plain TD3，完整 500k | **保留**，是 route-tree 对照组 |
| `latentonly_route_tree` | 24 MB | route-blocking-tree 新环境的 latent-only + progress，完整 500k | **保留**，已有对比分析依赖 |
| `latent_only_action_aux_dim16` | 27 MB | route/safety action-conditioned auxiliary，完整 500k | **保留**，独立消融 |
| `latent_only_affordance_dim16` | 34 MB | 两个完整 500k：6 月 3 日错误雷达切片版 + 6 月 8 日修复版 | **保留并清理**：只留 6 月 8 日修复版；Projector TSV 已被修复版覆盖 |

## 5. 可回收空间

### 低风险立即清理

- 两个已确认与 `.0` 完全相同的 zip 副本：
  - `latent_only_soft/...1779264282....zip`
  - `latent_only_soft_beta_schedule/...1779282062....zip`
- 5 个 88 字节空事件所在文件/目录。
- `baseline` 的 8k 冒烟 run。
- 6 月 3 日修复前的 affordance 事件对。

仅 zip 副本即可回收约 **562 MB**。

### 按本报告“删除/清理”执行后的预计空间

- 不处理 `v1trust` 历史版本时，预计可回收约 **2.6 GB**。
- 若再把 `v1trust` 的两个旧完整版本移出活跃 `runs/`，可额外减少约 **524 MB**。
- 第一阶段完成后，`runs/` 可从约 7.4 GB 降到约 4.3-4.8 GB；第二阶段再通过事件压缩和 Projector 抽样继续缩减。

## 6. 第二阶段待处理项

本轮尚未执行以下目录内部优化：

- 每 10k step 的 50 组 Projector 快照是否改为每 50k 或只保留关键 step。
- 旧 1 GB 事件文件是否生成稀疏、可直接读取的精简事件文件后再删除原件。
- `v1trust` 三次完整训练、`baseline` 两次完整训练的性能对比与最终择优。
- 为每次训练使用独立子目录，避免今后同一目录内多 run 自动合并。

## 7. 判定依据

- 使用 TensorBoard TFRecord 流式扫描读取每个事件文件的 tag、summary 数量和最大 step。
- 使用事件内 `config/logging`、`config/train_info` 区分新 logger 和 action-aux/affordance 配置。
- 使用 Git 历史确认关键版本：
  - `d254075`：冻结 `baseline_forest_v1`
  - `4d9f678`：更新 TD3 latent/V1Trust workflow
  - `bb28236`：加入 latent-only utilities、稀疏 logging 和 Projector
  - `3e7c3d1`：加入 route-blocking tree
  - `5a4bfe3`：加入 affordance variant，并修复观测布局
- 使用 SHA256 确认两个 zip 是对应 `.0` 的精确副本；`v1trust` zip 则是独立旧 run。
