# 感知冻结清单

代码、评测和审计资产提交为 `87a1e3ef710e8f2a24d0793110bdf7d7c61e5559`。注释标签 `perception-v1.0-freeze` 指向包含本清单的最终提交。
使用 `git rev-parse perception-v1.0-freeze^{commit}` 解析最终提交。

## 冻结考场

- 固定训练／验证森林池 `f8a871f25663cc0d6141c7cc9e42cefc1d7579b72a50a778833261a659dbdec2`。
- Suite A/B/C `f970e1a6a2e40b1c5ce54254a860323771b5db11bdac691802f2ff8bd7a698f0`。
- Suite D `70e7d5ee163c9829f32ae87199cdc718cb5b911364f5bfe91636fd20ba769c5a`。
- 文件和哈希见 [封存清单](docs/perception_freeze/manifests)、[T4 报告](docs/perception_freeze/t4_summary_report.json) 与 [T5 报告](docs/perception_freeze/t5_summary_report.json) 。

## 200k / 240k 核心总账

每格为三训练种子合计。A 与 D 的真实穿口分别沿用各自的冻结判定。

| 节点 | 评测项 | T5-Dynamic | T3-Static | RV1 |
| --- | --- | ---: | ---: | ---: |
| 200k | 固定森林 | 211/240 | 229/240 | 207/240 |
| 200k | A 封存森林 | 207/240 | 226/240 | 195/240 |
| 200k | A 真实穿口 | 42/72 | 48/72 | 55/72 |
| 200k | B · 55 树 | 27/60 | 35/60 | 29/60 |
| 200k | C · 密度突变 | 42/60 | 45/60 | 40/60 |
| 200k | Suite D1 | 22/60 | 30/60 | 30/60 |
| 200k | Suite D2 | 37/60 | 40/60 | 55/60 |
| 200k | Suite D3 | 13/60 | 35/60 | 8/60 |
| 200k | Suite D4 | 0/60 | 0/60 | 0/60 |
| 200k | Suite all | 72/240 | 105/240 | 93/240 |
| 240k | 固定森林 | 230/240 | 232/240 | 218/240 |
| 240k | A 封存森林 | 232/240 | 221/240 | 211/240 |
| 240k | A 真实穿口 | 40/72 | 35/72 | 68/72 |
| 240k | B · 55 树 | 48/60 | 55/60 | 42/60 |
| 240k | C · 密度突变 | 56/60 | 47/60 | 41/60 |
| 240k | Suite D1 | 44/60 | 38/60 | 34/60 |
| 240k | Suite D2 | 34/60 | 30/60 | 33/60 |
| 240k | Suite D3 | 20/60 | 33/60 | 0/60 |
| 240k | Suite D4 | 0/60 | 10/60 | 0/60 |
| 240k | Suite all | 98/240 | 111/240 | 67/240 |

## RTX 4080 SUPER 效率

三个方法采用同一组 Stage3 的 1,000 个物理状态。时延为端到端感知与 Actor 推理。Replay 数值为实测主机驻留内存，三组用于存储 Replay 的 CUDA 张量增量均为 0。

| 模型 | 编码器参数 / 每网络 | Actor / Critic 参数 | Replay MB | 均值 / P99（ms） |
| --- | ---: | ---: | ---: | ---: |
| T5-Dynamic (v2.2) | 48,800 | 136,611 / 224,930 | 505.262 | 10.89 / 14.15 |
| T3-Static | 48,800 | 136,611 / 224,930 | 503.190 | 9.98 / 12.90 |
| RV1 (A1) | 145,472 | 233,283 / 321,602 | 1483.284 | 21.96 / 28.17 |

RV1 使用相同并行扫描器时为 8.16 / 10.50 ms。T5 的稀疏边界数据和待提交回合不计入上述 Replay 驻留测量。

## 18 个检查点

只读归档根目录为 `/opt/uav-training-20261103/artifacts_frozen/perception_v1`，合计 6,887,801,938 字节。
归档保留了下表在原运行根目录下的相对布局。原始路径与字节数见 [检查点索引](docs/perception_freeze/checkpoint_hashes.json)。
归档可使用 [SHA256SUMS](docs/perception_freeze/SHA256SUMS.txt) 校验。

| 模型 | Seed | 节点 | 归档相对路径 | SHA256 |
| --- | ---: | ---: | --- | --- |
| RV1 (A1) | 43 | 200k | `runs/formal_gpu_a1_seed43_240k_20261118a/checkpoint_200000.pt` | `d07e0aba739f2cf1768b074f364a5c00a5be02cbe09ca02fbbbc6f2356d1dcd8` |
| RV1 (A1) | 43 | 240k | `runs/formal_gpu_a1_seed43_240k_20261118a/checkpoint_240000.pt` | `1ecf07ff637aa4a28d17ec23ff6f825df061f57ee505f6f2c3c5797ee1a4ad85` |
| RV1 (A1) | 44 | 200k | `runs/rv1_baseline_seed44_240k/train/checkpoint_200000.pt` | `0c8f95119d896cc33ffd89baa73c261cb46eb755b88a29ebd1a3bf0899b3cb8a` |
| RV1 (A1) | 44 | 240k | `runs/rv1_baseline_seed44_240k/train/checkpoint_240000.pt` | `ca29b9295b941cba00f7bdb1a3b29efaa673a348638de4e155e994e7a983d12c` |
| RV1 (A1) | 45 | 200k | `runs/rv1_baseline_seed45_240k/train/checkpoint_200000.pt` | `0fdfbd4307e1dad332f94209926eec08141b2a3a145dea6406b159a04b07b956` |
| RV1 (A1) | 45 | 240k | `runs/rv1_baseline_seed45_240k/train/checkpoint_240000.pt` | `3e3d40b6eb3406975a544431211bd4ec6230444bcc6ca1d121431f896cbe983a` |
| T3-Static | 43 | 200k | `runs/msgp_conv7_circ16_dual_seed43_240k/train/checkpoint_200000.pt` | `bbd7467afd2ad8abcfd7a458241e6fde19bf058955a272bebacf2d6df7e96f93` |
| T3-Static | 43 | 240k | `runs/msgp_conv7_circ16_dual_seed43_240k/train/checkpoint_240000.pt` | `2f8adae99afee0d4f1f8dbbedfcf51a1311636e13946c0b9d009dd4a705839f5` |
| T3-Static | 44 | 200k | `runs/msgp_conv7_circ16_dual_seed44_240k/train/checkpoint_200000.pt` | `4bf299cf8bffdf297b9429f7900d30cf391d35c5b5b570ab4419d34933aed398` |
| T3-Static | 44 | 240k | `runs/msgp_conv7_circ16_dual_seed44_240k/train/checkpoint_240000.pt` | `478814e9db20c665eb3f7db07fad940618f15c4a413ad2b23ccad5baa8b0b705` |
| T3-Static | 45 | 200k | `runs/msgp_conv7_circ16_dual_seed45_240k/train/checkpoint_200000.pt` | `c5a18dd97c249a592ca02d535469abd08e046644dfe0a7bbec6b0474f064c57d` |
| T3-Static | 45 | 240k | `runs/msgp_conv7_circ16_dual_seed45_240k/train/checkpoint_240000.pt` | `76542e0fe69896f439eb6e9c339b2aed8169751a5d4f3ca80d37cf43ba3adddb` |
| T5-Dynamic (v2.2) | 43 | 200k | `runs/msgp_conv7_circ16_dyn_v22_seed43_240k/train/checkpoint_200000.pt` | `d10a2abccba849dade30a71721499358d87ac944123b2b3473fc56425fcdd4f4` |
| T5-Dynamic (v2.2) | 43 | 240k | `runs/msgp_conv7_circ16_dyn_v22_seed43_240k/train/checkpoint_240000.pt` | `12fece7c26f36d4b410be8ce560712acb4bd850a23dc937a1aa48595caa8a9c3` |
| T5-Dynamic (v2.2) | 44 | 200k | `runs/msgp_conv7_circ16_dyn_v22_seed44_240k/train/checkpoint_200000.pt` | `7bcdc57b845718e411ac77dda9b19d0adeb59b21eea31d57faa2d0f66ef30f9f` |
| T5-Dynamic (v2.2) | 44 | 240k | `runs/msgp_conv7_circ16_dyn_v22_seed44_240k/train/checkpoint_240000.pt` | `ec4f4022dcee70029012c03a343c4f4e3ba334b90b0c074d922e6b24ed79f3bd` |
| T5-Dynamic (v2.2) | 45 | 200k | `runs/msgp_conv7_circ16_dyn_v22_seed45_240k/train/checkpoint_200000.pt` | `54c8bad94347abbec4d1142629194da4a9ad0fd0bd649b41cbcb36c9fd55c573` |
| T5-Dynamic (v2.2) | 45 | 240k | `runs/msgp_conv7_circ16_dyn_v22_seed45_240k/train/checkpoint_240000.pt` | `1d9beb7bc83294108f6ccaf5cc5e8ddb1b36a1ce1f95d0ee9bb16efb75c5e9c7` |

## 环境与复现

- WSL2 内核 6.18.58.1，Python 3.10.12，PyTorch 2.11.0+cu128，CUDA 构建版本 12.8。
- RTX 4080 SUPER，驱动 591.86。NumPy 1.26.4，PyBullet 3.2.7，Gymnasium 1.2.3，SciPy 1.15.3，Matplotlib 3.10.8。
- Gym-PyBullet-Drones 2.0.0+snapshot.20260907.asset1。本地供应包 `gym_pybullet_drones-2.0.0+snapshot.20260907.asset1-py3-none-any.whl`，SHA256 `6f5065e5fceca747453e606a930bdf769611b59ddca56c5c05a4b49619914b33`。
- 运行源码位于 [src/perception_runtime](src/perception_runtime)。`fusion_modes.py` 提供 `static` 和 `phys_dynamic_v22`。原训练入口保留历史课程与训练前置资产依赖。下面的 CPU 复评入口使用本冻结包和归档中的 18 个检查点。

```bash
export PERCEPTION_ARCHIVE=/opt/uav-training-20261103/artifacts_frozen/perception_v1
export PY=/opt/uav-training-20261103/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PY" tools/perception_reproduce.py verify --archive "$PERCEPTION_ARCHIVE"
PYTHONPATH=tools "$PY" -m unittest discover -s tests -p test_perception_freeze.py -v
"$PY" tools/perception_reproduce.py replay --archive "$PERCEPTION_ARCHIVE" \
    --model phys_dynamic_v22 --seed 43 --step 240000 --suite d \
    --output local_reproductions/t5-seed43-suite-d-240k
```

可将模型参数换成 `rv1` 或 `static`，考场换成 `fixed` 或 `abc`。复评器按原始字节拼接固定森林池，校验哈希后再使用。

## 已知边界

- Pre3 的 Circ32 离线对照把开口占用率／自由率定位峰从 4/72 提到 46/72。4 m 组仍为 0/24。证据见 [Pre3 诊断](docs/perception_freeze/pre3_far_ablation.json)。
- 240k T5 的 D3/D4 到达为 20/60、0/60，静态 T3 为 33/60、10/60。静态 T3 作为中间层主对照，T5 留作动态调制对照。
- T5 的 CPU 前置门禁中，Suite C 最大单帧门控变化为 0.10582，超过 0.05 阈值。后续正式训练使用了已授权的门禁覆盖，见 [门禁原始结果](docs/perception_freeze/preflight_v22_check.json)。
