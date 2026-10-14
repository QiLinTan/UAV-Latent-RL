# 开源地图与仿真资源建议核查

2026-10-13。针对用户粘贴的 EGO-Swarm、Mockamap、SensorSimulator、Agile Autonomy、Flightmare、MARSIM 及三个数据集建议，核查官方仓库 README、部分生成器源码/配置和本项目接口。核心仓库链接真实存在，建议的总体方向合理；没有在本轮安装外部工程、下载地图资源包、编译或运行这些仿真器。以下明确区分已核实事实与接入判断。

## 资源用途与取舍

| 资源 | 已核实内容 | 对当前项目的判断 |
|---|---|---|
| [EGO-Planner-Swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm) | 有参数化随机柱体/环形点云生成代码，支持 X/Y 尺寸等参数 | 最适合优先参考参数设计和分布；在现有 PyBullet 几何接口内实现 |
| [ROS2 random-forest 移植版](https://github.com/plkdz/ros2-random-forest-map-generator) | README 明确了参数语义和全局/局部发布路径，也说明固定 seed 功能 | 文档有参考价值；本轮部分源码页面未能获取，不声称已完整审计其实现 |
| [Mockamap](https://github.com/HKUST-Aerial-Robotics/mockamap) | README 有 Post2D、Perlin3D 示例；源码还含迷宫生成逻辑 | 适合后续复杂占据场景；先建立三维碰撞与可达性表示 |
| [TJU SensorSimulator](https://github.com/TJU-Aerial-Robotics/SensorSimulator) | README 列出溶洞、柱子、迷宫、森林、房间，以及点云/深度/LiDAR 仿真 | 洞穴/迷宫是有价值的下一阶段参考；不是现成 PyBullet RL 环境 |
| [Agile Autonomy](https://github.com/uzh-rpg/agile_autonomy) | 官方论文代码；提供预训练、数据链接、数据采集和训练流程、环境配置方法 | 用于核对论文任务定义和参考实验，不直接替换 TD3 |
| [MARSIM](https://github.com/hku-mars/MARSIM) | README 声明发布 10 套低/高分辨率地图，给出 .pcd 示例、动态障碍和 ROS2 分支 | 真实感几何测试资源候选；须转换并校验，不能直接当成 PyBullet 碰撞场景 |
| [Flightmare](https://github.com/uzh-rpg/flightmare) | Unity 渲染与动力学模块解耦，提供 RL 接口；文档列出四类场景 | 后期视觉或交叉仿真候选；“必须整体迁移平台”说得过于绝对 |

优先级不应全标为五星：目前最快形成可复核导航实验的是参数化圆柱/box 场景。复杂占据地图、真实点云和仿真平台属于不同的工程层次；可用资源丰富不代表接入成本相同。

## EGO 参数不能按名称照搬

对当前 master 的 [random_forest_sensing.cpp](https://github.com/ZJU-FAST-Lab/ego-planner-swarm/blob/master/src/uav_simulator/map_generator/src/random_forest_sensing.cpp) 检查显示：

- 活跃柱体分支把 `lower_rad/upper_rad` 读入宽度参数，半径由宽度乘内部随机缩放后再除以二得到。
- main 使用 `random_device` 的种子，没有读取 `ObstacleShape/seed`；[simulator.xml](https://github.com/ZJU-FAST-Lab/ego-planner-swarm/blob/master/src/planner/plan_manage/launch/simulator.xml) 的该参数位于注释的节点片段，不能据此证明运行可重复。
- 全局点云发布后的提前 return 同样存在于上游，局部搜索代码未执行。生成器发布地图本身不是错误，但全局点云不能被当成 Actor 的局部传感器输入。
- `min_distance` 是中心间距；高度范围、点云分辨率和地图 Z 尺度需要分别解释，Z 参数不等于障碍高度自动裁剪。

这说明 EGO 值得借鉴，仍需要建立本项目自己的明确几何与 RNG 契约。固定 seed 后也应保存几何实例与 hash；Python/C++ 随机序列不同，不能声称相同 seed 必然复现同一地图。

## 其他资源的关键核查

**SensorSimulator** 的 [官方说明](https://github.com/TJU-Aerial-Robotics/SensorSimulator) 确认场景编号 1/2/3/5/6 分别为溶洞/柱子/迷宫/森林/房间。森林模式另需树木点云文件，README 将 CPU 版本标为已弃用，并说明部分 Perlin/迷宫生成来源于 Mockamap。故这些资源存在重叠，没必要一开始独立集成两套完整系统。本项目 catalog 中的 SensorSimulator `source_url` 当前只指向场景参考来源，`generated_each_reset` 仍是本地程序生成，不能把已有小森林称为官方 SensorSimulator 原始场景。

**Agile Autonomy** 的 [依赖文件](https://github.com/uzh-rpg/agile_autonomy/blob/main/dependencies.yaml) 实际指定 `antonilo/flightmare_agile_autonomy` 的 `main` 作为 `rpg_flightmare`，并包含 RotorS 和相应控制依赖。复现该论文不能简单替换成通用 Flightmare 最新版本。官方 README 支持其为模仿学习/DAgger 相关流程，环境配置与训练框架应分别借鉴。

**Flightmare** 的 [场景说明](https://github.com/uzh-rpg/flightmare/blob/master/docs/source/first_steps/envs_and_navigation.rst) 列出 Industrial/Warehouse/Garage/NatureForest；[构建 FAQ](https://github.com/uzh-rpg/flightmare/blob/master/docs/source/building_flightmare_binary/faq.rst) 说明公开 Git 场景源码只包含 Industrial，其余场景与二进制包/外部资产有关。FAQ 中数量文字和枚举有不一致，应以指定包实际内容为准。README 的渲染/动力学解耦意味着以后可以研究保留 PyBullet 动力学、单独接外部渲染器；同步、位姿和碰撞一致性仍需工程验证。本轮未验证这种混合接口。

**MARSIM** 官方 README 支持“10 套低/高分辨率真实感地图、.pcd 使用示例、动态障碍和 ROS2 分支”的说法。这是官方发布声明，本轮没有下载资源包逐图确认尺寸、原始采集来源或重建质量。“真实感”不能自动替换成“全部为真实世界采集的地图”。建议先选一个小裁剪区域，检查单位/坐标系、地面、树干、空洞和路径是否合理，再进行体素化或碰撞几何拟合。

**Mockamap** 的 [maps.cpp](https://github.com/HKUST-Aerial-Robotics/mockamap/blob/master/src/maps.cpp) 说明 Perlin3D 是噪声阈值生成占据点，不能直接等同可通行的矿井；Post2D 的随机柱状对象在实现中具有方形截面，也不应按名称当成圆柱森林。须分别检查自由空间连通性、无人机膨胀后的通行净空和地图边界。

## 三个数据集的实际用途

- [Wild-Places](https://github.com/csiro-robotics/Wild-Places)：官方定位是自然环境 LiDAR place recognition 数据集；可参考真实森林点云结构，仍需重建交互几何。
- [Mid-Air](https://github.com/montefiore-institute/midair-dataset)：多模态合成低空飞行序列，适合感知模块研究。
- [TartanAir](https://tartanair.org/)：模拟环境的多模态传感器/轨迹数据；V2 已含 LiDAR 和 occupancy 相关接口，不宜仅概括为 RGB 数据集。但离线轨迹样本不等于已经接通当前 PyBullet 环境。

粘贴文本将地图生成器优先于大型视觉数据集的建议符合当前 rays 策略需求。是否引入数据集应由具体感知或几何测试目标决定。

## 本项目已具备的接口与真正缺口

不用从零建立独立地图加载体系：`envs/map_atlas/spec.py` 的 `MapSpec/StaticObstacleSpec` 已定义 cylinder/box/mesh；`runtime.py` 已用同一障碍规格创建碰撞体和视觉体，现有 rays 对 PyBullet 世界进行测距。catalog 中已有 tunnel/maze 本地 proxy，适合作为原语接入回归测试，但不能作为官方 SubT/FlightBench 地图结果。

需要优先补齐：

1. `xy_half_extent` 是单一方形尺度，矩形 X/Y/Z 边界以及 ForestLayoutGenerator/课程/地图校验需要统一。
2. 任务时长目前 12s，垂直终止范围固定为 0.08–2.75m，不能直接支持建议的 20×20×5m 场景。当前 Full scratch 平面命令上限是 **0.8m/s**，粘贴文本的 0.6m/s 属于旧参考：40m 直线理想时间约 50s；需登记转弯、加速和捕获的额外预算。
3. 当前路线校验是固定高度二维搜索。复杂三维场景必须做按无人机尺寸和安全间隙膨胀后的三维连通性检查；几何可达性还不等于给定速度下动态可行。
4. `MapRuntime.nearest_clearance()` 对 mesh 直接返回 `inf`，cylinder/box 使用垂直重叠门控后计算 XY 净空；不能把现有指标直接当成复杂三维场景完整 Euclidean 净空。mesh 加载能力与正确的安全净空能力是两回事。
5. 观测绝对位置、目标归一化、感知范围和起点课程需审计尺度迁移。当前 GRU 接收 8 帧显式窗口，不能因名字含 Recurrent 就假定具备整回合地图记忆；先用场景实验辨认感知盲区、有限记忆与优化问题。
6. 当前 `path_efficiency` 是净目标距离进展/轨迹长度。粘贴文本建议的是可行参考路径效率，通常应另定义 `L_reference/L_actual`；复杂通道中两者含义不同，应新增而非改名覆盖旧指标。可行参考路径也要说明分辨率和安全膨胀，不声称已经是真实最短路径。

## 接入建议

第一步在现有 atlas 上增加矩形森林和 box 窄缝实例，使用统一 manifest、固定几何 hash 和可配置任务时间。生成器输出连续几何规格，点云/体素是派生表示，避免先把精确圆柱离散成点云、再拟合回来造成误差。

第二步采用 10–15m 中距离低速任务做 reset、碰撞、遮挡、ray 和净空一致性检查及冻结策略评估，再扩到 30×60m、40m 的森林和 20–40m 的通道。长任务的折扣时域、输入分布和课程配置须登记为新协议，不改旧实验名。

第三步再用 Mockamap/SensorSimulator 生成占据场景，补齐三维可达性与距离场。最后选 MARSIM 的小范围场景完成点云转换，再开展真实感地图泛化验证。Flightmare 可作为后续独立或混合渲染验证方向。

同意粘贴文本关于不泄漏全局地图、碰撞与感知一致、训练/选模/测试划分、训练 seed 与地图 seed 分开，以及“目标导航与自主探索需要不同任务定义”的建议。所有预期可行实例都要记录筛选条件，不能只对失败地图事后剔除。小地图、短路线、好看的渲染和严格安全性能分别报告。

本轮只新增核查文档；没有修改网络、奖励、训练配置或现有地图。

## 2026-10-13实施进展

20×10m任务已注册`full_scratch_ego20x10_curr_v1`，冻结128/20/40地图池并接入路径起点与10/20/30/40树密度课程。seed43已完成2k→6k P0，工程门槛通过，但学习后出现动作饱和/越界；停在6k先审计，不直接开启长训。见[实现与P0结果](2026-10-13_full_scratch_ego20x10_curr_v1_p0.md)。
