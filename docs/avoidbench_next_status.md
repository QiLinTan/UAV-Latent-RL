# AvoidBench Next Status

## 2026-06-22 Baseline Update

Branch:

```text
avoidbench-rl-env-smoke
```

Local freeze commits:

```text
a2601b31ac5a11c4ae650086bb08cc2ffe8f2f54
7c07126ab30dff7cf858557a892b98ed66c6f455
```

The first commit freezes the AvoidBench RL environment and Plain TD3 smoke
baseline. The second records the freeze status and evidence paths.

### Workspace Container

`noetic_ab_workspace` was recreated from
`noetic_avoidbench_unitydepth_fixed:local` without NVIDIA GPU device requests.
The previous GPU-bound container could not start while the host NVIDIA driver
was unavailable:

```text
nvidia-container-cli: initialization error: nvml error: driver not loaded
```

`tools/avoidbench_container.sh` now defaults to `AVOIDBENCH_ENABLE_GPU=0`.
Set `AVOIDBENCH_ENABLE_GPU=1` only when the NVIDIA runtime is known to work.

Mesa launch script:

```text
scripts/launch_avoidbench_ros_mesa.sh
```

The script prints the display and Mesa variables, sources the AvoidBench
environment, and starts:

```bash
roslaunch avoid_manage rotors_gazebo.launch
```

Host X11 access is required for Unity in the container:

```bash
xhost +SI:localuser:root
```

Latest container checks:

- `./tools/avoidbench_container.sh recreate`: passed
- `./tools/avoidbench_container.sh check`: passed
- Mesa launch: Unity connected to Flightmare
- strict ROS probe: `ROS_INTERFACES_READY`
- expected ROS endpoints: `10/10`
- topics/services: `83/47`
- standalone Unity RGB/depth probe with `--spawn-obstacles false`: passed
- default indoor dynamic scene-change probe: still fails with
  `Scene changed: False`

### Navigation Smoke Analysis

New analysis script:

```text
scripts/analyze_avoidbench_navigation.py
```

Outputs:

```text
docs/avoidbench_navigation_smoke_analysis.md
runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke/navigation_analysis_summary.json
```

Result for `20260609-141223-navigation_smoke`:

- episodes: `25`
- done reasons: `{"timeout": 25}`
- mean distance delta: `-0.008545 m`
- positive-progress episodes: `8/25`
- collision count: `0`
- mean height: `1.200355 m`
- most active scaled action dimension: `yaw_rate`
- mean raw actor action: `vx=-0.015252`, `vy=0.177500`
- goal delta is present in observation indices `13:16`
- goal delta is numerically consistent with `distance_to_goal`
- progress signal is weak: absolute progress reward is about `2.13%` of
  progress-plus-regularization magnitude

Conclusion:

- the old navigation smoke learned stable flight, not navigation;
- the policy biases sideways instead of toward the mostly `+x` goal;
- reward weighting is suspect because regularization dominates tiny progress;
- action frame/mapping still requires live probing;
- longer navigation training is not justified yet.

### Goal-Direction Hand Policy Probe

New probe:

```text
scripts/probe_avoidbench_goal_direction_policy.py
```

Latest run:

```text
runs/avoidbench_goal_direction_probe/20260622-133135/
```

Result:

| Strategy | Frame | Distance Delta | Done Reason | Collision |
| --- | --- | ---: | --- | --- |
| zero | world | 0.000000 m | collision | true |
| constant_forward | world | 0.000092 m | collision | true |
| goal_direction | world | 0.001510 m | collision | true |
| goal_direction | body | 0.001509 m | collision | true |

The probe did not pass. Goal-direction produced a tiny positive x displacement,
but every episode terminated on collision and `/hummingbird/collision`
remained true afterward.

### Stage 1 Decision

Stage 1 `stage1_short_nav` training was not started.

Reason:

- the hand-coded goal-direction policy did not satisfy the gate;
- collision state is currently unreliable after reset;
- native `avoid_manage_node` mission/collision ownership still interferes with
  RL episode ownership.

Current permission status:

- longer Plain TD3 navigation: blocked
- Stage 1 short navigation training: blocked until collision/reset ownership is
  fixed
- latent-only/depth/auxiliary-loss work: blocked
- four-motor RPM work: blocked

Recommended next step:

1. isolate or disable native `avoid_manage_node` mission lifecycle during RL
   control probes;
2. make reset wait for `/hummingbird/collision == False`;
3. rerun the goal-direction probe until at least one frame gives clear distance
   decrease without collision;
4. only then create `stage1_short_nav`.

Date: 2026-06-09

Branch used for this gated run:

```text
avoidbench-rl-env-smoke
```

## Current gated result

Stages 0 through 6 completed. The previous first-reset failure is superseded by
the hardened reset results below.

- reset hardening: passed
- reset-only stress: passed twice at `20/20`
- zero/constant/random step stress: passed twice
- reward/done/info contract: implemented and stress-tested
- Plain TD3 smoke: passed for `2000` environment steps
- latent integration: not started

No existing file was deleted and no Git commit was created.

## Stage 0: protected runtime

- branch: `avoidbench-rl-env-smoke`
- fixed image found: `noetic_avoidbench_unitydepth_fixed:local`
- official launch: `roslaunch avoid_manage rotors_gazebo.launch`
- strict ROS probe: `ROS_INTERFACES_READY`
- expected endpoints: `10/10`
- topics: `83`
- services: `47`

Stage log:

```text
runs/avoidbench_reset_debug/20260609-123308/stage0_status.txt
```

## Stage 1: reset diagnosis

Added:

```text
scripts/debug_avoidbench_reset.py
```

The diagnostic recorded service calls, target pose, z/vz timelines, autopilot
state transitions, publisher subscriber counts, and actual publish counts.

Initial diagnostic result:

- resets: `3/3`
- takeoff height reached: `3/3`
- every reset publisher had one subscriber
- first reset transition included `BREAKING -> HOVER`
- later resets followed different state paths

This confirmed a startup timing race rather than a missing ROS interface.

Artifacts:

```text
runs/avoidbench_reset_debug/20260609-123308/reset_debug.json
runs/avoidbench_reset_debug/20260609-123308/reset_hardened_debug.json
```

## Stage 2: reset hardening

`AvoidBenchRLEnv.reset()` now:

1. calls `/gazebo/set_model_state`
2. waits for fresh odometry to settle near the target
3. repeatedly publishes `reset_reference_state`
4. repeatedly publishes `bridge/arm`
5. publishes `autopilot/start` when the autopilot is `OFF`
6. repeatedly publishes `force_hover`
7. requires the autopilot to reach `HOVER`
8. requires stable z and vertical velocity above the takeoff threshold
9. retries the full sequence up to two additional times

Default reset parameters:

- takeoff height: `1.10 m`
- takeoff timeout: `10.0 s`
- reset retry: `2`
- repeated publish duration: `0.75 s`
- repeated publish interval: `0.075 s`
- odometry settle timeout: `3.0 s`
- hover state timeout: `10.0 s`
- settle frames: `5`

The hardened diagnostic published each reset message ten times and completed
all three resets in `HOVER`.

## Stage 3: reset-only stress

Final post-reward reset gate:

```text
runs/avoidbench_env_stress/20260609-045838/
```

Result:

- successful resets: `20/20`
- maximum consecutive successes: `20`
- final z: approximately `1.1996 m`
- ROS timeout: none
- Gazebo service failure: none
- Python exception: none

An earlier hardened run also passed `20/20`:

```text
runs/avoidbench_env_stress/20260609-044110/
```

## Stage 4: step/action stress

Final post-reward stress artifacts:

```text
runs/avoidbench_env_stress/20260609-045936/
```

Result:

- zero action: `3 x 100`, no drift, no collision
- constant forward: `3` episodes, two reached the goal
- random action: `5 x 200`, no collision or runtime failure
- maximum step time: `0.453 s`
- mean step time: approximately `0.451 s`
- reset failure: none
- deadlock: none

## Stage 5: reward, done, and info

Observation shape remains fixed at `(17,)`.

Default reward parameters are centralized in
`AvoidBenchRewardDoneConfig`:

- progress scale: `1.0`
- collision penalty: `5.0`
- height penalty scale: `0.10`
- action penalty scale: `0.01`
- goal bonus: `5.0`
- timeout penalty: `0.5`
- target height: `1.2 m`
- height tolerance: `0.30 m`

Done conditions:

- collision
- goal reached
- episode timeout
- height too low: below `0.40 m`
- height too high: above `3.00 m`
- xy out of bounds: above `10.0 m`
- odometry timeout: `2.0 s`

Every step info now contains:

- `position`
- `velocity`
- `distance_to_goal`
- `previous_distance_to_goal`
- `progress`
- `collision`
- `height`
- `done_reason`
- `autopilot_state`
- `action_norm`
- `step_time`
- `reset_retry_count`

## Stage 6: Plain TD3 smoke

Added:

```text
scripts/train_avoidbench_plain_td3_smoke.py
```

The fixed AvoidBench image does not contain PyTorch. No package was installed.
The smoke run used a separate client container with the existing read-only
`drones` Conda environment and only the pure Python ROS compatibility packages
exposed to Python 3.10.

Final run:

```text
runs/avoidbench_plain_td3_smoke/20260609-051809/
```

Result:

- status: `OK`
- completed steps: `2000`
- replay buffer size: `2000`
- TD3 updates: `1900`
- episodes completed: `43`
- collisions: `0`
- checkpoint files: saved
- logs: saved
- environment crash: none
- ROS/Gazebo failure: none

The smoke objective passed, but policy quality did not:

- actor actions saturated near their limits early
- done reasons:
  - `height_too_high`: `36`
  - `height_too_low`: `5`
  - `out_of_bounds`: `1`
  - `timeout`: `1`

This historical result motivated the conservative action and height-reward
gates below. Its saturation and height-instability conclusions are superseded
by the new 5,000-step hover and navigation measurements.

## Current readiness

The environment is ready for further **lowdim Plain TD3 navigation tuning**:

- reset is repeatable
- step is stable
- replay/update/checkpoint/logging paths work
- conservative actions no longer saturate
- height terminations were eliminated in both new 5,000-step runs

It is not ready for long navigation training because the current navigation
reward produced no meaningful average progress or goal completion.

It is not ready for latent-only work because:

- the current training observation is lowdim-only
- depth/image replay and synchronization have not been validated
- no encoder or latent transition path is connected to this ROS env
- the lowdim policy has stable control but has not learned navigation
- adding latent complexity now would confound environment and policy issues

## Recommended next step

1. tune lowdim navigation progress and directional incentives
2. inspect observation and reward scales before increasing training length
3. add evaluation episodes separated from exploration
4. run multiple short Plain TD3 navigation seeds
5. require reproducible positive progress before depth/latent work

## Files intentionally not modified

- existing TD3 trainer
- latent modules
- auxiliary losses
- four-motor RPM control

## Final low-cost checks

Completed on 2026-06-09:

- `python3 -m py_compile` passed for the AvoidBench environment and new probe,
  stress, reset-debug, and Plain TD3 smoke scripts
- `git diff --check` passed
- related tests passed: `8 passed in 1.84s`

The first pytest attempt was blocked during plugin loading because the host
automatically loaded the ROS 2 `launch_testing` plugin from `/opt/ros/humble`.
The project tests passed after rerunning with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; no test dependency was installed or
modified.

## 2026-06-09 lowdim Plain TD3 baseline gates

Completed after the original 2,000-step smoke:

- old-run analysis:
  `runs/avoidbench_plain_td3_smoke/20260609-051809/analysis_summary.json`
- conservative reset gate:
  `runs/avoidbench_env_stress/20260609-131459/` (`20/20`)
- conservative full stress:
  `runs/avoidbench_env_stress/20260609-131613/` (`OK`)
- hover smoke:
  `runs/avoidbench_plain_td3_smoke/20260609-133209-hover_smoke/`
- navigation smoke:
  `runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke/`

Action presets are centralized in `envs/avoidbench/rl_env.py`:

- `legacy`: `(0.20, 0.20, 0.15, 0.30)`
- `conservative`: `(0.12, 0.12, 0.04, 0.12)`

Height reward now continuously penalizes absolute height error, vertical
velocity, and normalized z action. Only severe height violations below `0.4 m`
or above `2.5 m` terminate an episode.

### Old smoke versus new smoke

| Metric | Old | Hover | Navigation |
| --- | ---: | ---: | ---: |
| Height termination | 95.35% | 0% | 0% |
| Mean episode length | 45.30 | 200 | 200 |
| Raw actor saturation | per-dimension unavailable | 0% | 0% |
| Collision | 0 | 0 | 0 |

Hover passed its gate and showed stable height control. Navigation remained
stable but did not learn meaningful progress: mean distance improvement was
`-0.0085 m`, only `8/25` episodes had positive progress, and no episode reached
the goal.

### Current decision

- the environment and Plain TD3 runtime are stable
- further lowdim navigation reward/action tuning is justified
- long training with the current navigation objective is not justified
- latent-only, depth encoder, and auxiliary-loss work remain blocked

See `docs/avoidbench_plain_td3_baseline_report.md` for the full comparison and
decision rationale.

## Persistent Docker workspace

Do not manually copy this repository into an existing container. The old
`noetic_ab_glx` container was created without a project bind mount, so its
`/workspace/UAV-AvoidBench-RL` directory can become stale.

Use the managed container instead:

```bash
cd /home/tequial/projects/UAV-AvoidBench-RL
./tools/avoidbench_container.sh check
./tools/avoidbench_container.sh enter
```

The managed container is named `noetic_ab_workspace`. It provides:

- a live bind mount from the host repository;
- `/workspace/UAV-AvoidBench-RL` as the default working directory;
- the project root in `PYTHONPATH`;
- ROS Noetic and `/AvoidBench/devel` setup through the entry helper.

After entering it, module commands work without another `cd`:

```bash
python3 -m scripts.stress_avoidbench_rl_env --help
```

Use `source tools/setup_avoidbench_env.sh` if entering the container through a
manual `docker exec` command instead of the launcher.
