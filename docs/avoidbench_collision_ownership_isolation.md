# AvoidBench Collision Ownership Isolation

Date: 2026-06-26

## Goal

Do not train. The next task is to make the collision/done signal trustworthy.

The specific question is:

```text
Who sets collision=True, when is it set, and why does reset not clear it?
```

This is an environment ownership problem. Stage 1 remains blocked until
reset-only collision sanity passes.

## New Probe

Script:

```text
scripts/probe_avoidbench_collision_ownership.py
```

The probe records a 5-second static window after reset at 0.05-0.1 second
intervals and compares:

- ROS collision topic: `/hummingbird/collision`
- Gazebo model state: `/gazebo/get_model_state`
- Gazebo contact topics, if any `gazebo_msgs/ContactsState` topics exist
- manager state topics: `/hummingbird/task_state` and `/hummingbird/metrics`
- optional direct `avoidbridge.getQuadCollisionState()`
- direct bridge-only static mode for A/B testing without `avoid_manage_node`

Outputs:

```text
runs/avoidbench_collision_ownership/<timestamp>/summary.json
runs/avoidbench_collision_ownership/<timestamp>/samples.jsonl
runs/avoidbench_collision_ownership/<timestamp>/samples.csv
```

## Current Result

Manual A/B data and the follow-up bridge sweep point to a Unity/avoidbridge
collision ownership problem, not a TD3/action-frame/Gazebo-contact problem.

Artifacts:

```text
runs/avoidbench_collision_ownership/20260626-103344/summary.json
runs/avoidbench_collision_ownership/20260626-105044/summary.json
runs/avoidbench_collision_ownership/20260626-112506/summary.json
runs/avoidbench_collision_ownership/20260626-112735/summary.json
runs/avoidbench_collision_ownership/20260626-113015/summary.json
```

Observed facts:

- `official-reset`: `/hummingbird/collision` was true for all 100 samples, with
  first true at about `0.005 s`.
- `official-reset`: Gazebo model state stayed near `(0, 0, 1.2)`, speed stayed
  near zero, and no Gazebo contact signal reported active contact.
- `official-reset`: `reset_info.collision=True` while the older env info still
  said `done_reason=running`. This has been corrected in
  `envs/avoidbench/rl_env.py`; reset info now reports `reset_valid=False` and
  `done_reason=collision` when collision is already true after reset.
- `bridge-static`: direct `avoidbridge.getQuadCollisionState()` was true from
  the first sample, with Unity ready and `scene_changed=False`.
- `bridge-sweep`: direct bridge collision was true at all tested positions:
  `(0,0,1.2)`, `(0,0,2)`, `(0,0,5)`, `(5,0,2)`, `(0,5,2)`, and `(-5,0,2)`.
- `bridge-sweep --spawn-obstacles`: still reported `scene_changed=False` and
  collision true at all tested positions.
- `bridge-static` started directly at `(0,0,5)` and still reported
  `initial_direct_bridge.collision=True`, so the high-altitude result is not
  just contamination from a previous low-altitude sample in the same process.
- Unity direct-bridge log repeatedly reported collision with `asphalt_tile`:
  `/tmp/avoidbench_direct_bridge_collision.log`.

Current interpretation:

- This is not explained by the hand-written goal policy.
- This is not explained by immediate Gazebo physical contact in the collected
  data.
- This is unlikely to be only "the reset pose `(0,0,1.2)` is inside an
  obstacle", because high and offset sweep positions also report collision.
- The active suspect is Unity collision geometry/layers around the ground tile
  (`asphalt_tile`), plus scene-change/spawn not completing in the direct bridge
  path. This is no longer a policy or reward issue.

## Experiment A: Official Manager Runtime

Start the official runtime in one container shell:

```bash
cd /workspace/UAV-AvoidBench-RL
./scripts/launch_avoidbench_ros_mesa.sh
```

Confirm readiness in a second sourced shell:

```bash
source /workspace/UAV-AvoidBench-RL/tools/setup_avoidbench_env.sh
python3 -m scripts.probe_avoidbench_ros \
  --strict \
  --wait-timeout 30 \
  --namespace /hummingbird
```

Run the ownership probe:

```bash
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode official-reset \
  --namespace /hummingbird \
  --model-name hummingbird \
  --observe-seconds 5.0 \
  --sample-period 0.05 \
  --action-preset conservative
```

Default `official-reset` intentionally does not create a second avoidbridge
client. The official `avoid_manage_node` already owns Unity. Use
`--direct-bridge` only when intentionally probing whether a second Python
avoidbridge client can connect without disturbing the manager.

Interpretation:

- ROS collision true while Gazebo model state is stable near `(0, 0, 1.2)` and
  no Gazebo contact topic reports contact means the failure is not proven as a
  Gazebo physical collision.
- collision true before or immediately after reset means reset is returning an
  invalid RL episode.
- collision flips after task_state/metrics change or scene refresh points to
  native manager mission ownership.

## Experiment B: Direct Bridge Static Runtime

This experiment should be run without the official `avoid_manage_node` runtime
owning Unity.

Stop `rotors_gazebo.launch`, confirm there is no active non-defunct
`AvoidBench.x86_64`, then start only the Unity binary with Mesa:

```bash
unset __VK_LAYER_NV_optimus
unset __NV_PRIME_RENDER_OFFLOAD
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
cd /AvoidBench/src/avoidbench/unity_scene/AvoidBench
./AvoidBench.x86_64 -logFile /tmp/avoidbench_direct_bridge_collision.log
```

In a second sourced shell:

```bash
source /workspace/UAV-AvoidBench-RL/tools/setup_avoidbench_env.sh
cd /workspace/UAV-AvoidBench-RL
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-static \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --reset-position 0.0 0.0 1.2 \
  --observe-seconds 5.0 \
  --sample-period 0.05
```

Optional scene refresh check:

```bash
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-static \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --reset-position 0.0 0.0 1.2 \
  --observe-seconds 5.0 \
  --sample-period 0.05 \
  --spawn-obstacles
```

Interpretation:

- A fails and B passes: current `AvoidBenchRLEnv` reset conflicts with
  `avoid_manage_node` ownership, or the manager is refreshing stale collision
  state.
- A fails and B also reports collision at the same static pose: Unity scene or
  avoidbridge collision geometry marks the reset pose as colliding.
- B passes without obstacle spawning but fails after `--spawn-obstacles`: the
  spawned scene or mission parameters place the start in/near collision.

## Experiment C: Direct Bridge Position Sweep

Use this after starting only the Unity binary as in Experiment B:

```bash
source /workspace/UAV-AvoidBench-RL/tools/setup_avoidbench_env.sh
cd /workspace/UAV-AvoidBench-RL
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-sweep \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --observe-seconds 0.7 \
  --sample-period 0.1
```

The 2026-06-26 sweep reported `collision=True` for every default sweep
position. That result makes a single bad reset pose unlikely. The same sweep
with official-style scene spawning was:

```bash
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-sweep \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --observe-seconds 0.7 \
  --sample-period 0.1 \
  --spawn-obstacles
```

Result: `scene_changed=False`, `collision=True` for all positions.

Because the direct Unity log repeatedly names `asphalt_tile`, inspect Unity
ground collision handling before any more RL training:

- whether the vehicle collision collider overlaps the ground tile after the
  ROS-to-Unity pose transform;
- whether the drone size/collision radius sent to Unity is too large;
- whether the ground/asphalt layer is incorrectly included in the drone
  collision flag or `checkCollisionState()` obstacle mask;
- whether direct bridge `spawnObstacles()/SpawnNewObs()` is failing to trigger
  scene-change feedback in the standalone Unity mode.

## Source Audit Notes

Relevant native source paths inside the container:

```text
/AvoidBench/src/avoidbench/avoid_manage/src/avoid_manage.cpp
/AvoidBench/src/avoidbench/avoid_manage/include/avoid_manage.hpp
/AvoidBench/src/avoidbench/avoidlib/src/bridges/avoidbench_bridge.cpp
/AvoidBench/src/avoidbench/avoidlib/src/bridges/unity_bridge.cpp
/AvoidBench/src/avoidbench/avoidlib/src/objects/quadrotor.cpp
/AvoidBench/src/avoidbench/avoidlib/include/avoidlib/bridges/unity_message_types.hpp
```

Findings:

- `avoid_manage` initializes `collision_state` and `last_collision_state` to
  `true`, but direct bridge mode bypasses `avoid_manage`, so that default does
  not explain the B/sweep result by itself.
- `Quadrotor` initializes its internal `collision_` to `false`.
- `AvoidbenchBridge::getQuadCollisionState()` returns
  `unity_bridge_ptr_->collisions[0]` if Unity has sent collision data, otherwise
  false.
- `UnityBridge::handleOutput()` clears and repopulates `collisions` from the
  Unity JSON field `pub_vehicles[0].collision`.
- Official `avoid_manage` does more than Gazebo reset: it calls
  `setParamFromMission()`, `spawnObstacles()`, repeatedly calls `SpawnNewObs()`
  until `ifSceneChanged()`, then checks start/end with `checkCollisionState()`
  before resetting Gazebo to the selected start.

Practical implication: a plain Gazebo `set_model_state` reset is not enough to
make collision trustworthy. The RL environment needs either to run under a
clean official manager-owned mission, or to explicitly implement the same
scene-change and collision-check handshake before accepting reset.

## What To Inspect In Results

Use `samples.csv` for timing:

- `elapsed_s`
- `position_*`
- `velocity_*`
- `height`
- `ros_collision`
- `autopilot_state`
- `gazebo_model_position_*`
- `gazebo_contact_active_count`
- `task_state_latest`
- `metrics_latest`
- `direct_bridge_unity_ready`
- `direct_bridge_scene_changed`
- `direct_bridge_collision`

Key timing questions:

- collision true at the first sample?
- collision flips after 1-5 seconds?
- collision flips near a task_state/metrics change?
- collision flips when scene_changed becomes true?
- Gazebo model state remains stable while ROS collision flips?
- any Gazebo contact topic exists and reports active contacts?

## Required Passing Standard

Do not resume Stage 1 until all are true:

- reset-only `20/20` succeeds;
- reset returns collision false;
- zero action or no action for 5 seconds stays collision false;
- four-direction low-speed actions do not immediately collision;
- goal-direction policy lowers distance in most episodes without large
  collision rate.

Only after this gate should Stage 1 short navigation training resume.
