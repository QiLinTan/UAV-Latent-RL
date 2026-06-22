# AvoidBench Current Status

## Status

As of 2026-06-08, the AvoidBench Unity-depth bridge is working in the
`noetic_ab_glx` Docker container.

Completed:

- Unity starts and displays the AvoidBench scene.
- `avoidbridge.AvoidbenchBridge` connects to Unity.
- `updateUnity()` returns `True`.
- Mission setup, obstacle spawning, and scene-change confirmation work.
- Unity RGB, depth, and collision observations are available.
- The Python process exits normally; the previous double-free is fixed.

Verified output:

```text
Unity ready: True
Scene changed: True
image 0: shape=(480, 640, 3), dtype=uint8
image 1: shape=(480, 640, 1), dtype=float32
collision: False
PROBE_EXIT=0
```

The Unity image probe has also completed successfully with
`--image-mode unity --steps 100`.

## Startup

Use the repository launcher from the host. It permanently avoids stale
container copies by bind-mounting the current checkout and also configures the
container workdir and `PYTHONPATH`:

```bash
cd /home/tequial/projects/UAV-AvoidBench-RL
./tools/avoidbench_container.sh create
./tools/avoidbench_container.sh check
./tools/avoidbench_container.sh enter
```

The default managed container is `noetic_ab_workspace`. To replace an existing
managed container without deleting it:

```bash
./tools/avoidbench_container.sh recreate
```

The previous container is renamed with a timestamped `_backup_...` suffix.

Inside a manually created container, initialize the same environment with:

```bash
source /workspace/UAV-AvoidBench-RL/tools/setup_avoidbench_env.sh
```

Do not rely on a project directory copied into a Docker image. The project must
appear in `docker inspect <container> --format '{{json .Mounts}}'` as a bind
mount from `/home/tequial/projects/UAV-AvoidBench-RL`.

Start Unity in one container terminal:

```bash
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench.x86_64
```

In another sourced container terminal, run the bridge probe from the project
directory:

```bash
python3 -m scripts.probe_avoidbench \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --image-mode unity \
  --steps 100

echo "PROBE_EXIT=$?"
```

Expected final result:

```text
PROBE_EXIT=0
```

## ROS Interface Probe

Before implementing an RL environment, start the relevant ROS master, Gazebo,
flight pilot, autopilot, and bridge launch files. Then run:

```bash
python3 -m scripts.probe_avoidbench_ros
```

For CI or launch validation, require every expected endpoint:

```bash
python3 -m scripts.probe_avoidbench_ros --strict
```

The probe is read-only. It lists topics and services and inspects the type and
connection information for:

- `/flight_pilot/state_estimate`
- `/goal_point`
- `/gazebo/set_model_state`
- `/autopilot/reset_reference_state`
- `/autopilot/start`
- `/autopilot/force_hover`
- `/bridge/arm`

If both list commands report `Unable to communicate with master`, the ROS
control stack has not been started yet. Start ROS master and the Gazebo,
flight-pilot, autopilot, and bridge launch files before interpreting an
endpoint as genuinely missing.

## Current Limitations

`avoidbridge` is a Unity rendering, scene, point-cloud, and collision bridge.
It is not a complete reinforcement-learning environment.

Still missing:

- an action command interface;
- dynamics advancement through ROS/Gazebo;
- deterministic episode reset;
- reward calculation;
- success, collision, and timeout termination;
- a Gymnasium-compatible `reset()` and `step(action)` API.

Do not start TD3 training yet. The next stage is to use the ROS probe results
to define a small control/dynamics adapter and then wrap it as
`AvoidBenchRLEnv`.
