# AvoidBench Workspace Container

Date: 2026-06-10

## Current container

The managed AvoidBench runtime is:

```text
container: noetic_ab_workspace
image: noetic_avoidbench_unitydepth_fixed:local
project: /workspace/UAV-AvoidBench-RL
```

The host checkout is bind-mounted read/write from:

```text
/home/tequial/projects/UAV-AvoidBench-RL
```

Use the repository helper from the host:

```bash
cd /home/tequial/projects/UAV-AvoidBench-RL
./tools/avoidbench_container.sh check
./tools/avoidbench_container.sh enter
```

The container check confirms the bind mount, ROS environment, Python module
imports, and stress-test entry point.

## Stable ROS launch

Inside `noetic_ab_workspace`, start the complete AvoidBench ROS, Gazebo,
RotorS, autopilot, avoid manager, and Unity stack with:

```bash
./scripts/launch_avoidbench_ros_mesa.sh
```

Additional `roslaunch` arguments are forwarded unchanged:

```bash
./scripts/launch_avoidbench_ros_mesa.sh gui:=true
```

Do not run multiple Unity or `rotors_gazebo.launch` instances in the same
container. Stop the existing launcher before starting another one.

The launcher:

1. forces Mesa software rendering;
2. sources ROS Noetic and `/AvoidBench/devel` through
   `tools/setup_avoidbench_env.sh`;
3. starts `roslaunch avoid_manage rotors_gazebo.launch`.

It also prints the active display and Mesa-related environment variables before
starting ROS.

## Mesa workaround

The current container's default NVIDIA/GLX path is not usable. Unity and
`glxinfo` fail while creating a GLX context:

```text
X Error of failed request: BadValue
Major opcode: GLX
Minor opcode: X_GLXCreateContext or X_GLXCreateNewContext
```

`nvidia-smi` in the container also reports:

```text
Failed to initialize NVML: Unknown Error
```

The stable workaround is:

```bash
unset __VK_LAYER_NV_optimus
unset __NV_PRIME_RENDER_OFFLOAD
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
```

This selects Mesa llvmpipe. It is CPU-rendered and slower than working GPU
rendering, but it provides a stable Unity and ROS integration runtime.

## Validation status

Validation performed on 2026-06-10 produced the following results.

Passed:

- the launcher remained alive and started ROS, Gazebo, RotorS, autopilot,
  `avoid_manage_node`, and Unity;
- standalone Unity remained alive and reported `Sockets bound`;
- `probe_avoidbench` connected to Unity and completed 100 RGB/depth updates
  with `--spawn-obstacles false`;
- RGB output was `(480, 640, 3)` `uint8`;
- depth output was `(480, 640, 1)` `float32`;
- `probe_avoidbench_ros --strict` reported `ROS_INTERFACES_READY`;
- the strict ROS probe found 83 topics, 47 services, and 10/10 expected
  endpoints;
- reset-only conservative stress passed 20/20 with no failures:
  `runs/avoidbench_env_stress/20260610-085245/`.

Not passed:

- the default indoor `probe_avoidbench` connected successfully but
  `Scene changed` remained false during dynamic object spawning;
- full conservative stress did not complete without collision terminations.

Two full-stress attempts were recorded:

```text
runs/avoidbench_env_stress/20260610-085341/
runs/avoidbench_env_stress/20260610-085713/
```

Both attempts completed their initial 20-reset gate. The first attempt reused
the runtime from reset-only stress and aborted after three consecutive
collision terminations. The second attempt used a freshly restarted launcher:
two zero-action episodes completed 100 steps, then collision became true in
the third zero-action episode and caused subsequent immediate terminations.

Source inspection indicates that the native `avoid_manage_node` mission timer
is interfering with RL episode ownership. Its mission loop can time out,
invoke its own Gazebo reset, and advance or regenerate the Unity mission while
`AvoidBenchRLEnv` is running an independent reset/step sequence. The observed
failure timing is consistent with that behavior, but the required mission-loop
isolation or collision reset has not been implemented in this change.

## Validation commands

With the launcher running in one terminal, use a second sourced shell:

```bash
source /opt/ros/noetic/setup.bash
source /AvoidBench/devel/setup.bash
cd /workspace/UAV-AvoidBench-RL
```

Confirm the ROS interfaces:

```bash
python3 -m scripts.probe_avoidbench_ros \
  --strict \
  --wait-timeout 30 \
  --namespace /hummingbird
```

Confirm the Unity bridge data path:

```bash
python3 -m scripts.probe_avoidbench \
  --image-mode unity \
  --steps 100
```

If the dynamic scene-change check fails but Unity data access is the only
target, retry with:

```bash
python3 -m scripts.probe_avoidbench \
  --image-mode unity \
  --steps 100 \
  --spawn-obstacles false
```

Run reset-only stress:

```bash
python3 -m scripts.stress_avoidbench_rl_env \
  --mode reset-only \
  --num-resets 20 \
  --action-preset conservative
```

Run full stress:

```bash
python3 -m scripts.stress_avoidbench_rl_env \
  --mode full \
  --num-resets 20 \
  --action-preset conservative
```

## Replacement decision

`noetic_ab_workspace` replaces the old ad hoc containers for:

- reproducible project bind mounts;
- Mesa-based Unity startup;
- Python Unity RGB/depth access;
- the complete ROS/Gazebo interface;
- repeated reset-only environment validation.

It is not yet certified as a complete replacement for long-running full
environment stress. Keep the previous known-good full-stress result as the
runtime baseline until the native AvoidBench mission loop is isolated from the
RL environment or collision state is reset reliably.

## Known limitations

- NVIDIA GPU rendering is unresolved; the launcher intentionally uses
  llvmpipe.
- The fixed AvoidBench image does not include PyTorch.
- This container validation does not cover TD3 training.
- This container validation does not cover latent or depth-encoder training.
- Full stress currently encounters collision state changes associated with the
  native mission lifecycle.
- Indoor dynamic obstacle scene-change confirmation currently fails, although
  Unity connection and RGB/depth acquisition pass.
- A bridge client that exits during scene setup can leave a standalone Unity
  ZMQ session unable to accept a second client; restart standalone Unity
  before retrying that isolated probe.

## Scope

This workspace is validated only for the AvoidBench environment runtime and
the low-dimensional ROS environment interface. TD3 and latent integration are
outside this container replacement gate.
