# AvoidBench ROS Runtime Result

## Runtime setup

Date: 2026-06-09

Official launch command confirmed from source and executed successfully:

```bash
source /opt/ros/noetic/setup.bash
source /AvoidBench/devel/setup.bash
roslaunch avoid_manage rotors_gazebo.launch
```

Execution notes:

- The original `noetic_ab_glx` container could not be restarted because its
  saved HostConfig still required the `nvidia` runtime, which currently fails
  with `NVML: Driver/library version mismatch`.
- To avoid changing the image or rebuilding anything, runtime validation was
  performed in a fresh temporary container created from:

```text
noetic_avoidbench_unitydepth_fixed:local
```

- The temporary container was started with:
  - host networking
  - `/tmp/.X11-unix` mounted
  - `DISPLAY=:0`
  - `LIBGL_ALWAYS_SOFTWARE=1`

Observed launch behavior:

- Gazebo started.
- RotorS spawned `hummingbird`.
- `autopilot`, `rpg_rotors_interface`, and `avoid_manage_node` started.
- `avoid_manage_node` loaded `task_outdoor.yaml`.
- ROS image topics `/rgb/left`, `/rgb/right`, and `/depth` appeared.
- X11 emitted `Authorization required, but no authorization protocol specified`,
  but the ROS graph still reached a usable runtime state.

## Validation command

Read-only strict probe executed inside the running container:

```bash
cd /workspace/UAV-AvoidBench-RL
python3 -m scripts.probe_avoidbench_ros \
  --strict \
  --wait-timeout 30 \
  --namespace /hummingbird
```

Result:

```text
state: ROS_INTERFACES_READY
expected endpoints found: 10/10
topics discovered: 83
services discovered: 47
exit code: 0
```

## Required endpoints

All expected step/reset endpoints were present at runtime.

Found topics:

- `/hummingbird/ground_truth/odometry`
  - type: `nav_msgs/Odometry`
  - publisher: `/gazebo`
  - subscribers: `/hummingbird/autopilot`, `/hummingbird/avoid_manage_node`,
    `/hummingbird/rpg_rotors_interface`
- `/hummingbird/autopilot/reset_reference_state`
  - type: `rpg_quadrotor_msgs/TrajectoryPoint`
  - publisher: `/hummingbird/avoid_manage_node`
  - subscriber: `/hummingbird/autopilot`
- `/hummingbird/autopilot/start`
  - type: `std_msgs/Empty`
  - publisher: `/hummingbird/avoid_manage_node`
  - subscriber: `/hummingbird/autopilot`
- `/hummingbird/autopilot/force_hover`
  - type: `std_msgs/Empty`
  - publisher: `/hummingbird/avoid_manage_node`
  - subscriber: `/hummingbird/autopilot`
- `/hummingbird/bridge/arm`
  - type: `std_msgs/Bool`
  - publisher: `/hummingbird/avoid_manage_node`
  - subscriber: `/hummingbird/rpg_rotors_interface`
- `/hummingbird/goal_point`
  - type: `nav_msgs/Path`
  - publisher: `/hummingbird/avoid_manage_node`
- `/hummingbird/autopilot/velocity_command`
  - type: `geometry_msgs/TwistStamped`
  - subscriber: `/hummingbird/autopilot`
- `/hummingbird/autopilot/reference_state`
  - type: `rpg_quadrotor_msgs/TrajectoryPoint`
  - subscriber: `/hummingbird/autopilot`
- `/hummingbird/autopilot/pose_command`
  - type: `geometry_msgs/PoseStamped`
  - subscriber: `/hummingbird/autopilot`

Found service:

- `/gazebo/set_model_state`
  - node: `/gazebo`
  - type: `gazebo_msgs/SetModelState`

Missing endpoints:

- none from the strict probe set

## Additional runtime topics

These appeared and are useful for later control-loop work:

- `/clock`
- `/rgb/left`
- `/rgb/right`
- `/depth`
- `/hummingbird/collision`
- `/hummingbird/metrics`
- `/hummingbird/task_state`
- `/hummingbird/iter_time`
- `/hummingbird/autopilot/control_command_input`
- `/hummingbird/autopilot/feedback`
- `/hummingbird/autopilot/trajectory`
- `/hummingbird/control_command`
- `/hummingbird/command/motor_speed`
- `/hummingbird/gazebo/command/motor_speed`

## Additional runtime services

These appeared and are useful for later reset diagnostics:

- `/gazebo/pause_physics`
- `/gazebo/unpause_physics`
- `/gazebo/reset_simulation`
- `/gazebo/reset_world`
- `/gazebo/get_model_state`
- `/gazebo/set_model_configuration`
- `/gazebo/get_physics_properties`
- `/gazebo/spawn_urdf_model`

## Step-1 conclusion

The official AvoidBench ROS/Gazebo/autopilot/avoid_manage system is now
confirmed to expose the minimum endpoints required for a first reset/step
adapter. The next immediate task is not training; it is verifying the odometry
stream and then testing whether a very small high-level action on
`/hummingbird/autopilot/velocity_command` produces a measurable state change.

## Step-2 odometry probe

Validation command:

```bash
cd /workspace/UAV-AvoidBench-RL
python3 -m scripts.probe_avoidbench_state \
  --odom-topic /hummingbird/ground_truth/odometry \
  --duration 5 \
  --strict
```

Observed result:

```text
messages_received: 2476
estimated_frequency_hz: 499.85
status: ODOMETRY_OK
frame_id: world
child_frame_id: hummingbird/base_link
position: (+0.0000, +0.0000, +0.0600)
linear_velocity: (+0.0000, -0.0000, +0.0000)
orientation_xyzw: (+0.00000, +0.00000, +0.00000, +1.00000)
angular_velocity: (-0.0000, -0.0000, +0.0000)
exit code: 0
```

Interpretation:

- odometry is readable at roughly 500 Hz
- pose and twist are stable
- the system is initially idle, so action-response validation must handle the
  autopilot mode transition explicitly
