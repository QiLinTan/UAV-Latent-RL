# AvoidBench ROS Startup Plan

## Source audit

The official AvoidBench ROS entry point in this checkout is:

```bash
roslaunch avoid_manage rotors_gazebo.launch
```

Source: `/AvoidBench/src/avoidbench/avoid_manage/launch/pilot/rotors_gazebo.launch`.
Its default quadrotor namespace is `hummingbird`.

The launch starts the complete stack:

1. `gazebo_ros/empty_world.launch` with
   `avoid_manage/resources/worlds/simple.world`.
2. `rotors_gazebo/spawn_mav.launch`, using
   `mav_generic_odometry_sensor.gazebo`.
3. `rpg_rotors_interface/rpg_rotors_interface`.
4. `autopilot/autopilot` by default, or
   `rpg_mpc/autopilot_mpc_instance` with `use_mpc:=true`.
5. `avoid_manage/avoid_manage_node`.
6. `unity_scene/AvoidBench.x86_64` unless `use_unity_editor:=true`.

`test_py.launch` is an alternate development launch. It uses
`avoid_manage.py`, starts Gazebo paused, and adds the RQt flight GUI. The C++
`rotors_gazebo.launch` is the primary path for adapter work.

Both launch files currently load `task_outdoor.yaml` directly. Running the
indoor task requires a small launch/configuration change; there is no existing
launch argument that switches the task YAML.

## Interface ownership

### Gazebo and state feedback

- Gazebo is started by the `gazebo_ros/empty_world.launch` include.
- The `gazebo_ros/gzserver` wrapper loads `libgazebo_ros_api_plugin.so`.
  That standard plugin provides `/gazebo/set_model_state`.
- The RotorS odometry plugin publishes `ground_truth/odometry`.
- The launch remaps both `autopilot/state_estimate` and
  `flight_pilot/state_estimate` to `ground_truth/odometry`.

With the default namespace, the actual state topic is:

```text
/hummingbird/ground_truth/odometry
```

There is no separate node expected to publish the literal
`/flight_pilot/state_estimate` in this launch. It is a logical input name on
`avoid_manage`, resolved through the launch remap.

### Avoid manager

`avoid_manage_node` publishes:

```text
/rgb/left
/depth
/hummingbird/goal_point
/hummingbird/task_state
/hummingbird/metrics
/hummingbird/collision
/hummingbird/autopilot/reset_reference_state
/hummingbird/autopilot/start
/hummingbird/autopilot/force_hover
/hummingbird/bridge/arm
```

It subscribes to the remapped state topic and uses
`/gazebo/set_model_state` during mission reset.

The `autopilot/*` lifecycle endpoints and `bridge/arm` are topics, not
services. `/gazebo/set_model_state` is the service.

### Candidate action topics

The RPG autopilot subscribes to these high-level commands:

```text
/hummingbird/autopilot/velocity_command
  geometry_msgs/TwistStamped
/hummingbird/autopilot/reference_state
  rpg_quadrotor_msgs/TrajectoryPoint
/hummingbird/autopilot/pose_command
  geometry_msgs/PoseStamped
```

The first RL adapter should prefer `velocity_command` or
`reference_state`. Do not publish motor RPM directly. Before selecting one,
confirm its frame convention, watchdog timeout, update rate, and behavior
after `autopilot/start`.

## Suggested startup

Inside `noetic_ab_glx`:

```bash
source /opt/ros/noetic/setup.bash
source /AvoidBench/devel/setup.bash
roslaunch avoid_manage rotors_gazebo.launch
```

The launch defaults to headless Gazebo (`gui:=false`) while starting the Unity
renderer. Useful alternatives:

```bash
roslaunch avoid_manage rotors_gazebo.launch gui:=true
roslaunch avoid_manage rotors_gazebo.launch use_unity_editor:=true
roslaunch avoid_manage rotors_gazebo.launch use_mpc:=true
```

In a second sourced terminal:

```bash
cd /workspace/UAV-AvoidBench-RL
python3 scripts/probe_avoidbench_ros.py --wait-timeout 30 --strict
```

If the repository is mounted elsewhere, change only the `cd` path.

## Readiness checks

The stack is ready for RL adapter development only after:

```bash
rostopic info /hummingbird/ground_truth/odometry
rostopic info /hummingbird/autopilot/velocity_command
rostopic info /hummingbird/autopilot/reference_state
rostopic info /hummingbird/autopilot/reset_reference_state
rostopic info /hummingbird/autopilot/start
rostopic info /hummingbird/autopilot/force_hover
rostopic info /hummingbird/bridge/arm
rostopic info /hummingbird/goal_point
rosservice info /gazebo/set_model_state
```

Also verify `/clock` advances and state timestamps continue changing. Topic
existence alone does not prove that Gazebo is unpaused or that control is
effective.
