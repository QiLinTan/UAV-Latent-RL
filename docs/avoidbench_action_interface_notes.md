# AvoidBench Action Interface Notes

## Goal

Pick the first high-level action topic for a minimal `step(action)` loop without
touching motor RPM or trainer code.

## Current runtime candidates

From the live `rotors_gazebo.launch` runtime:

- Preferred high-level inputs:
  - `/hummingbird/autopilot/velocity_command`
  - `/hummingbird/autopilot/reference_state`
  - `/hummingbird/autopilot/pose_command`
  - `/hummingbird/autopilot/trajectory`
- Mode/reset helpers:
  - `/hummingbird/autopilot/start`
  - `/hummingbird/autopilot/force_hover`
  - `/hummingbird/autopilot/reset_reference_state`
  - `/hummingbird/bridge/arm`
- Lower-level or downstream topics, not recommended for the first RL adapter:
  - `/hummingbird/autopilot/control_command_input`
  - `/hummingbird/control_command`
  - `/hummingbird/command/motor_speed`
  - `/hummingbird/gazebo/command/motor_speed`

## Why `autopilot/velocity_command` is the first choice

`/AvoidBench/src/ThirdP/rpg_quadrotor_control/control/autopilot/include/autopilot/autopilot_inl.h`
subscribes directly to:

- `autopilot/pose_command`
- `autopilot/velocity_command`
- `autopilot/reference_state`
- `autopilot/reset_reference_state`
- `autopilot/control_command_input`

The same file also shows that the velocity command path:

- stores the latest `desired_velocity_command_`
- applies a timeout with automatic zeroing
- integrates velocity into `reference_state_`
- updates heading from `twist.angular.z`

The same autopilot source also matters for controller mode:

- current runtime feedback reported `autopilot_state: 0`
- `/AvoidBench/src/ThirdP/rpg_quadrotor_common/rpg_quadrotor_msgs/msg/AutopilotFeedback.msg`
  maps `0 -> OFF`
- `autopilot_inl.h` only accepts velocity commands in `HOVER` or
  `VELOCITY_CONTROL`

So a missing motion response does not automatically mean the topic is wrong. It
can simply mean the autopilot is still `OFF`.

That is the right abstraction level for a first RL adapter:

- high-level enough to avoid motor-level control
- direct enough to support a real `step(action)`
- bounded by timeout, which is useful for safe probing

## Why not start from lower-level command topics

Do not start from these for the first loop:

- `control_command_input`
- `control_command`
- `command/motor_speed`
- `gazebo/command/motor_speed`

Reason:

- they sit lower in the control stack
- they make the reset/startup contract harder
- they increase the chance of coupling RL code to controller internals
- they violate the current constraint of avoiding four-motor RPM control

## Recommended first action contract

Initial `step(action)` should publish a small high-level velocity command:

- action shape: `(vx, vy, vz, yaw_rate)`
- topic: `/hummingbird/autopilot/velocity_command`
- message type: `geometry_msgs/TwistStamped`
- first safe limits:
  - `|vx|, |vy| <= 0.5 m/s`
  - `|vz| <= 0.3 m/s`
  - `|yaw_rate| <= 0.5 rad/s`

## Required safety checks before publishing

Before any test action or future `step(action)`:

1. odometry must be readable from `/hummingbird/ground_truth/odometry`
2. action duration must be short
3. default test magnitude must be tiny
4. a zero command should be published immediately after the test window

That is why `scripts/probe_avoidbench_action.py` defaults to list-only mode and
requires `--send-test-action` before it publishes anything.

The probe also keeps any mode transition explicit:

- `--publish-start-before-test`
- `--publish-arm-before-test`

Both are opt-in and remain off by default.

## What to verify next

Before writing a real `AvoidBenchRLEnv.step()` implementation, confirm:

1. whether `/hummingbird/autopilot/velocity_command` alone moves the vehicle
2. whether `bridge/arm` and `autopilot/start` must be replayed after reset
3. whether `autopilot/reset_reference_state` is required after Gazebo pose reset
4. whether `avoid_manage_node` overrides externally published velocity commands
5. whether `goal_point` can be used as the episode target source

## Practical conclusion

The first minimal control-loop attempt should use:

- observation source: `/hummingbird/ground_truth/odometry`
- action topic: `/hummingbird/autopilot/velocity_command`
- reset helpers: `/gazebo/set_model_state`,
  `/hummingbird/autopilot/reset_reference_state`,
  `/hummingbird/autopilot/start`,
  `/hummingbird/bridge/arm`

That is the narrowest path from "bridge works" to "reset/step loop works"
without jumping into TD3 or low-level motor control.

## Validation result

Read-only topic classification:

```bash
cd /workspace/UAV-AvoidBench-RL
python3 -m scripts.probe_avoidbench_action --namespace /hummingbird
```

Observed:

- 15 candidate action/control topics
- `velocity_command`, `reference_state`, `pose_command`, and `trajectory` are visible
- `control_command` and motor-speed topics are correctly separated as downstream

Guarded test without prepare:

```bash
python3 -m scripts.probe_avoidbench_action \
  --namespace /hummingbird \
  --send-test-action \
  --action-topic /hummingbird/autopilot/velocity_command \
  --axis x \
  --magnitude 0.05 \
  --duration 0.2
```

Observed:

- `autopilot_state_before: OFF`
- `autopilot_state_after_prepare: OFF`
- `status: ACTION_RESPONSE_NOT_DETECTED`

Guarded test with explicit prepare and hover wait:

```bash
python3 -m scripts.probe_avoidbench_action \
  --namespace /hummingbird \
  --send-test-action \
  --publish-arm-before-test \
  --publish-start-before-test \
  --action-topic /hummingbird/autopilot/velocity_command \
  --axis x \
  --magnitude 0.20 \
  --duration 1.0 \
  --response-window 3.0 \
  --strict
```

Observed:

- `autopilot_state_before: HOVER`
- `autopilot_state_after_wait: HOVER`
- `delta_position_axis: +0.09279`
- `max_velocity_delta_axis: +0.11977`
- `status: ACTION_RESPONSE_DETECTED`

Interpretation:

- direct velocity commands are valid, but only after the autopilot is in a
  controllable state
- a reliable minimal RL reset must include the mode-transition sequence, not
  just the action topic
