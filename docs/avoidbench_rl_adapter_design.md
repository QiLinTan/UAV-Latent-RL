# AvoidBench RL Adapter Design

## Current boundary

`avoidbridge` is a Unity scene and sensor bridge. It updates rendering from a
provided quadrotor state, returns RGB/depth/collision data, and manages scene
obstacles. It does not:

- advance vehicle dynamics from an action;
- own the Gazebo simulation clock;
- reset vehicle and controller state atomically;
- compute reward;
- define episode termination;
- expose Gym-style `reset()` and `step(action)`.

Therefore a successful Unity probe is necessary but is not an RL environment.
The missing layer is a ROS/Gazebo/autopilot adapter.

## Proposed `reset()`

An initial `AvoidBenchRLEnv.reset()` should:

1. Stop active motion with `/hummingbird/autopilot/force_hover`.
2. Sample or load a mission, start pose, goal, and obstacle seed.
3. Reset the Gazebo model through `/gazebo/set_model_state`.
4. Publish the matching
   `/hummingbird/autopilot/reset_reference_state`.
5. Spawn or confirm the Unity obstacle scene.
6. Arm and start the autopilot in the validated order.
7. Wait for fresh odometry, RGB/depth, collision, and goal data.
8. Clear episode counters and return the first observation plus reset info.

Reset needs timeouts and freshness checks. It must fail explicitly if Gazebo,
the controller, Unity, or sensor streams are stale.

## Proposed `step(action)`

An initial `AvoidBenchRLEnv.step(action)` should:

1. Validate and clip the normalized action.
2. Convert it to a high-level command.
3. Publish the command at the controller's required rate for one fixed control
   interval.
4. Wait for a newer state/sensor sample, not merely the latest cached sample.
5. Build the observation.
6. Compute reward and termination from the same timestamp window.
7. Return `(observation, reward, terminated, truncated, info)`.

The environment must define whether one step advances wall time or Gazebo
simulation time. Deterministic training requires fixed control duration and
well-defined handling of dropped or delayed ROS messages.

## Action space

Use a high-level action first:

```text
[vx, vy, vz, yaw_rate]
```

published as `geometry_msgs/TwistStamped` to
`/hummingbird/autopilot/velocity_command`, or use a short-horizon local
`TrajectoryPoint` on `/hummingbird/autopilot/reference_state`.

Velocity control is the simpler first integration. A local target may be more
stable when perception runs slower than the inner controller. Direct motor RPM
would couple the policy to attitude stabilization, actuator dynamics, and
high-rate safety constraints, so it is out of scope for the initial adapter.

## Observation

Suggested fields:

- RGB or encoded RGB features;
- depth or encoded depth features;
- position, orientation, linear velocity, and angular velocity;
- goal direction and distance in the body frame;
- previous action;
- collision flag;
- sensor ages or validity flags.

All fields need explicit units, frames, normalization, shapes, dtypes, and
timestamp policy.

## Reward

Start with a small, inspectable reward:

- positive progress: decrease in goal distance;
- terminal success bonus;
- collision penalty;
- small time penalty;
- optional bounded command-smoothness penalty.

Avoid adding many shaping terms before state/action timing is validated. Log
each reward component separately in `info`.

## Termination

`terminated=True`:

- goal reached;
- collision;
- vehicle outside the allowed workspace;
- unrecoverable attitude or altitude limit.

`truncated=True`:

- episode time or step limit;
- ROS/Unity sensor timeout;
- Gazebo or controller health failure.

## `info`

Suggested fields:

```text
is_success
collision
goal_distance
progress
episode_steps
sim_time
state_age
rgb_age
depth_age
action_clipped
reward_progress
reward_collision
reward_time
termination_reason
```

## Required validation before TD3

Do not connect TD3 until these interfaces are present and behaviorally tested:

- `/hummingbird/ground_truth/odometry` updates with valid timestamps;
- `/gazebo/set_model_state` resets pose and zeroes twist;
- `/hummingbird/autopilot/reset_reference_state` synchronizes the controller;
- `/hummingbird/bridge/arm` arms the RotorS interface;
- `/hummingbird/autopilot/start` starts control;
- `/hummingbird/autopilot/force_hover` stops commanded motion safely;
- `/hummingbird/autopilot/velocity_command` or
  `/hummingbird/autopilot/reference_state` moves the vehicle predictably;
- `/hummingbird/goal_point` supplies the active mission goal;
- `/hummingbird/collision`, RGB, and depth correspond to the same vehicle pose;
- reset can run repeatedly without stale observations or accumulating state.

Only after a scripted, non-learning rollout passes repeated reset/step tests
should the adapter be exposed to TD3.
