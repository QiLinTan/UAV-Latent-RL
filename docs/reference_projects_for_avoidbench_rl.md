# Reference Projects For AvoidBench RL

## Scope

This workspace does not contain local clones of `tudelft/mavrl` or
`rickstaa/ros-gazebo-gym`. The summary below is based on upstream source and
upstream generated API documentation, not on locally checked-out repositories.

Key upstream sources used:

- `tudelft/mavrl`
  - `train_policy.py`
  - `mav_baselines/torch/envs/vec_multi_env_wrapper.py`
  - `configs/control/config_new.yaml`
  - `configs/control/config_new_out.yaml`
  - `configs/control/config.yaml`
  - `avoider_vel_cmd.py`
- `rickstaa/ros-gazebo-gym`
  - `ros_gazebo_gym.core.ros_launcher`
  - `ros_gazebo_gym.core.gazebo_connection`
  - `ros_gazebo_gym.core.controllers_connection`
  - `ros_gazebo_gym.robot_gazebo_env`
  - `ros_gazebo_gym.robot_envs.panda_env`
  - `ros_gazebo_gym.task_envs.panda.panda_reach`

## MAVRL

### 1. RL environment class or environment interface

The main training entry point is not a pure Python Gym env class inside `mavrl`.
Instead:

1. `train_policy.py` imports `AvoidVisionEnv_v1` from `flightgym`.
2. It instantiates `AvoidVisionEnv_v1(...)`.
3. It then wraps that object with `mav_baselines/torch/envs/vec_multi_env_wrapper.py`
   via `wrapper.VisionEnvVec(...)`.

The practical conclusion is:

- the actual simulator environment is owned by AvoidBench `flightgym`;
- `mavrl` adds the SB3-facing vectorized wrapper and the policy/training code;
- `mavrl` itself does not contain the underlying physics/reward/reset logic.

This matters for the current project because your local AvoidBench version does
not expose `flightgym.AvoidVisionEnv_v1`; it exposes `avoidbridge` instead. So
`mavrl` cannot be reused as-is.

### 2. How `reset()` and `step()` are implemented

The visible Python behavior is in `VisionEnvVec`.

`reset(random=True)`:

1. Calls the underlying native env reset:
   `self.wrapper.reset(self._state_observation, random)`.
2. Calls `render(0)` twice to refresh Unity-side images.
3. Pulls a depth image with `getDepthImage()`.
4. Preprocesses depth to `uint8` using a `0..12m -> 0..255` mapping.
5. Fills sequence memory buffers for `image` and `state`.
6. Returns a dict observation.

`step(action)`:

1. Reshapes the action to `[num_envs, act_dim]`.
2. Delegates the actual simulator step to:
   `self.wrapper.step(action, state, reward_components, done, extra_info)`.
3. Calls `render(0)` again.
4. Pulls a fresh depth image with `getDepthImage()`.
5. Shifts sequence buffers and appends new depth/state frames.
6. Returns:
   `(obs_dict, reward, done, info_list)`.

Important boundary:

- reward calculation, episode termination, collision handling, and simulator
  state transition happen in the native AvoidBench `flightgym` layer;
- the Python wrapper mostly converts native outputs into SB3/Gym-shaped data.

### 3. Action space

`VisionEnvVec` exposes:

- `spaces.Box(low=-1, high=1, shape=(act_dim,))`

The config files show the concrete simulator action settings:

- `simulation.action_mode: 1`
- comment says `0 velocity, 1 acceleration`
- `act_max: [4.0, 4.0, 1.0, 0.6]`
- `act_min: [-4.0, -4.0, -1.0, -0.6]`

So the policy outputs normalized `[-1, 1]` actions, and the env interprets
them as a 4D high-level command. In the ROS deployment bridge
`avoider_vel_cmd.py`, that 4D command is explicitly treated as:

- body-frame `ax, ay, az`
- yaw rate

and then integrated into a world-frame velocity command before publishing to a
ROS topic.

Practical migration takeaway:

- keep the RL action normalized;
- scale it inside the adapter;
- stay at the high-level velocity/reference layer, not motor RPM.

### 4. Observation space and depth usage

`VisionEnvVec` exposes a dict observation:

```python
{
    "image": Box(shape=(n_seq, 256, 256), dtype=uint8),
    "state": Box(shape=(n_seq, goal_obs_dim), dtype=float64),
}
```

The wrapper feeds the policy with depth, not RGB:

- `step()` and `reset()` call `getDepthImage()`;
- depth is clipped to `12m`, normalized, and stored as `uint8`;
- the wrapper does not use RGB in the policy input path.

The config used by the ROS deployment bridge confirms the intended state size:

- `goal_obs_dim: 7`
- `use_depth: true`
- `seq_len: 1`

`avoider_vel_cmd.py` reconstructs a 7D state feature vector as:

1. log distance to goal
2. horizontal body-frame speed
3. goal bearing
4. horizontal velocity direction
5. vertical position error
6. vertical velocity
7. yaw

So the learned policy structure is:

- perception branch: depth
- low-dimensional branch: compact goal-and-motion state

### 5. Reward and done

Visible reward structure in `mavrl`:

- `step()` returns the last element of `reward_components`
  as the scalar reward;
- the remaining named components are tracked in episode info;
- reward names come from the native env with `getRewardNames()`.

The config files show the intended reward decomposition:

- collision penalty coefficient
- distance penalty coefficient
- speed penalty coefficient
- vertical penalty coefficient
- angular penalty coefficient
- input penalty coefficient
- yaw penalty coefficient

Visible termination hints:

- `done` comes from native `_single_done`;
- `max_t: 5.0` is configured;
- `reset_if_collide: true` is configured.

So the Python layer does not define `done`; it trusts the native simulator env.

### 6. How it connects AvoidBench / ROS / Unity

Training path:

- direct AvoidBench native env via `flightgym.AvoidVisionEnv_v1`
- Unity lifecycle is explicit:
  - `connectUnity()`
  - `spawnObstacles(...)`
  - wait for `ifSceneChanged()`
  - `getPointClouds(...)`
  - `readPointClouds(...)`
- evaluation env shares the same Unity pointer via
  `setUnityFromPtr(train_env.wrapper.getUnityPtr())`

This path is not ROS-first. It is direct native env plus Unity rendering.

Deployment path:

- `avoider_vel_cmd.py` is the ROS bridge for running a trained policy
- subscribes to:
  - `/depth`
  - odometry
  - `/hummingbird/goal_point`
- publishes:
  - velocity command
  - controller activation flag
  - iteration timing

So in `mavrl`, ROS is mainly a deployment/runtime bridge, not the PPO training
interface.

### 7. Designs worth migrating into current `UAV-AvoidBench-RL`

What is worth copying:

- keep the trainer independent from simulator specifics;
- make the env adapter return a Gym-friendly dict observation;
- keep a compact low-dimensional state branch separate from the image branch;
- normalize actions to `[-1, 1]` and scale them only inside the adapter;
- treat Unity scene management as explicit lifecycle work, not as part of
  policy code;
- keep short sequence buffers near the env wrapper, not hidden in the trainer.

What should not be copied directly:

- the hard dependency on `flightgym.AvoidVisionEnv_v1`;
- the assumption that Unity/depth access lives inside one native env object;
- the assumption that training can bypass ROS entirely.

Current repo mapping:

- [envs/avoidbench/backend.py](/home/tequial/projects/UAV-AvoidBench-RL/envs/avoidbench/backend.py)
  is the right place for binding loading and ROS/service/topic connection helpers;
- [envs/avoidbench/adapter.py](/home/tequial/projects/UAV-AvoidBench-RL/envs/avoidbench/adapter.py)
  is the right place for a Gym-facing adapter layer;
- [envs/avoidbench/observation.py](/home/tequial/projects/UAV-AvoidBench-RL/envs/avoidbench/observation.py)
  already fits the idea of a compact low-dimensional observation branch.

## ros-gazebo-gym

### 1. Environment encapsulation structure

`ros-gazebo-gym` is much more explicit about layering than `mavrl`.

The stack is:

1. `core/ros_launcher`
   - starts `roscore`
   - launches ROS launch files
2. `core/gazebo_connection`
   - wraps Gazebo pause/unpause/reset/set-state services
3. `core/controllers_connection`
   - wraps controller-manager list/switch/reset behavior
4. `robot_gazebo_env`
   - generic Gym base class for ROS/Gazebo envs
5. `robot_envs/...`
   - robot-specific IO and readiness checks
6. `task_envs/...`
   - task-specific action mapping, observations, reward, and done

This is the cleanest reference model for your next AvoidBench adapter step.

### 2. How ROS/Gazebo is started

`ROSLauncher.initialize()`:

1. checks whether a ROS master is online;
2. starts `roscore` in a subprocess if none exists;
3. initializes a ROS node if ROS time is not initialized.

`ROSLauncher.launch(...)`:

1. resolves the catkin workspace;
2. optionally installs the requested ROS package if missing;
3. builds a command like:
   `. ./devel/setup.bash; roslaunch <pkg> <launch> key:=value ...`
4. runs it in a managed subprocess;
5. stores the process handle for later cleanup.

This is much more robust than assuming the user manually started all required
launch files in the correct order.

### 3. How it waits for topic/service readiness

`GazeboConnection` waits on core Gazebo services during initialization:

- `/gazebo/pause_physics`
- `/gazebo/unpause_physics`
- `/gazebo/reset_simulation`
- `/gazebo/reset_world`
- `/gazebo/get_model_state`
- `/gazebo/set_model_state`
- `/gazebo/get_link_state`
- `/gazebo/set_model_configuration`
- `/gazebo/get_physics_properties`
- `/gazebo/set_physics_properties`

`ControllersConnection` waits on controller-manager services:

- `list_controllers`
- `switch_controller`

It also checks whether Gazebo is paused by looking for `/clock` and using
`rospy.wait_for_message("/clock", Clock, timeout=...)`.

At the env level, `RobotGazeboEnv._reset_sim()` calls
`_check_all_systems_ready()` multiple times around reset boundaries, so
readiness is not a one-time startup check.

This pattern is directly relevant to AvoidBench.

### 4. How it resets simulation

`RobotGazeboEnv.reset()` is centralized:

1. `_reset_sim()`
2. `_init_env_variables()`
3. `_update_episode()`
4. `_get_obs()`
5. `_get_info()`

`_reset_sim()` does more than one service call:

1. unpause Gazebo;
2. reset controllers;
3. wait until systems are ready;
4. capture the robot joint state before reset;
5. pause Gazebo;
6. call `gazebo.reset_sim()`;
7. unpause Gazebo;
8. reapply robot joint configuration with `set_model_configuration(...)`;
9. optionally set the robot init pose;
10. reset controllers again;
11. recheck readiness.

`GazeboConnection.reset_sim()` itself supports three reset modes:

- `SIMULATION`
- `WORLD`
- `NO_RESET_SIM`

This is the strongest reference for your future AvoidBench `reset()`: it treats
reset as a sequence, not a single command.

### 5. How it wraps ROS topics/services into Gym `step()`

`RobotGazeboEnv.step(action)` does:

1. `gazebo.unpause_sim()`
2. `_set_action(action)`
3. `_get_obs()`
4. optional `gazebo.pause_sim()`
5. `_is_done(obs)`
6. `_compute_reward(obs, done)`
7. `_get_info()`
8. return `(obs, reward, done, False, info)`

The key design point is that the base class owns the RL loop skeleton, while
robot/task subclasses fill in the robot-specific details:

- `_set_action`
- `_get_obs`
- `_is_done`
- `_compute_reward`
- `_check_all_systems_ready`

This separation is exactly what `UAV-AvoidBench-RL` still needs.

## What to copy into the current project

### Recommended architecture

For the current repo, the best hybrid is:

1. take `ros-gazebo-gym`'s layered separation for launch/reset/readiness;
2. take `mavrl`'s observation design idea:
   depth branch plus compact state branch;
3. keep the current `avoidbridge` integration only as the Unity/render bridge,
   not as the full RL env.

### Proposed mapping onto current files

- [envs/avoidbench/backend.py](/home/tequial/projects/UAV-AvoidBench-RL/envs/avoidbench/backend.py)
  should grow toward a `GazeboConnection + ROS endpoint` utility layer;
- [envs/avoidbench/adapter.py](/home/tequial/projects/UAV-AvoidBench-RL/envs/avoidbench/adapter.py)
  should stay the thin binding adapter for Unity images/collision, and a future
  RL env should wrap above it rather than bury reset/reward logic inside it;
- [envs/avoidbench/observation.py](/home/tequial/projects/UAV-AvoidBench-RL/envs/avoidbench/observation.py)
  is the correct place to keep compact depth features if you do not want to
  train directly on full-resolution depth;
- a future `AvoidBenchRLEnv` should follow the `RobotGazeboEnv` pattern:
  - `reset()` owns the full reset ceremony
  - `step()` owns the RL loop skeleton
  - robot-specific ROS calls stay in helper/backend layers
  - reward/done stay in the task env layer

## Bottom line

`mavrl` is useful mainly as a policy-side reference:

- normalized 4D high-level actions;
- depth plus compact state observations;
- thin Python wrapper around a native simulator env.

`ros-gazebo-gym` is useful mainly as a systems-side reference:

- launch orchestration;
- wait-for-topic/service patterns;
- multi-step reset;
- explicit Gym base-class structure for ROS/Gazebo tasks.

For `UAV-AvoidBench-RL`, the closer architectural match is `ros-gazebo-gym`.
For observation and high-level action design, `mavrl` is the better reference.
