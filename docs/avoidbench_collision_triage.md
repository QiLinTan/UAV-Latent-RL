# AvoidBench Collision Triage

Date: 2026-06-22

## Decision

Do not start Stage 1 short navigation training.

The current blocker is a task-interface consistency issue around collision
ownership/reset, not a TD3 learning issue. The strongest evidence is that
`/hummingbird/collision` becomes true during reset-only observation with no
action published, and in a follow-up run it is already true immediately after
reset.

Current classification:

| Candidate cause | Status | Evidence |
| --- | --- | --- |
| stale or uncleared collision flag after reset | strongly supported | reset-only checks reported collision before any action or during no-action observation |
| native `avoid_manage_node` mission lifecycle interfering with RL reset | strongly supported | `avoid_manage_node` is still the publisher for reset/start/goal and owns the native mission loop |
| Unity/Gazebo collision-state mismatch | supported | Gazebo odometry is stable near `(0, 0, 1.2)` while the collision topic reports true |
| initial reset pose inside a Unity obstacle | possible | reset-only collision can have this symptom, but the current evidence cannot separate it from stale Unity collision state |
| action frame/sign error | not the primary blocker | horizontal action signs mostly match the commanded axes before collision truncates the tests |
| straight-line goal policy truly hitting an obstacle | not proven | zero action and reset-only observation also collide, so this cannot be blamed on a direct goal policy yet |

## Runtime Conditions

The first triage run found two active Unity processes, so it was treated as
non-authoritative. The stale standalone Unity process was killed and the ROS
launcher was restarted.

Authoritative runtime:

- container: `noetic_ab_workspace`
- launch command: `./scripts/launch_avoidbench_ros_mesa.sh`
- Unity log: `Flightmare Unity is connected`
- active Unity count: one non-defunct `AvoidBench.x86_64` from
  `avoidbench_render`
- ROS readiness: `ROS_INTERFACES_READY`
- discovered endpoints: `83` topics, `47` services, `10/10` expected
  endpoints

Relevant process snapshot after cleanup:

```text
roslaunch avoid_manage rotors_gazebo.launch
avoid_manage_node flight_pilot/state_estimate:=ground_truth/odometry
AvoidBench.x86_64 __name:=avoidbench_render
gzserver simple.world
```

## Probe Command

The triage probe now supports three phases and an `all` mode:

```bash
python3 -m scripts.probe_avoidbench_goal_direction_policy \
  --namespace /hummingbird \
  --mode all \
  --episodes 1 \
  --reset-checks 2 \
  --reset-observe-seconds 2.0 \
  --mapping-steps 30 \
  --steps 25 \
  --speed 0.08 \
  --action-preset conservative \
  --frame auto \
  --output-root runs/avoidbench_collision_triage
```

Outputs:

```text
runs/avoidbench_collision_triage/<timestamp>/summary.json
runs/avoidbench_collision_triage/<timestamp>/reset_sanity.csv
runs/avoidbench_collision_triage/<timestamp>/action_map.csv
runs/avoidbench_collision_triage/<timestamp>/episodes.csv
runs/avoidbench_collision_triage/<timestamp>/step_trace.csv
```

The script records:

- `collision_step`
- `collision_position`
- `collision_before_first_action`
- `collision_action`
- `collision_distance_to_goal`
- `collision_height`
- `collision_autopilot_state`
- per-step trace rows for reset, observe, and action phases

## Clean Single-Unity Result

Authoritative clean-restart artifact:

```text
runs/avoidbench_collision_triage/20260622-143520/
```

Reset-only sanity:

`first collision time` is wall time from the start of that probe episode, so it
includes the reset/takeoff sequence before the no-action observation loop.

| Episode | collision before observe | collision observed | first collision time | position | height | autopilot |
| --- | --- | --- | ---: | --- | ---: | --- |
| 0 | true | true | 0.0 s | near `(0, 0, 1.1993)` | 1.1993 | HOVER |
| 1 | false | true | 3.2839 s | near `(0, 0, 1.1997)` | 1.1997 | HOVER |

This means collision can be true without any RL action. In episode 1, reset
initially returned non-collision, but collision became true during the
2-second no-action observation window after the reset sequence.

Four-direction low-speed action mapping:

| Command | Collision step | Delta x | Delta y | Delta z | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `+x` | 1 | +0.000092 | +0.000000 | +0.000108 | too short, but x sign is positive |
| `-x` | 4 | -0.007032 | -0.000022 | +0.000279 | x sign matches |
| `+y` | 4 | +0.000005 | +0.006737 | +0.000218 | y sign matches |
| `-y` | 4 | +0.000044 | -0.019330 | +0.000153 | y sign matches |

The mapping test is not a pass because every direction ended in collision.
However, the observed horizontal signs do not support an action-frame/sign
reversal as the primary blocker.

Goal-direction policy:

| Strategy | Collision step | Distance delta | Done reason |
| --- | ---: | ---: | --- |
| zero | 3 | -0.000219 m | collision |
| constant forward | 4 | +0.007081 m | collision |
| goal direction, world | 3 | +0.008327 m | collision |
| goal direction, body | 4 | +0.014470 m | collision |

Goal-direction can produce small positive distance progress, but it cannot be
accepted as a Stage 1 gate because all runs terminate by collision. Zero action
also terminates by collision, so the collision source must be fixed before
interpreting hand-policy quality.

## Sticky-Collision Reproduction

Follow-up artifact without restarting the launcher:

```text
runs/avoidbench_collision_triage/20260622-143638/
```

Both reset-only checks reported:

```text
collision_before_observe = true
first_collision_time = 0.0
autopilot_state = HOVER
```

This confirms that once the collision topic becomes true, the current RL reset
sequence does not reliably clear it before the next episode or next probe run.

## Stage 1 Gate

The probe summary reports:

```text
reset_collision_sanity_passed: false
action_mapping_passed: false
goal_direction_policy_passed: false
stage1_short_navigation_allowed: false
```

Training is still blocked for:

- longer Plain TD3 navigation;
- Stage 1 short navigation;
- latent/depth integration.

## Required Next Fix

The next isolation procedure is documented in:

```text
docs/avoidbench_collision_ownership_isolation.md
```

Use:

```text
scripts/probe_avoidbench_collision_ownership.py
```

Do this before training:

1. Isolate RL probes from the native `avoid_manage_node` mission lifecycle, or
   make the RL environment use that lifecycle consistently instead of fighting
   it.
2. Change `AvoidBenchRLEnv.reset()` so it waits for a fresh collision message
   after Gazebo pose reset and hover stabilization, and treats collision=true
   as a reset failure instead of returning a valid episode.
3. Add explicit reset diagnostics for the publisher of `/hummingbird/collision`
   and the timestamp/order of collision messages relative to Gazebo reset,
   Unity mission update, and autopilot hover.
4. Compare official-manager reset against direct bridge-only static collision
   checks so Gazebo/ROS manager collision and avoidbridge collision are not
   conflated.
5. After reset-only collision is consistently false, rerun four-direction
   action mapping for at least 20-50 low-speed steps per direction.
6. Only after action mapping and reset sanity pass, rerun goal-direction in an
   explicitly obstacle-free short-distance setup.

If reset collision is fixed and the hand-coded policy then collides only while
moving through the full AvoidBench scene, create a true Stage 1 scene:

- target distance: 1-2 m;
- target direction: fixed `+x`;
- no obstacle near the start-goal corridor;
- low speed;
- 100-200 step horizon;
- collision logged but not allowed to obscure the first distance-progress
  validation.
