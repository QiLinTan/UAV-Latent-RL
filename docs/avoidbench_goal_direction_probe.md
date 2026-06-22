# AvoidBench Goal-Direction Probe

This probe verifies whether hand-coded velocity commands can reduce
`distance_to_goal` before further TD3 training.

Run with the AvoidBench ROS/Gazebo stack already started:

```bash
python3 -m scripts.probe_avoidbench_goal_direction_policy \
  --namespace /hummingbird \
  --episodes 1 \
  --steps 40 \
  --speed 0.08 \
  --action-preset conservative \
  --frame auto
```

The probe compares:

- zero action;
- constant forward action;
- goal-direction action under a world-frame assumption;
- goal-direction action under a body-frame assumption.

Outputs:

```text
runs/avoidbench_goal_direction_probe/<timestamp>/summary.json
runs/avoidbench_goal_direction_probe/<timestamp>/episodes.csv
```

Interpretation:

- if goal-direction improves distance more than zero action, action mapping is
  likely usable for Stage 1;
- if neither world nor body goal-direction improves distance, stop training
  and inspect action frame, goal delta, and distance calculation;
- if collision or native mission resets appear, fix environment ownership
  before training.

## Latest Result

Run:

```text
runs/avoidbench_goal_direction_probe/20260622-133135/
```

Command:

```bash
python3 -m scripts.probe_avoidbench_goal_direction_policy \
  --namespace /hummingbird \
  --episodes 1 \
  --steps 25 \
  --speed 0.08 \
  --action-preset conservative \
  --frame auto
```

Result:

| Strategy | Frame | Distance Delta | Done Reason | Collision |
| --- | --- | ---: | --- | --- |
| zero | world | 0.000000 m | collision | true |
| constant_forward | world | 0.000092 m | collision | true |
| goal_direction | world | 0.001510 m | collision | true |
| goal_direction | body | 0.001509 m | collision | true |

The probe did not pass the Stage 1 gate. Goal-direction moved slightly in the
expected +x direction, but every episode terminated on `collision`, and
`/hummingbird/collision` remained true after the run.

Decision:

- do not start Stage 1 training yet;
- do not run longer Plain TD3 navigation;
- do not start latent or depth work;
- first fix the native collision/mission ownership issue so reset starts from
  a reliable non-collision state.

Likely causes to inspect next:

- Unity collision state is not cleared by the current RL reset sequence;
- native `avoid_manage_node` mission lifecycle still owns Unity/Gazebo state;
- the reset pose may be interpreted as colliding by Unity even while Gazebo
  odometry and height are stable;
- `AvoidBenchRLEnv.reset()` clears only its Python collision cache, but the
  next ROS collision message can immediately restore `True`.
