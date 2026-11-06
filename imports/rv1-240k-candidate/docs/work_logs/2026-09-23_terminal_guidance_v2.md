# Terminal guidance v2 K1 ablation

## Evidence motivating this branch

The independent 20-seed evaluation of terminal-v1 produced 0% success, 5%
collision and 95% timeout.  The action interface was linear, but every sampled
state between 0.3 m and 1.2 m from the goal still commanded approximately
0.8 m/s.  At 0.3--0.5 m, 82.8% of commands pointed away from the goal.  The
critic Q scan was nearly flat (mean within-state range 0.28) and selected a
0.8 m/s candidate in 133/177 states.

This is sufficient evidence to unfreeze the terminal reward while keeping the
task and all other experimental variables fixed.

## Controlled change

Preset:

`upper_dslpid_recurrent_multitree_terminal_v2_h8_k1`

Unchanged from terminal-v1:

- ray-fixed 96-ray observation and h8 recurrent TD3;
- 917-dimensional observation and 3-D world-velocity action;
- smooth planar action projection;
- 15 Hz Upper decision rate and K1 acceleration-limited transition;
- train/eval maps, seeds, curriculum and forest geometry;
- 0.2 m success radius and all terminal outcome rewards;
- progress, proximity, time and route-overshoot semantics.

Changed:

- reward contract is named `terminal_guidance_v5`;
- the old scalar near-goal speed/away penalties are disabled in this preset;
- inside 1.2 m, actual 3-D velocity is trained to follow a vector pointing at
  the goal;
- desired speed decreases linearly from 0.8 m/s at 1.2 m to 0.1 m/s at the
  unchanged 0.2 m success boundary;
- the guidance activation ramps in over the outer 0.2 m;
- the velocity tracking rate has weight 160 and is integrated using the K1
  decision interval (1/15 s).

The capture speed stays non-zero so the reward does not teach the policy to
stop immediately outside the success sphere.  The full 3-D target direction
also couples terminal XY capture and target-height convergence without a
separate hard-coded height controller.

Near-goal reset randomization is deliberately not included in this branch.
Terminal-v1 already produced 16/19 T1 or T3 timeouts, so near-goal states were
well represented.  Changing both reward and reset distribution would make the
result uninterpretable.  A separate coverage branch is only justified if this
reward produces the correct local Q ordering but still lacks successful
terminal transitions.

## Verification

- Directed terminal velocity ranks above stop, tangential, full-speed and
  away actions.
- After normal progress is included, the scheduled target speed ranks above
  0.8 m/s at distances 0.3, 0.5, 0.8 and 1.0 m.
- The tracking cost is frequency invariant between 15 Hz and 120 Hz.
- Terminal outcome rewards remain success=+250, timeout=-180,
  bound=-250 and collision=-300.
- Final full suite: 164 passed plus 4 parameterized distance checks.
- A 2-step K1 smoke run constructed the 917-D h8 environment, evaluated and
  saved a selected checkpoint successfully.

## Training command

```bash
cd /home/tequila/UAV-Latent-RL

/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 \
  upper_dslpid_recurrent_multitree_terminal_v2_h8_k1 \
  --total_steps 120000
```

The preset writes to:

- `runs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1/`
- `checkpoints/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1/`
- `analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_trajectory/`
- `analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_collision_windows/`
- `analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_timeout_diagnostics/`

## Independent 20-seed evaluation

Run this only after training completes:

```bash
/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 \
  upper_dslpid_recurrent_multitree_terminal_v2_h8_k1 \
  -- \
  --evaluate_only true \
  --load_checkpoint checkpoints/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1/model_final \
  --eval_interval 1 \
  --eval_episodes 20 \
  --eval_gui false \
  --eval_stepsleep false \
  --eval_quiet true \
  --log_dir runs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_formal_eval \
  --trajectory_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_formal_eval_trajectory \
  --collision_diagnostic_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_formal_eval_collision_windows \
  --timeout_diagnostic_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_formal_eval_timeout_diagnostics \
  --eval_action_interface_audit_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_formal_eval_action_interface_audit \
  --eval_terminal_xy_q_scan_output_dir analysis_outputs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1_formal_eval_terminal_xy_q_scan
```

Do not combine this branch with occupancy unless the independent evaluation
shows that T1/T3 and near-goal saturation improve without increasing collision
above the clean rayfix K1 baseline.

## Completed result

Training completed at 120k steps. `model_final` correctly restored the 44k
peak, but none of the 61 periodic 10-episode evaluations recorded a success.

Independent seeds 0--19:

- success: 0/20;
- collision: 0/20;
- timeout: 17/20;
- height bound: 3/20;
- mean final goal distance: 1.884 m;
- mean absolute height error: 0.975 m;
- mean final height: 2.460 m;
- timeout classes: T3=15 and T2=2.

The selected policy never entered the 1.2 m 3-D guidance region, so its total
terminal velocity tracking penalty was zero.  In seed 0 it commanded upward
motion from the beginning: mean z action was +0.47 above 6 m goal distance,
+0.48 from 4--6 m and +0.78 from 3--4 m.  It only commanded descent after the
vehicle was already near the height ceiling.  Because z error alone grew to
about 1.45 m, the 3-D distance gate prevented the intended terminal guidance
from activating even after XY became relatively close.

A no-training `Actor XY + P height` ablation also produced 0/20 success, 5%
collision and 95% timeout, with mean final distance 2.496 m and all 19 timeouts
classified T2.  Therefore fixing height only at deployment does not recover
this policy; its XY behavior was learned jointly with the climb and is also no
longer a valid terminal policy.

Decision: terminal-v2 is rejected.  Do not train it longer, combine it with
occupancy, or migrate it to Full.  The clean rayfix K1 remains the best current
baseline.  The next branch must avoid creating a negative reward moat around
the goal and must provide successful terminal-transition coverage before
another full 120k run.
