# Terminal action audit and occupancy-v1 branch

## Evidence from the frozen K1 checkpoint

The audit re-evaluated the selected clean rayfix K1 checkpoint on fixed seeds
`0..19`. Ten T1 episodes contributed 1,810 Upper decisions.

- The action mapping is numerically correct. Normalized planar magnitudes
  `0.1`, `0.3`, and `0.5` produce command speeds `0.08`, `0.24`, and
  `0.40 m/s`. Mapping MAE is approximately `1.4e-8`.
- The actor itself is saturated. Mean raw planar norm is `1.349`, median is
  `1.413`, and `99.83%` of T1 decisions have norm at least `0.95`.
- The old square actor output and circular executable velocity introduce a
  many-to-one action alias. Actions outside the XY unit disk are radially
  normalized by the environment, but the critic sees the unprojected action.
- On 152 safe T1 states with `0.3 < d_goal < 1.2 m`, the critic selected the
  raw saturated actor action 136 times and a `0.8 m/s` goal-directed action 15
  times. It selected `0.6 m/s` once and never selected stop or lower speeds.
- The raw actor action and its unit-disk projection execute the same physical
  command. Nevertheless, `min(Q1,Q2)` rates the raw action higher in `99.34%`
  of scanned states, by `0.789` on average. This is direct evidence of critic
  inconsistency caused by the aliased action representation.

The diagnosis therefore points to the action representation and critic/actor
optimisation interface before reward tuning. The V4.3 reward remains frozen.

Audit artifacts:

- `analysis_outputs/upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1_action_interface_audit/`
- `analysis_outputs/upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1_terminal_xy_q_scan/`

## Terminal-v1 controlled fix

Preset: `upper_dslpid_recurrent_multitree_terminal_v1_h8_k1`

The only experimental change is the executable planar action representation:

1. Actor XY logits are smoothly squashed into the unit disk.
2. Exploration actions are projected into the same disk before execution and
   replay insertion.
3. TD3 target-policy actions are projected after target smoothing noise.
4. Actor loss, critic training, replay, and the environment now use the same
   action coordinates.

No reward, map, seed, history, curriculum, dynamics, or success condition is
changed.

```bash
cd /home/tequila/UAV-Latent-RL

/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_terminal_v1_h8_k1 \
  --total_steps 120000
```

The primary acceptance criterion is a substantial reduction in T1 timeout,
with actor planar magnitudes no longer concentrated at the disk boundary.

## Independent occupancy-v1 branch

Preset: `upper_dslpid_recurrent_multitree_occupancy_v1_h8_k1`

This branch derives from the clean rayfix K1 baseline, not terminal-v1. It adds
one single-scale persistent world-frame local map:

- tri-state voxel values: Free `-1`, Unknown `0`, Occupied `+1`;
- ray interiors update Free and finite endpoints update Occupied;
- old observations persist while they remain in the UAV-centred sliding window;
- resolution `0.4 m`, shape `12 x 12 x 6`, physical coverage approximately
  `4.8 x 4.8 x 2.4 m`;
- 864 occupancy values are appended once to the current state, not duplicated
  into each H8 history frame;
- total observation dimension: 1,781;
- no Multi-scale representation and no depth camera.

Estimated replay memory is approximately 816 MiB for the 60k main buffer plus
272 MiB for the 20k success buffer.

```bash
/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_occupancy_v1_h8_k1 \
  --total_steps 120000
```

Do not combine terminal-v1 and occupancy-v1 until each independent ablation has
completed the same fixed-seed formal evaluation.
