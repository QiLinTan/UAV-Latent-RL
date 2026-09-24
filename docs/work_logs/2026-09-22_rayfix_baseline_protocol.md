# Rayfix baseline protocol (stage 0)

This protocol freezes the post-rayfix stage-0 task before any occupancy-map or
reward change. Both baselines use 96 rays, H8 recurrent TD3, 917-D observation,
3-D world-frame velocity action at 15 Hz, the V4.3 `rate_integrated_v4` reward,
and the identical procedural-forest train/validation split with fixed evaluation
seeds `0..19`.

Only action-to-state execution differs:

| Preset | Execution |
| --- | --- |
| `upper_dslpid_recurrent_multitree_rayfix_v1_h8_full` | ReferencePacket → 120 Hz DSLPID → full UAV |
| `upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1` | acceleration-limited velocity kinematics |

Do not alter reward weights, history length, action space, curriculum, map IDs,
or ray geometry within this comparison. Training outputs use new directories;
pre-rayfix and mixed-log `v4_3_h8_k1` outputs are retained only as historical
baselines.

Formal evaluation must use at least 20 episodes. It records standard navigation
metrics, two-second collision windows, and timeout classes T1–T4.

## Training

Run each baseline from scratch. The new `runs/`, `checkpoints/`, trajectory,
collision, and timeout paths are owned only by that run.

```bash
cd /home/tequila/UAV-Latent-RL

/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_rayfix_v1_h8_full \
  --total_steps 120000

/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1 \
  --total_steps 120000
```

The periodic in-training evaluation uses 10 fixed seeds. It is used for model
selection only; do not treat it as the final result.

## Independent 20-episode evaluation

Run the following once per selected `model_final` after training. Each command
uses evaluation seeds `0..19` and separate, empty output directories. Replace
`<preset>` with `upper_dslpid_recurrent_multitree_rayfix_v1_h8_full` or
`upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1`.

```bash
/home/tequila/miniconda3/envs/drones/bin/python \
  -m scripts.run_change_td3 <preset> -- \
  --evaluate_only true \
  --load_checkpoint checkpoints/<preset>/model_final \
  --eval_interval 1 \
  --eval_episodes 20 \
  --eval_quiet false \
  --log_dir runs/<preset>_formal_eval \
  --trajectory_output_dir analysis_outputs/<preset>_formal_eval_trajectory \
  --collision_diagnostic_output_dir analysis_outputs/<preset>_formal_eval_collision_windows \
  --timeout_diagnostic_output_dir analysis_outputs/<preset>_formal_eval_timeout_diagnostics
```

Read the outcome from `runs/<preset>_formal_eval/evaluation_metrics.jsonl`,
the seed-0 XY/XYZ trajectory directory, collision windows, and
`timeout_diagnostics_latest.json`. Do not reuse any of these directories for a
second run; give a repeat a new suffix such as `_repeat2`.
