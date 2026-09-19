# Native Actor Map-Isolated Probe v1

Date: 2026-09-07

## Purpose

Check whether `latent_only_affordance_dim16_legacy_reward_rerun/model_best`
can serve as a nonprivileged reference-sequence teacher under the
map-isolated benchmark contract.

No A* planner, PID route teacher, or sealed test split is used.

## Checkpoint

- Checkpoint: `checkpoints/latent_only_affordance_dim16_legacy_reward_rerun/model_best`
- Training log: `runs/latent_only_affordance_dim16_legacy_reward_rerun`
- Logged best step: 440000
- Logged best eval success: 1.0
- Logged best final goal distance: 0.17448142671759884

## Formal Benchmark Probe

Command:

```bash
/home/tequila/miniconda3/envs/drones/bin/python scripts/evaluate_native_actor_map_isolated.py --maps-per-family 2 --output analysis_outputs/native_actor_map_isolated_legacy_best_v1 --dataset-output data/reference_planner/native_actor_map_isolated_legacy_best_v1
```

Result:

- Splits read: train, validation
- Test maps read: false
- Maps per family: 2
- Episodes: 36
- Successes: 0
- Reference samples: 0
- Dominant termination: attitude_bound

The formal nonprivileged dataset was not written because there were no
successful native Actor trajectories.

## Diagnostic Pose Probes

These probes are not formal map-isolated evidence; they isolate the likely
failure cause.

Zero-yaw only:

- Episodes: 18
- Successes: 0
- Reference samples: 0

Canonical training-axis start/goal:

- Episodes: 18
- Successes: 3
- Reference samples: 78
- Clear maps: 2/2 success
- Symmetric maps: 1/2 success
- All other sampled obstacle families: 0 success

## Interpretation

The rerun checkpoint is a valid successful native Actor for the legacy stage2
training/evaluation contract.  It is not a valid teacher for the current
map-isolated long-horizon forest contract.

The key bottleneck is not merely obstacle density.  Clear maps fail under the
formal random start/goal/yaw benchmark, while canonical-axis clear maps succeed.
This indicates strong dependence on the original training coordinate regime and
insufficient pose/map generalization.

## Decision

Do not train Planner-v1 from this Actor on the formal map-isolated benchmark
yet.  The immediate next step should be to make the high-level interface
coordinate-consistent and then retrain/evaluate the native or reference-planner
teacher under map-isolated train/validation maps.
