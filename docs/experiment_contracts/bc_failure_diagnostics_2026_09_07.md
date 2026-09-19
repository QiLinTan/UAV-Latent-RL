# BC Failure Diagnostics

Date: 2026-09-07

Scope: diagnostic only.  No model, training, reward, environment, or sensor
interface was changed.

Script:

```bash
/home/tequila/miniconda3/envs/drones/bin/python analysis/analyze_bc_closed_loop_failures.py --output analysis_outputs/bc_failure_diagnostics/latent_affordance_margin_v2_forest24.json
```

Primary output:

- `analysis_outputs/bc_failure_diagnostics/latent_affordance_margin_v2_forest24.json`

## 1. Current BC Dataset

Dataset:

- `data/reference_planner/latent_affordance_margin_v2`
- Samples: 4252
- Successful teacher episodes: 30
- Split episodes: train 22, validation 4, test 4
- Input: 27D planner input
- Target: 5 x 6 future position-offset and velocity sequence
- Teacher: `oracle_astar_reference_plus_dslpid`
- Successful episodes only: true

Scenario samples:

| Scenario | Successful episodes | Samples |
|---|---:|---:|
| clear | 10 | 1340 |
| single | 10 | 1356 |
| forest24 | 10 | 1556 |

Important dataset property:

The teacher is privileged.  These data are useful for diagnosing BC and
observability, but they do not prove deployable nonprivileged navigation.

## 2. 24-Tree Closed-Loop Failure Statistics

Student:

- Frozen latent checkpoint: `checkpoints/latent_only_affordance_dim16/model_final`
- Planner: `checkpoints/reference_planner/latent_affordance_margin_v2/planner_final.pt`
- Controller: logical async ReferencePacket + DSLPID
- Scenario: 24 trees + route-blocking tree
- Seeds: 7000-7004

Result:

| Metric | Value |
|---|---:|
| Episodes | 5 |
| Success rate | 0.0 |
| Collision rate | 1.0 |
| Mean final distance | 4.9601 m |
| Mean minimum clearance | 0.000668 m |
| Worst minimum clearance | 0.000447 m |
| Mean path length | 2.7742 m |

All five failed by collision.

## 3. Student Closed-Loop vs BC Dataset Nearest Neighbor

Planner inputs were standardized with BC train-set statistics.  Student
closed-loop planner inputs were compared against BC train inputs.

| Metric | Value |
|---|---:|
| BC train-train nearest median | 0.0058 |
| BC train-train nearest p95 | 4.7902 |
| Student-to-train nearest median | 2.5860 |
| Student-to-train nearest p95 | 8.0187 |
| Student-to-train nearest max | 9.9041 |
| Student OOD fraction, using train p95 | 0.2235 |

Interpretation:

Distribution shift is real: about 22.3% of student closed-loop high-level states
fall outside the BC train-set p95 nearest-neighbor radius.  However, many early
failures occur before the state is formally OOD.

Student prediction vs nearest teacher reference:

| Metric | Value |
|---|---:|
| First reference position delta median | 0.0401 m |
| First reference position delta p95 | 0.0729 m |
| First heading angle delta median | 21.61 deg |
| First heading angle delta p95 | 53.06 deg |

The planner usually resembles the nearest teacher in the immediate first point,
so the failure is not only a one-step imitation error.

## 4. Similar Input, Different Teacher Reference

Within the BC dataset, similar 27D planner inputs sometimes map to materially
different teacher references.

Significant conflict definition:

- full position RMSE > 0.10 m, or
- final 0.5s offset delta > 0.25 m, or
- first-heading angle > 60 deg

Conflict rates:

| Standardized radius | Pairs | Final delta >0.25m | Full pos RMSE >0.10m | Velocity RMSE >0.25m/s | Heading >60deg |
|---:|---:|---:|---:|---:|---:|
| 5.0 | 20187 | 4.60% | 3.96% | 7.53% | 0.66% |
| 7.5 | 21179 | 5.46% | 4.89% | 8.82% | 0.95% |
| 10.0 | 21260 | 5.59% | 5.01% | 8.96% | 1.03% |

Interpretation:

The conflict rate is not huge, but it is nonzero and appears exactly in a task
where route choice can depend on history or obstacle layout not captured by the
27D instantaneous input.  This supports a temporal-context diagnosis.

## 5. First Obvious Planning Error Per Failed Episode

| Seed | Category | First step | Time | NN distance | Neighbor max heading split | Predicted seq clearance | Actual clearance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7000 | observation ambiguity | 0 | 0.000s | 2.524 | 68.2 deg | 0.757 m | 0.788 m |
| 7001 | observation ambiguity | 192 | 1.600s | 2.394 | 116.3 deg | 0.691 m | 1.048 m |
| 7002 | other | 272 | 2.267s | 2.607 | 0.0 deg | -0.004 m | 0.400 m |
| 7003 | observation ambiguity | 448 | 3.733s | 3.553 | 82.7 deg | 0.275 m | 0.322 m |
| 7004 | observation ambiguity | 264 | 2.200s | 3.810 | 84.1 deg | 0.397 m | 0.333 m |

Episode 7002 is classified as `other` because the nearest teacher references
are not contradictory, but the student's own predicted 0.5s sequence already
passes through a near-collision tube.

## 6. Failure Cause Split

| Cause | Count |
|---|---:|
| distribution_shift | 0 |
| observation_ambiguity | 4 |
| horizon_limitation | 0 |
| other | 1 |

Important nuance:

Although first-error classification is mostly observation ambiguity,
distribution shift still exists globally: 22.3% of student planner states are
outside the train p95 nearest-neighbor radius.  It is not the first visible
error in most sampled failures, but it will still hurt closed-loop recovery.

## 7. DAgger vs Temporal Context

Evidence for DAgger:

- Student closed-loop states differ from successful teacher states.
- Student OOD fraction is 22.3%.
- BC was trained only on successful teacher rollouts, so the model has little
  correction data after it drifts.

Evidence for Temporal Context:

- Four of five sampled failures show similar 27D inputs with conflicting
  teacher references before collision.
- Dataset-level conflicts are nonzero even among nearby inputs.
- Immediate first-point imitation is often close, while route choice still
  becomes wrong later.

Judgment:

Use both, but not as one uncontrolled jump.

Recommended order:

1. Add temporal context first in analysis/training interface, because most
   first visible failures are ambiguous under the instantaneous 27D input.
2. Then run DAgger or closed-loop correction data collection using that
   temporally richer input, because DAgger with the same ambiguous 27D input may
   collect more data without resolving the aliasing.

## 8. Boundaries Preserved

This diagnostic did not modify models, train networks, change sensors, change
reward, or change the environment.

No Temporal Planner, DAgger, or sensor redesign was started.
