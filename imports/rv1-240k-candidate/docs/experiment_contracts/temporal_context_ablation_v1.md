# Temporal Context Ablation v1 — invalidated causal result

Date: 2026-09-09

## Intended question

Does an explicit 0.53 s environmental history reduce the instantaneous
Planner's 24-tree observation ambiguity before adding DAgger?

The frozen Encoder, privileged offline A*/DSLPID teacher, 5x6 reference target,
0.5 s reference horizon, logical asynchronous packet/PID navigation loop,
reward, environment configuration, and 24-tree evaluation seeds 7000--7004
were held fixed.  B1 adds the same eight frames by direct concatenation; B2
encodes the same frames using a one-layer, 32-hidden-unit GRU.

Temporal frame: relative goal (3), existing eight world-fixed range rays (8),
normalized linear velocity (3), yaw sin/cos (2).  It deliberately excludes the
legacy 240-D motor-action history already available to the frozen Encoder.

## Mandatory data-provenance correction

`latent_affordance_margin_v2` contains only `episode_id`, `step_id`, 27-D
planner input, and reference target.  It has no raw observation trajectory,
rollout seed, or tree specification.  An attempted exact replay proved that
its clear portion can be recreated but its single-obstacle rows cannot: a
candidate that has the same sample length had up to 0.975 difference in input
and 0.302 m/s-or-metre target component difference versus its nominal source
episode.  This is caused by map randomness at environment construction not
being archived.  Attaching a replayed history to those labels would have made a
false paired experiment.

Therefore a new provenance-complete paired cohort was collected.  It has the
same 30 successful teacher episodes (10 clear, 10 single, 10 forest24) and the
same BC labels in two views: 27-D B0 and 27-D plus 8x16 B1/B2.  It has 4,231
samples, versus the legacy archive's 4,252.  Each accepted map's seed and tree
specification are stored.  Eighteen deterministic forest candidate maps were
screened to obtain ten successful A*/PID demonstrations; rejected trajectories
are not training samples.  This preserves the successful training-cohort size,
but it is a replacement cohort, not the historical archive.

## Models and offline test split metrics

All three were trained for 100 epochs with the unchanged target-normalized BC
loss and velocity weight 0.5, using episode-separated 22/4/4 train/validation/
test splits.

| Group | Input | Test position RMSE | Test velocity RMSE |
|---|---:|---:|---:|
| B0 instant | 27 | 0.0158 m | 0.0857 m/s |
| B1 history-stack | 27 + 8x16 | 0.0154 m | 0.0843 m/s |
| B2 GRU | 27 + GRU(8x16,32) | 0.0145 m | 0.0821 m/s |

The similar offline values do not establish closed-loop validity.

## Closed-loop outcome

All results below use five seeds per scenario and 0--50 ms logical packet-ready
delay.  The historical B0 checkpoint was separately re-run with the same
evaluator and reproduced the prior sanity condition: clear 5/5 success,
single 5/5 success, forest24 0/5 success and 5/5 collisions, with first-error
categories 4 ambiguity / 1 other.

The newly trained paired B0 did **not** meet that sanity condition: clear and
single were both 0/5 success (mean final distance 0.551 m and 0.488 m).  Hence
the B1/B2 comparison cannot be interpreted as an effect of temporal history.

| Group | forest24 success | forest24 collision | final distance | minimum clearance | contextual OOD |
|---|---:|---:|---:|---:|---:|
| B0 paired instant | 0/5 | 4/5 | 5.859 m | 0.043 m | 23.75% |
| B1 history-stack | 0/5 | 3/5 | 4.272 m | 0.038 m | 31.20% |
| B2 GRU | 0/5 | 3/5 | 5.147 m | 0.090 m | 62.92% |

OOD is calibrated independently in each group's representation space, so it
is a within-run diagnostic rather than an absolute cross-dimensional score.

All five paired-cohort forest failures were labeled `observation_ambiguity` at
step 0 for B0, B1 and B2 under the existing nearest-neighbour diagnostic.
Each temporal decision had `history_padding_count=7`: it contains only eight
copies of the initial frame, not 0.5 s of observed motion.  This confirms the
pre-registered limitation for seed 7000: temporal context cannot resolve a
genuinely initial-state ambiguity.  It does *not* show that GRU learned no
useful later-time behaviour; the prerequisite B0 sanity check failed first.

## Timing and async scope

No runtime-async scheduler, IPC channel, buffer, PID, or reference packet code
was changed.  In the logical navigation evaluation, mean/p95 planner inference
was B0 0.175/0.262 ms, B1 0.225/0.317 ms, B2 0.516/0.693 ms: all are well below
the 8.33 ms lower-loop compute budget.  This is not a replacement for a new
spawned-worker end-to-end runtime timing trial; the already frozen runtime
async Phase-1 result remains the concurrency evidence.

## Decision

Do **not** start DAgger and do **not** claim that Temporal Context solved or
failed to solve the original 24-tree ambiguity.  First repair the experimental
baseline while preserving the paired cohort: establish a B0 checkpoint that
reproduces clear and single success under the new cohort's fixed split and
evaluation protocol, then train B1/B2 from that exact contract and rerun this
ablation.  Once that passes, inspect seeds after at least one non-padded
high-level decision separately from t=0.  Only if later ambiguity decreases
while late closed-loop OOD remains should DAgger be introduced.
