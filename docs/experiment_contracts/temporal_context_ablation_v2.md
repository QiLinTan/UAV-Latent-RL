# Temporal Context Ablation v2 — paired B0/B1/B2 result

Date: 2026-09-09

This report supersedes `temporal_context_ablation_v1.md`, whose replacement
cohort B0 did not first meet the clear/single closed-loop sanity condition.

## Corrections before comparison

1. The legacy 27-D archive does not contain histories, map specs, or complete
   rollout provenance.  A paired replacement cohort was therefore collected:
   30 successful oracle A*/DSLPID episodes, 4,231 samples, 10 clear / 10
   single / 10 forest24, with identical B0 rows and B1/B2 history rows.
2. The paired cohort reuses the historical `latent_affordance_margin_v2`
   episode partition (22 train / 4 validation / 4 test) and recomputes target
   normalization from that train split.  The earlier replacement cohort had
   silently used a newly randomized split.
3. Evaluation constructs each environment with the reported episode seed,
   rather than only calling `reset(seed=...)` after map creation.  Thus
   single/forest tree maps are now deterministic for seeds 6000--6004 and
   7000--7004.
4. A B0 training-seed sweep found seed 0 and seed 7 pass clear/single 5/5,
   whereas seed 1 does not.  Seed 0 was frozen as the common training seed for
   B0, B1 and B2 before temporal models were trained.  This is an engineering
   sanity-gate selection, not a blind generalization estimate; future work
   should report multi-seed intervals.

All groups retain frozen Encoder, oracle teacher, BC target/loss, five-point
0.5 s reference, logical async packet timing, DSLPID, reward, and environment.

## Inputs

* B0: current latent16 + goal3 + ranges8 (27-D).
* B1: B0 plus direct stack of eight 16-D environmental frames.
* B2: B0 plus one-layer GRU(8x16, hidden32).
* Each 16-D frame is goal3 + existing world-fixed range rays8 + normalized
  velocity3 + yaw sin/cos2.  The frozen Encoder's existing action history is
  not duplicated.

The first seven high-level decisions carry repeat-first-frame padding.  To
avoid calling this real temporal information, first-error diagnostics below
begin only at high-level step 56 (0.467 s), when all eight entries came from
distinct high-level observation times.

## Offline metrics

| Group | Test position RMSE | Test velocity RMSE |
|---|---:|---:|
| B0 | 0.0288 m | 0.1210 m/s |
| B1 | 0.0265 m | 0.1235 m/s |
| B2 | 0.0288 m | 0.1260 m/s |

Small offline differences remain insufficient evidence for a closed-loop gain.

## Closed-loop result: five fixed seeds per scenario

| Group | clear | single | forest24 success | forest24 collision | final distance | min clearance |
|---|---:|---:|---:|---:|---:|---:|
| B0 instant | 5/5 | 5/5 | 0/5 | 5/5 | 5.194 m | 0.0011 m |
| B1 history stack | 0/5 | 0/5 | 0/5 | 5/5 | 5.820 m | 0.0010 m |
| B2 GRU | 5/5 | 5/5 | 0/5 | 4/5 | 4.328 m | 0.0162 m |

B1 is rejected as a viable historical-input design because it loses the
clear/single baseline.  B2 preserves that baseline and shows a modest forest
safety/progress movement, but it does not reach the goal in any forest seed.

## Post-history first-error and OOD diagnostic

| Group | first-error ambiguity after 0.467 s | forest contextual OOD |
|---|---:|---:|
| B0 | 5/5 | 19.11% |
| B1 | 5/5 | 27.59% |
| B2 | 5/5 | 27.70% |

For B0, four seeds first meet the ambiguity criterion at 0.467 s and one at
1.20 s.  For B1 and B2, all five first meet it at 0.467 s.  The OOD value is
calibrated independently within each representation space, so it is a
diagnostic rather than a cross-dimensional absolute distance; nevertheless B2
does not show the expected decrease.

## Interpretation and decision

The hypothesis "an explicit 0.5 s sparse-range/goal/velocity/yaw history
removes the forest's decisive observation ambiguity" is **not supported**.
The GRU history is computationally cheap and keeps simple navigation working,
but after real history exists it neither reduces ambiguity count nor reduces
closed-loop OOD, and forest success remains zero.

Therefore do not start DAgger yet.  DAgger would add labels at states whose
available 0.5 s observation history still maps to conflicting teacher routes;
it would not establish that the missing information has been supplied.  The
next justified experiment is stronger *environmental memory* (for example an
ego-pose accumulated local occupancy representation built from the existing
eight rays), with B0 and B2 retained as fixed controls.  This does not require
RGB/depth, reward changes, PID changes, or unfreezing the Encoder.

## Runtime note

The runtime async worker, IPC, packet buffer and PID were not modified.  In
the logical evaluation B2 planner inference p95 is 1.20--1.36 ms, below the
8.33 ms lower-loop compute budget; this is not a new spawned-worker timing
certificate.
