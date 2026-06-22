# AvoidBench Navigation Smoke Analysis

Run: `runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke`

## Executive conclusion

- Episodes: 25
- Done reasons: `{"timeout": 25}`
- Mean distance delta: `-0.008545 m`
- Positive-progress episodes: `8/25`
- Most active scaled action dimension: `yaw_rate`
- Mean raw actor action vx/vy: `-0.015252` / `0.177500`

The run is stable but not a navigation baseline. It mostly hovers near
the reset height while commanding sideways velocity more strongly than
goal-directed `+vx`.

## Episode and distance statistics

- episode length mean/min/max: `200.0` / `200.0` / `200.0`
- return mean/min/max: `-2.9627912243528653` / `-11.91169926130917` / `-2.363524717910318`
- initial distance mean: `4.999946384429932`
- final distance mean: `5.008491840362549`
- distance delta mean/min/max: `-0.008545455932617187` / `-0.04625225067138672` / `0.03514719009399414`
- per-step progress mean/std: `-4.2727279663085935e-05` / `0.0007513555223029652`

## Height and safety

- collision count: `0`
- height mean/min/max: `1.200355242395401` / `1.1943068504333496` / `1.2298325300216675`
- vertical velocity mean/std/min/max: `-4.58125521799027e-05` / `0.000953861997862173` / `-0.01650121435523033` / `0.013927972875535488`
- stable-hover-without-progress episodes: `25`

## Action diagnostics

| dimension | raw mean | raw std | scaled mean | scaled std | normalized saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| vx | -0.015252 | 0.052647 | -0.001980 | 0.016539 | 0.12% |
| vy | 0.177500 | 0.071830 | 0.021034 | 0.016962 | 0.14% |
| vz | -0.005811 | 0.025796 | -0.000230 | 0.005351 | 0.04% |
| yaw_rate | -0.073473 | 0.082114 | -0.008606 | 0.018224 | 0.10% |

The actor does not saturate. The main issue is direction: the learned
mean actor output is biased toward `+vy`, while the goal delta indicates
the target is primarily in `+x`.

## Reward balance

- sum progress reward: `-0.2136363983154297`
- sum abs progress reward: `1.3330440521240234`
- sum regularization penalties: `61.356144251537366`
- abs progress share of progress plus regularization: `0.021264337411211406`
- z-action penalty mean: `0.009128412419867237`
- action penalty mean: `0.002322174695775169`

The progress signal is very small. Regularization, especially z-action
and action penalties, is large enough to make low-motion behavior an
attractive early solution.

## Observation and frame checks

- goal delta observation slice: `[13, 16]`
- mean goal delta: `[5.002854643821716, -0.040375476463415265, -0.00035519471168518065]`
- distance consistency abs error max: `4.014236427707374e-07`

Observation indices 13:16 contain goal_position - position. For this run the target is mostly +x from reset, so positive vx is the direct low-level command to test before more training.

The goal delta is present and numerically consistent with
`distance_to_goal`. The unresolved question is command frame: the launch
sets `velocity_estimate_in_world_frame=false`, so a live hand-coded
policy must verify whether world-frame or body-frame goal direction
actually reduces distance.

## Front/back 20-step comparison

- first-window progress mean: `3.151416778564453e-05`
- last-window progress mean: `-6.449031829833985e-05`

There is no clear late-episode improvement. The policy remains stable
but does not increasingly point toward the goal.

## Diagnosis

- most likely cause: The policy learned stable low-action flight but did not learn to command +vx toward the +x goal. The strongest actor bias is +vy, while mean vx is slightly negative.
- coordinate/action mapping suspicion: Still open. The observation goal delta is present and numerically consistent, but the live action frame must be verified with a hand-coded goal-direction policy.
- reward weight suspicion: High. The absolute progress signal is tiny compared with z-action/action regularization, so hovering or sideways low-risk commands can dominate early learning.
- task difficulty suspicion: Moderate. A 5 m goal over 200 low-speed steps is not extreme, but it is too complex to debug before proving hand-coded goal-direction progress.

## Decision

Do not continue longer navigation training yet. Run the
`constant-goal-direction` hand-coded policy probe first. If that probe
cannot reliably lower distance, fix action frame, goal delta, or distance
calculation before Stage 1 training. Do not start latent work.
