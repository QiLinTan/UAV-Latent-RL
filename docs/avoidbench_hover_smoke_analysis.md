# AvoidBench Plain TD3 Smoke Analysis

Run: `runs/avoidbench_plain_td3_smoke/20260609-133209-hover_smoke`

## Executive conclusion

- 0/25 completed episodes ended because of height bounds.
- Maximum raw actor saturation was 0.00% on vx; no near-limit collapse is indicated.
- Per-step action, height, vertical velocity, reward terms, and terminal context are available for this run.
- This run provides a stable-control measurement with full per-step diagnostics.

## Episode statistics

- completed episodes: 25
- episode length mean/median/min/max: 200.0000 / 200.0000 / 200.0000 / 200.0000
- return mean/median/min/max: -5.6880 / -4.7671 / -27.6280 / -4.3375
- done reasons: `{"timeout": 25}`
- height-bound terminations: 0 (0.00%)
- collisions: 0
- reset retry mean/max: 0.0000 / 0.0000

## Action and optimization signals

- critic loss mean/max: 0.0033 / 0.0216
- mean step time mean/max: 0.4513 / 0.4523

| dimension | mean | std | min | max | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| vx | -0.0572 | 0.0617 | -0.1917 | 0.2459 | 0.00% |
| vy | -0.0454 | 0.0508 | -0.4854 | 0.0464 | 0.00% |
| vz | 0.0047 | 0.0218 | -0.2525 | 0.0723 | 0.00% |
| yaw_rate | -0.0561 | 0.1092 | -0.2415 | 0.1872 | 0.00% |

Most saturated dimension: `vx`.

## Height and terminal context

- height mean/min/max: 1.2006 / 1.1958 / 1.2252
- vertical velocity mean/std/min/max: -0.0000 / 0.0009 / -0.0132 / 0.0118
- recorded height-terminal windows: 0

## Distance behavior

- final distance mean/min/max: 5.0104 / 4.9003 / 5.0645
- first completed episode final distance: 4.9003
- last completed episode final distance: 5.0645
- first-to-last final-distance improvement: -0.1642
- per-step progress mean/std: -0.0001 / 0.0008

Equal episode horizons are required before interpreting final-distance changes as navigation learning.

## Analysis conclusion

- 0/25 completed episodes ended because of height bounds.
- Maximum raw actor saturation was 0.00% on vx; no near-limit collapse is indicated.
- Per-step action, height, vertical velocity, reward terms, and terminal context are available for this run.
