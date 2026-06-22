# AvoidBench Plain TD3 Smoke Analysis

Run: `runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke`

## Executive conclusion

- 0/25 completed episodes ended because of height bounds.
- Maximum raw actor saturation was 0.00% on vx; no near-limit collapse is indicated.
- Per-step action, height, vertical velocity, reward terms, and terminal context are available for this run.
- This run provides a stable-control measurement with full per-step diagnostics.

## Episode statistics

- completed episodes: 25
- episode length mean/median/min/max: 200.0000 / 200.0000 / 200.0000 / 200.0000
- return mean/median/min/max: -2.9628 / -2.5108 / -11.9117 / -2.3635
- done reasons: `{"timeout": 25}`
- height-bound terminations: 0 (0.00%)
- collisions: 0
- reset retry mean/max: 0.0000 / 0.0000

## Action and optimization signals

- critic loss mean/max: 0.0012 / 0.0098
- mean step time mean/max: 0.4513 / 0.4514

| dimension | mean | std | min | max | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| vx | -0.0153 | 0.0526 | -0.2088 | 0.2311 | 0.00% |
| vy | 0.1775 | 0.0718 | -0.3237 | 0.3935 | 0.00% |
| vz | -0.0058 | 0.0258 | -0.2020 | 0.1324 | 0.00% |
| yaw_rate | -0.0735 | 0.0821 | -0.4321 | 0.0794 | 0.00% |

Most saturated dimension: `vx`.

## Height and terminal context

- height mean/min/max: 1.2004 / 1.1943 / 1.2298
- vertical velocity mean/std/min/max: -0.0000 / 0.0010 / -0.0165 / 0.0139
- recorded height-terminal windows: 0

## Distance behavior

- final distance mean/min/max: 5.0085 / 4.9649 / 5.0462
- first completed episode final distance: 4.9649
- last completed episode final distance: 4.9984
- first-to-last final-distance improvement: -0.0335
- per-step progress mean/std: -0.0000 / 0.0008

Equal episode horizons are required before interpreting final-distance changes as navigation learning.

## Analysis conclusion

- 0/25 completed episodes ended because of height bounds.
- Maximum raw actor saturation was 0.00% on vx; no near-limit collapse is indicated.
- Per-step action, height, vertical velocity, reward terms, and terminal context are available for this run.
