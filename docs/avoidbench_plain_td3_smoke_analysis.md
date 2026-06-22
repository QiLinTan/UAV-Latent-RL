# AvoidBench Plain TD3 Smoke Analysis

Run: `runs/avoidbench_plain_td3_smoke/20260609-051809`

## Executive conclusion

- 41/43 completed episodes ended because of height bounds.
- The aggregate actor std approached 1.0 after step 400, consistent with normalized tanh outputs collapsing toward action limits.
- The old run cannot distinguish which action dimension caused the height failures because it did not log per-step per-dimension actions, height, or vertical velocity.
- This run proves the training infrastructure worked, but it is not a stable control baseline.

## Episode statistics

- completed episodes: 43
- episode length mean/median/min/max: 45.3023 / 27.0000 / 26.0000 / 200.0000
- return mean/median/min/max: -4.1375 / -3.5029 / -24.0644 / 0.6661
- done reasons: `{"height_too_high": 36, "height_too_low": 5, "out_of_bounds": 1, "timeout": 1}`
- height-bound terminations: 41 (95.35%)
- collisions: 0
- reset retry mean/max: 0.1395 / 1.0000

## Action and optimization signals

- critic loss mean/max: 0.0058 / 0.0123
- mean step time mean/max: 0.4513 / 0.4514
- aggregate actor mean: -0.1340
- aggregate actor std mean/max: 0.9028 / 1.0000

Per-dimension action statistics are unavailable because the run did not record per-step actor outputs.

## Height and terminal context

Height, vertical velocity, and the ten steps before each height termination are unavailable because this run has no `steps.jsonl`.

## Distance behavior

- final distance mean/min/max: 4.4757 / 3.1756 / 10.9370
- first completed episode final distance: 3.7191
- last completed episode final distance: 4.4055
- first-to-last final-distance improvement: -0.6864

Equal episode horizons are required before interpreting final-distance changes as navigation learning.

## Analysis conclusion

- 41/43 completed episodes ended because of height bounds.
- The aggregate actor std approached 1.0 after step 400, consistent with normalized tanh outputs collapsing toward action limits.
- The old run cannot distinguish which action dimension caused the height failures because it did not log per-step per-dimension actions, height, or vertical velocity.
