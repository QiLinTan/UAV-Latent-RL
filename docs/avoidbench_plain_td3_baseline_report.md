# AvoidBench Lowdim Plain TD3 Baseline Report

## Scope

This report compares:

1. old smoke: `runs/avoidbench_plain_td3_smoke/20260609-051809/`
2. hover smoke: `runs/avoidbench_plain_td3_smoke/20260609-133209-hover_smoke/`
3. navigation smoke: `runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke/`

All new runs use low-dimensional observations and the high-level
`/hummingbird/autopilot/velocity_command` interface. No depth, latent encoder,
auxiliary loss, or motor RPM control was added.

## Gate results

| Metric | Old smoke | Hover smoke | Navigation smoke |
| --- | ---: | ---: | ---: |
| Steps | 2,000 | 5,000 | 5,000 |
| Completed episodes | 43 | 25 | 25 |
| Mean episode length | 45.30 | 200.00 | 200.00 |
| Height terminations | 41/43 (95.35%) | 0/25 (0%) | 0/25 (0%) |
| Collisions | 0 | 0 | 0 |
| Reset retries, mean | 0.140 | 0 | 0 |
| Raw actor saturation | unavailable per dimension | 0% on all dimensions | 0% on all dimensions |
| Mean height | unavailable | 1.2006 m | 1.2004 m |
| Height range | unavailable | 1.1958-1.2252 m | 1.1943-1.2298 m |
| Mean final distance | 4.4757 m, unequal horizons | 5.0104 m | 5.0085 m |
| Mean distance improvement | unavailable | -0.0104 m | -0.0085 m |
| Positive-progress episodes | unavailable | 3/25 | 8/25 |
| TD3 updates | 1,900 | 4,900 | 4,900 |

The hover gate passed clearly, so navigation smoke was allowed. Navigation
control stability passed, but navigation learning did not.

## Action interface and presets

The action dimensions are:

1. `vx`
2. `vy`
3. `vz`
4. `yaw_rate`

The centralized presets are:

| Preset | vx | vy | vz | yaw_rate |
| --- | ---: | ---: | ---: | ---: |
| `legacy` | 0.20 | 0.20 | 0.15 | 0.30 |
| `conservative` | 0.12 | 0.12 | 0.04 | 0.12 |

The trainer now records raw actor output, exploration noise, normalized action
before and after clipping, scaled velocity command, per-dimension saturation,
vertical velocity, height error, and reward terms on every step.

## Height reward and termination

The navigation defaults are:

- progress scale: `1.0`
- height-error penalty: `0.5 * abs(z - target_height)`
- vertical-velocity penalty: `0.2 * abs(vz)`
- normalized z-action penalty: `0.1 * abs(action_z)`
- normalized action penalty: `0.02 * ||action||^2`
- collision penalty: `5.0`
- goal bonus: `5.0`
- target height: `1.2 m`
- severe height termination: below `0.4 m` or above `2.5 m`

`hover_smoke` disables progress/goal incentives and increases height,
vertical-velocity, z-action, speed, and action regularization. Small height
errors receive continuous penalties; only severe deviations terminate.

## Questions answered

### 1. Why did the old smoke fail?

The dominant operational failure was height: 41 of 43 completed episodes ended
at a height bound. The aggregate actor standard deviation approached `1.0`
after roughly 400 steps, which is consistent with tanh outputs collapsing
toward normalized action limits. The old logger did not record per-dimension
actions or vertical state, so it cannot prove which dimension initiated the
failure.

### 2. Which action dimension saturated most?

This is not recoverable for the old run. In both new runs no raw actor dimension
saturated. The largest new action variability was `yaw_rate` in hover and
`yaw_rate`/`vy` in navigation, but neither was near saturation.

### 3. What caused the height failures?

The evidence points to a combination of actor collapse, the large legacy
vertical command range, and insufficient continuous height/vertical
regularization. The old done bounds were already broad (`0.4-3.0 m`), so an
overly strict termination threshold is not a credible primary cause.

Action scaling and reward shaping changed together, so this experiment cannot
separate their individual causal contribution. A later ablation would be
required for that claim.

### 4. Did the conservative preset reduce saturation?

Yes at the combined-system level. New raw actor saturation was `0%` for all
four dimensions across 4,900 actor-controlled steps in each run. The old
per-dimension percentage is unavailable, but its aggregate action standard
deviation averaged `0.903` and repeatedly approached `1.0`.

### 5. Did height shaping reduce height termination?

The combined conservative-action plus height-reward intervention reduced
height termination from `95.35%` to `0%` in both new runs. This is strong
engineering evidence that the control baseline is fixed, but not an isolated
reward ablation.

### 6. Is hover smoke stable?

Yes for this 5,000-step, single-seed gate:

- all 25 episodes reached the 200-step timeout
- no collision, reset retry, or runtime failure occurred
- mean height was `1.2006 m`
- raw actor saturation was `0%`

### 7. Is navigation smoke worth continuing?

The runtime and control loop are stable enough for further Plain TD3
experiments. The learned behavior is not yet useful navigation:

- mean distance improvement was `-0.0085 m`
- only 8 of 25 episodes had positive progress
- all episodes timed out
- no goal was reached

The policy mainly learned low-action stable flight rather than forward
navigation.

### 8. Is this ready as a latent baseline?

No. It is now a valid stable-control baseline, but not yet a navigation
baseline. Adding latent observations now would confound representation quality
with an unresolved navigation objective.

### 9. What remains missing?

- reproducible positive distance progress across multiple seeds
- at least occasional goal completion in the low-speed task
- validation that progress reward dominates the incentive to remain still
- observation/reward scale diagnostics for goal delta and action penalties
- evaluation runs separated from exploration training

### 10. What should happen next?

Continue with lowdim Plain TD3 only. Tune the navigation objective and scaling,
especially progress strength, directional velocity incentives, action penalty
balance, and observation normalization. Re-run short multi-seed navigation
smokes before considering depth or latent work.

## Decision

- Longer hover training: allowed but not necessary for the next diagnosis.
- Further Plain TD3 navigation tuning: allowed.
- Long navigation training with the current reward: not justified.
- Latent-only/depth/auxiliary work: blocked by the navigation-learning gap.
